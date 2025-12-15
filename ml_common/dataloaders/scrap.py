from typing import Tuple, Dict
import numpy as np
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.compute as pc


def arrow_table_to_structured_array(table: pa.Table) -> np.ndarray:
    """
    Convert a PyArrow Table to a NumPy structured array.
    """
    arrays = [col.to_numpy(zero_copy_only=False) for col in table.columns]
    names = table.column_names

    dtype = [(name, arr.dtype) for name, arr in zip(names, arrays)]

    structured = np.zeros(len(arrays[0]), dtype=dtype)
    for name, arr in zip(names, arrays):
        structured[name] = arr

    return structured


def load_pq(
    cluster_file: str,
    pulses_file: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, slice]]:
    """
    """

    cluster_ds = ds.dataset(cluster_file, format="parquet")
    clusters_tbl = cluster_ds.to_table()

    clusters_tbl = clusters_tbl.set_column(
        clusters_tbl.get_column_index("id"),
        "id",
        pc.cast(clusters_tbl["id"], pa.string())
    )

    id_array = clusters_tbl["id"]
    unique_ids = pc.unique(id_array).to_pylist()

    pulse_ds = ds.dataset(pulses_file, format="parquet")
    pulse_tbl = pulse_ds.to_table(
        filter=ds.field("id").isin(unique_ids)
    )

    pulse_tbl = pulse_tbl.set_column(
        pulse_tbl.get_column_index("id"),
        "id",
        pc.cast(pulse_tbl["id"], pa.string())
    )

    clusters_np = arrow_table_to_structured_array(clusters_tbl)
    pulses_np = arrow_table_to_structured_array(pulse_tbl)

    sort_idx = np.argsort(pulses_np["id"])
    pulses_np = pulses_np[sort_idx]

    index_map: Dict[str, slice] = {}

    if len(pulses_np) > 0:
        unique_ids_sorted, start_positions = np.unique(
            pulses_np["id"], return_index=True
        )

        for i, uid in enumerate(unique_ids_sorted):
            start = start_positions[i]
            end = (
                start_positions[i + 1]
                if i + 1 < len(start_positions)
                else len(pulses_np)
            )
            index_map[uid] = slice(start, end)

    return clusters_np, pulses_np, index_map


if __name__ == "__main__":
    load_pq("/Users/myhr/Documents/ELESDE/files/clusters/IC86.2020.genie_NuMu.023255.000000.parquet",
            "/Users/myhr/Documents/ELESDE/files/pulses/IC86.2020.genie_NuMu.023255.000000.parquet")