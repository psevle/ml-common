"""PyTorch IterableDataset for IceCube I3 files."""

import glob
import os
import random
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

try:
    from icecube import dataio, dataclasses, icetray, simclasses  # type: ignore
    ICECUBE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    dataio = dataclasses = icetray = None  # type: ignore
    ICECUBE_AVAILABLE = False

try:
    import nt_summary_stats
    HAS_SUMMARY_STATS = True
except ImportError:
    nt_summary_stats = None
    HAS_SUMMARY_STATS = False

def _expand_paths(paths: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            expanded.extend(sorted(glob.glob(os.path.join(path, '*.i3*'))))
        else:
            matches = glob.glob(path)
            expanded.extend(sorted(matches) if matches else [path])
    return expanded


class I3NoiseIterableDataset(IterableDataset):
    """Stream events from I3 files with lightweight inter-file mixing."""

    def __init__(
        self,
        files: Sequence[str],
        geometry_path: str,
        use_summary_stats: bool = True,
        pulse_key: str = 'OfflineInIcePulses',
        primary_key: str = 'I3MCTree',
        filter_key: str = None,
        required_filters: Optional[Sequence[str]] = None,
        sub_event_stream: Optional[str] = None,
        mix_n_files: int = 4,
        shuffle_files: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not ICECUBE_AVAILABLE:
            raise ImportError('I3NoiseIterableDataset requires the IceCube python modules.')

        if use_summary_stats and not HAS_SUMMARY_STATS:
            raise ImportError("nt_summary_stats package is required for summary stats processing. Please do 'pip install nt--summary-stats'.")
        self.use_summary_stats = use_summary_stats and HAS_SUMMARY_STATS

        self.file_paths = _expand_paths(files)
        if not self.file_paths:
            raise ValueError('Resolved I3 file list is empty.')

        self.geometry_path = geometry_path
        self.pulse_key = pulse_key
        self.primary_key = primary_key
        self.filter_key = filter_key
        self.required_filters = list(required_filters or [])
        self.sub_event_stream = sub_event_stream
        self.mix_n_files = max(1, int(mix_n_files))
        self.shuffle_files = shuffle_files
        self.random_seed = random_seed

        self.om_positions = self._load_geometry()

    def _load_geometry(self) -> dict:
        geo_file = dataio.I3File(self.geometry_path)
        geometry = None
        while geo_file.more():
            frame = geo_file.pop_frame()
            if frame and frame.Has('I3Geometry'):
                geometry = frame['I3Geometry']
                break
        geo_file.close()

        if geometry is None:
            raise RuntimeError('Geometry not found in provided GCD file.')

        positions = {}
        for omkey, omgeo in geometry.omgeo.items():
            positions[omkey] = np.array(
                [omgeo.position.x, omgeo.position.y, omgeo.position.z],
                dtype=np.float32,
            )
        return positions

    def _per_worker_files(self) -> List[str]:
        files = list(self.file_paths)

        # global shuffle with a seed shared across workers
        if self.shuffle_files:
            seed = 0 if self.random_seed is None else self.random_seed
            rng = random.Random(seed)
            rng.shuffle(files)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return files
        return files[worker_info.id :: worker_info.num_workers]


    def _prepare_file_order(self, files: List[str]) -> List[str]:
        if not self.shuffle_files:
            return files
        rng = random.Random(self.random_seed)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and self.random_seed is not None:
            rng.seed(self.random_seed + worker_info.id)
        files = list(files)
        rng.shuffle(files)
        return files

    def _passes_filters(self, frame) -> bool:
        if not self.required_filters:
            return True
        if not frame.Has(self.filter_key):
            return False
        mask = frame[self.filter_key]
        for name in self.required_filters:  # any-pass semantics
            if name in mask:
                if mask[name].condition_passed:
                    return True
        return False

    def _physics_events(self, path: str) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        i3_file = dataio.I3File(path)
        try:
            while i3_file.more():
                frame = i3_file.pop_frame()
                if frame is None or frame.Stop != icetray.I3Frame.DAQ:
                    continue
                if frame.Has("I3MCPESeriesMapWithoutNoise") and len(frame["I3MCPESeriesMapWithoutNoise"]) == 0:
                    continue

                if self.sub_event_stream is not None:
                    if not frame.Has('I3EventHeader'):
                        continue
                    header = frame['I3EventHeader']
                    if header.sub_event_stream != self.sub_event_stream:
                        continue
                if self.required_filters and not self._passes_filters(frame):
                    continue

                pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, self.pulse_key)
                if len(pulse_map) == 0:
                    continue

                positions: List[np.ndarray] = []
                times: List[float] = []
                charges: List[float] = []
                for omkey, series in pulse_map.items():
                    dom_position = self.om_positions[omkey]
                    for pulse in series:
                        positions.append(dom_position)
                        times.append(pulse.time)
                        charges.append(pulse.charge)
                if not positions:
                    continue

                pos = np.asarray(positions, dtype=np.float32)
                t = np.asarray(times, dtype=np.float32)
                q = np.asarray(charges, dtype=np.float32)

                coords = np.concatenate([pos, t[:, None]], axis=1)  # (N,4)
                #features = np.stack([t, np.log1p(q)], axis=1)        # (N,2)

                coords *= 1e-3            # meters/ns -> km/µs
                
                if self.use_summary_stats:
                    stats = nt_summary_stats.compute_summary_stats(t, q)
                    features = np.log1p(stats)

                else:
                    features = np.stack([t, np.log1p(q)], axis=1)
                    features[:, 0] *= 1e-3    # ns -> µs

                primary = dataclasses.get_most_energetic_neutrino(frame[self.primary_key])
                ishit = 1 if len(frame["I3MCPESeriesMapWithoutNoise"]) > 0 else 0

                energy = primary.energy
                log_e = np.float32(np.log10(energy))
                dir_x = np.sin(primary.dir.zenith) * np.cos(primary.dir.azimuth)
                dir_y = np.sin(primary.dir.zenith) * np.sin(primary.dir.azimuth)
                dir_z = np.cos(primary.dir.zenith)
                # labels = np.array([log_e, dir_x, dir_y, dir_z, ishit], dtype=np.float32)
                labels = np.array([ishit], dtype=np.float32)

                yield coords.astype(np.float32, copy=False), features.astype(np.float32, copy=False), labels
        finally:
            i3_file.close()

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        files = self._per_worker_files()
        if not files:
            return
        files = self._prepare_file_order(files)

        # Per-worker RNG for stochastic interleaving
        rng = random.Random(self.random_seed)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and self.random_seed is not None:
            rng.seed(self.random_seed + worker_info.id)

        # Keep up to mix_n_files open at once; interleave randomly among active,
        # replacing an exhausted iterator with the next file.
        pending = list(files)
        active: List[Iterator] = []
        for _ in range(min(self.mix_n_files, len(pending))):
            active.append(self._physics_events(pending.pop()))

        rr_idx = 0  # deterministic round-robin index when shuffle_files=False
        while active:
            idx = rng.randrange(len(active)) if self.shuffle_files else (rr_idx % len(active))
            it = active[idx]
            sample = next(it, None)  # avoid try/except for StopIteration
            if sample is None:
                if pending:
                    active[idx] = self._physics_events(pending.pop())
                else:
                    active.pop(idx)
            else:
                if not self.shuffle_files:
                    rr_idx += 1
                yield sample
