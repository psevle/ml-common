"""Unified dataloader factory for creating train/val dataloaders."""

from torch.utils.data import DataLoader, RandomSampler
from typing import Dict, Any, Tuple

from .mmap import MmapDataset
from .kaggle import KaggleDataset, load_sensor_geometry, get_icecube_file_names
from .i3 import I3IterableDataset, ICECUBE_AVAILABLE
from .i3_noise import I3NoiseIterableDataset
from ..utils.collators import IrregularDataCollator
from ..utils.samplers import RandomChunkSampler


def create_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from config.

    Supports unified mmap format, legacy Kaggle format, and direct I3 streaming.
    Auto-detects format and creates appropriate datasets.

    Args:
        cfg: Configuration dict with data_options and training_options

    Returns:
        (train_loader, val_loader)

    Config format:
        data_options:
            # Option 1: Single path with runtime splitting (mmap only)
            data_path: "/path/to/data"
            val_split: 0.2
            split_seed: 42

            # Option 2: Separate train/val paths
            train_data_path: "/path/to/train"
            valid_data_path: "/path/to/val"

            # Kaggle-specific (requires dataloader: "kaggle")
            geometry_path: "/path/to/geometry.csv"
            train_batch_dir: "/path/to/train_batches"
            train_batch_ranges: [0, 100]
            ...

            # I3 streaming (requires dataloader: "i3")
            train_files: ["/path/to/train/*.i3.zst", ...]
            valid_files: ["/path/to/val/*.i3.zst", ...]
            geometry_path: "resources/GeoCalibDetectorStatus_IC86.AVG_Pass2_SF0.99.i3.gz"
            pulse_key: "SRTInIcePulses"
            filter_names: ["MuonFilter_13", "SunFilter_13"]
            mix_n_files: 4
            sub_event_stream: "InIceSplit"

            # Common options
            use_summary_stats: true
            batch_size: 256  # or in training_options
            num_workers: 8   # or in training_options

        dataloader: "mmap"  # or "kaggle" (mmap auto-detects prometheus/icecube)
    """
    data_options = cfg['data_options']
    dataloader_type = cfg['dataloader']
    task = cfg.get('task') or data_options.get('task')

    if dataloader_type == 'i3':
        if not ICECUBE_AVAILABLE:
            raise ImportError('I3 dataloader requires the IceCube python modules.')

        required_filters = data_options.get('filter_names', [])
        mix_n_files = data_options.get('mix_n_files', 4)
        seed = data_options.get('seed')

        train_dataset = I3IterableDataset(
            files=data_options['train_files'],
            geometry_path=data_options['geometry_path'],
            pulse_key=data_options.get('pulse_key', 'SRTInIcePulses'),
            primary_key=data_options.get('primary_key', 'PolyplopiaPrimary'),
            filter_key=data_options.get('filter_key', 'FilterMask'),
            required_filters=required_filters,
            sub_event_stream=data_options.get('sub_event_stream', 'InIceSplit'),
            mix_n_files=mix_n_files,
            shuffle_files=data_options.get('shuffle_files', True),
            random_seed=seed,
        )

        valid_dataset = I3IterableDataset(
            files=data_options['valid_files'],
            geometry_path=data_options['geometry_path'],
            pulse_key=data_options.get('pulse_key', 'SRTInIcePulses'),
            primary_key=data_options.get('primary_key', 'PolyplopiaPrimary'),
            filter_key=data_options.get('filter_key', 'FilterMask'),
            required_filters=required_filters,
            sub_event_stream=data_options.get('sub_event_stream', 'InIceSplit'),
            mix_n_files=max(1, mix_n_files),
            shuffle_files=data_options.get('shuffle_files', True),
            random_seed=seed,
        )

        train_sampler = None
        val_sampler = None
    
    elif dataloader_type == 'i3noise':
        if not ICECUBE_AVAILABLE:
            raise ImportError('I3 dataloader requires the IceCube python modules.')

        required_filters = data_options.get('filter_names', [])
        mix_n_files = data_options.get('mix_n_files', 4)
        seed = data_options.get('seed')

        train_dataset = I3NoiseIterableDataset(
            files=data_options['train_files'],
            use_summary_stats=data_options['use_summary_stats'],
            geometry_path=data_options['geometry_path'],
            pulse_key=data_options.get('pulse_key', 'OfflineInIcePulses'),
            primary_key=data_options.get('primary_key', 'I3MCTree'),
            filter_key=data_options.get('filter_key', 'FilterMask'),
            required_filters=required_filters,
            sub_event_stream=data_options.get('sub_event_stream', None),
            mix_n_files=mix_n_files,
            shuffle_files=data_options.get('shuffle_files', True),
            random_seed=seed,
        )

        valid_dataset = I3NoiseIterableDataset(
            files=data_options['valid_files'],
            geometry_path=data_options['geometry_path'],
            pulse_key=data_options.get('pulse_key', 'OfflineInIcePulses'),
            primary_key=data_options.get('primary_key', 'I3MCTree'),
            filter_key=data_options.get('filter_key', 'FilterMask'),
            required_filters=required_filters,
            sub_event_stream=data_options.get('sub_event_stream', None),
            mix_n_files=max(1, mix_n_files),
            shuffle_files=data_options.get('shuffle_files', True),
            random_seed=seed,
        )

        train_sampler = None
        val_sampler = None

    else:
        # Single path with runtime splitting
        if 'data_path' in data_options:
            val_split = data_options.get('val_split', 0.2)
            split_seed = data_options.get('split_seed', 42)

            if dataloader_type == 'kaggle':
                raise ValueError("Runtime splitting not supported for Kaggle datasets. Use separate paths.")

            train_dataset = MmapDataset(
                mmap_paths=data_options['data_path'],
                use_summary_stats=data_options.get('use_summary_stats', True),
                split="train",
                val_split=val_split,
                split_seed=split_seed,
                task=task
            )

            valid_dataset = MmapDataset(
                mmap_paths=data_options['data_path'],
                use_summary_stats=data_options.get('use_summary_stats', True),
                split="val",
                val_split=val_split,
                split_seed=split_seed,
                task=task
            )

        else:
            # Separate train/valid paths
            if dataloader_type == 'kaggle':
                sensor_geometry = load_sensor_geometry(data_options['geometry_path'])

                train_batch_files = get_icecube_file_names(
                    data_options['train_batch_dir'],
                    data_options['train_batch_ranges'],
                    data_options.get('shuffle_files', False)
                )
                valid_batch_files = get_icecube_file_names(
                    data_options['valid_batch_dir'],
                    data_options['valid_batch_ranges'],
                    data_options.get('shuffle_files', False)
                )

                train_dataset = KaggleDataset(
                    meta_dir=data_options['train_meta_dir'],
                    batch_files=train_batch_files,
                    sensor_geometry=sensor_geometry,
                    use_summary_stats=data_options.get('use_summary_stats', True)
                )

                valid_dataset = KaggleDataset(
                    meta_dir=data_options['valid_meta_dir'],
                    batch_files=valid_batch_files,
                    sensor_geometry=sensor_geometry,
                    use_summary_stats=data_options.get('use_summary_stats', True)
                )
            else:
                # Unified mmap format
                train_dataset = MmapDataset(
                    mmap_paths=data_options['train_data_path'],
                    use_summary_stats=data_options.get('use_summary_stats', True),
                    task=task,
                )

                valid_dataset = MmapDataset(
                    mmap_paths=data_options['valid_data_path'],
                    use_summary_stats=data_options.get('use_summary_stats', True),
                    task=task,
                )

        train_len = len(train_dataset)
        if train_len == 0:
            train_sampler = None
            val_sampler = RandomChunkSampler(valid_dataset, valid_dataset.chunks) if dataloader_type == 'kaggle' else RandomSampler(valid_dataset)
            print("Warning: Training split is empty")
        elif dataloader_type == 'kaggle':
            train_sampler = RandomChunkSampler(train_dataset, train_dataset.chunks)
            val_sampler = RandomChunkSampler(valid_dataset, valid_dataset.chunks)
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = RandomSampler(valid_dataset)

    # Get batch size and num_workers
    batch_size = data_options.get('batch_size') or cfg['training_options']['batch_size']
    num_workers = data_options.get('num_workers') or cfg['training_options'].get('num_workers', 0)

    # Create dataloaders
    collate = IrregularDataCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        collate_fn=collate
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        collate_fn=collate
    )

    def _safe_len(dataset):
        try:
            return len(dataset)
        except TypeError:
            return None

    train_len = _safe_len(train_dataset)
    val_len = _safe_len(valid_dataset)

    train_desc = f"{train_len:,}" if train_len is not None else "iterable"
    val_desc = f"{val_len:,}" if val_len is not None else "iterable"

    print(f"Created dataloaders: train={train_desc}, val={val_desc}")

    return train_loader, val_loader
