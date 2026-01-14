"""Generic trainer class for PyTorch models."""

import csv
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext


class Trainer:
    """
    Generic trainer for PyTorch models with mixed precision, checkpointing, and logging.

    Supports both W&B and CSV logging. Provides flexible loss and metric computation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        cfg: Dict[str, Any],
        loss_fn: Optional[Callable] = None,
        metric_fn: Optional[Callable] = None,
        batch_prep_fn: Optional[Callable] = None,
        use_wandb: bool = False
    ):
        """
        Initialize Trainer.

        Args:
            model: PyTorch model
            device: Device to train on
            cfg: Config dict with training_options and project_save_dir
            loss_fn: Loss function (preds, labels) -> loss
            metric_fn: Optional metric function (preds, labels) -> dict
            batch_prep_fn: Optional batch preparation (coords, features, labels) -> (coords, features, batch_ids, labels)
            use_wandb: Whether to use W&B logging
        """
        self.model = model
        self.device = device
        self.cfg = cfg
        self.use_wandb = use_wandb
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.batch_prep_fn = batch_prep_fn or self._default_batch_prep

        # Extract training options
        training_opts = cfg['training_options']
        self.epochs = training_opts['epochs']
        self.lr = training_opts['lr']
        self.weight_decay = training_opts['weight_decay']
        self.batch_size = training_opts['batch_size']
        self.precision = training_opts.get('precision', 'fp32')
        self.save_epochs = training_opts.get('save_epochs', 5)
        self.grad_clip = training_opts.get('grad_clip', 1.0)

        # Setup optimizer, scheduler, mixed precision, and logging
        self.optimizer, self.scheduler = self._setup_optimizer_and_scheduler(training_opts)
        self.scaler = self._setup_mixed_precision()
        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')

    @staticmethod
    def _precision_to_dtype(precision: str) -> Optional[torch.dtype]:
        """Convert precision string to dtype. Only accepts 'fp16', 'bf16', 'fp32'."""
        precision = precision.lower()
        if precision == 'fp16':
            return torch.float16
        elif precision == 'bf16':
            return torch.bfloat16
        elif precision == 'fp32':
            return None  # No autocast for fp32
        else:
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of: 'fp16', 'bf16', 'fp32'"
            )

    def _setup_optimizer_and_scheduler(self, training_opts: Dict[str, Any]) -> Tuple:
        """Setup optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Default: cosine annealing that completes one cycle over all epochs
        T_max = training_opts.get('T_max', self.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return optimizer, scheduler

    def _setup_mixed_precision(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Setup mixed precision training."""
        self.amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.amp_dtype = self._precision_to_dtype(self.precision)

        # Fall back to fp16 if bf16 not supported on CUDA
        if self.amp_device == 'cuda' and self.amp_dtype is torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                print('Warning: CUDA bf16 not supported. Falling back to fp16.')
                self.amp_dtype = torch.float16

        # Only use scaler for CUDA fp16
        if self.amp_device == 'cuda' and self.amp_dtype is torch.float16:
            return torch.amp.GradScaler(self.amp_device)
        return None

    def _setup_logging(self):
        """Setup logging directories and CSV writer."""
        self.save_dir = Path(self.cfg['project_save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        if not self.use_wandb:
            self.csv_file = self.save_dir / 'metrics.csv'
            self.csv_writer = None
            self.csv_file_handle = None

    def _get_autocast_context(self, precision: str = None):
        """Get autocast context manager for the given precision."""
        if precision is None:
            precision = self.precision

        dtype = self._precision_to_dtype(precision)
        device = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Fall back to fp16 if bf16 not supported
        if device == 'cuda' and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            dtype = torch.float16

        return torch.amp.autocast(device, dtype=dtype) if dtype is not None else nullcontext()

    def _default_batch_prep(
        self, coords_b: torch.Tensor, features_b: torch.Tensor, labels_b: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Default batch preparation: extract batch indices from coords."""
        assert coords_b.dim() == 2 and coords_b.size(-1) >= 4
        batch_ids = coords_b[:, 0].long()
        coords = coords_b[:, 1:4]
        feats = features_b if (features_b is not None and features_b.numel() > 0) else None
        return coords, feats, batch_ids, labels_b

    def _init_csv_writer(self, metrics: Dict[str, float]):
        """Initialize CSV writer with appropriate fields."""
        base_fields = ['epoch', 'step', 'train_loss', 'val_loss', 'learning_rate']
        additional = [k for k in metrics.keys() if k not in base_fields]
        all_fields = base_fields + additional

        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file_handle, fieldnames=all_fields, extrasaction='ignore'
        )
        self.csv_writer.writeheader()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B or CSV."""
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except ImportError:
                pass
        else:
            if self.csv_writer is None:
                self._init_csv_writer(metrics)

            row = {'epoch': self.current_epoch, 'step': step or 0, **metrics}
            self.csv_writer.writerow(row)
            self.csv_file_handle.flush()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided")

        self.model.train()
        running_loss = 0.0
        try:
            num_batches = len(train_loader)
        except TypeError:
            num_batches = None
        batches_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {self.current_epoch+1}/{self.epochs}',
            leave=True,
            dynamic_ncols=True,
            total=num_batches,
            position=1,
        )

        for batch_idx, (coords, features, labels) in enumerate(pbar):
            coords, features, labels = coords.to(self.device), features.to(self.device), labels.to(self.device)
            coords, feats, batch_ids, labels = self.batch_prep_fn(coords, features, labels)
            batches_seen += 1

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with self._get_autocast_context():
                preds = self.model(coords, feats, batch_ids=batch_ids)
                loss = self.loss_fn(preds, labels)

            # Backward pass with gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'Avg': f'{running_loss/(batch_idx+1):.6f}'
            })

            if self.current_step % 50 == 0:
                metrics = {
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }
                if self.use_wandb and hasattr(self.loss_fn, 'current_weights'):
                    vmf_weight, _ = self.loss_fn.current_weights()
                    metrics['vmf_weight'] = float(vmf_weight)
                self.log_metrics(metrics, step=self.current_step)
            self.current_step += 1

        avg_loss = running_loss / max(1, batches_seen)
        return {'train_loss': avg_loss}

    def _print_profiling_results(self, forward_times: list, total_time: float):
        """Print profiling statistics."""
        total_forward = sum(forward_times)
        avg_forward = total_forward / len(forward_times) if forward_times else 0

        print("\n--- Inference Profiling ---")
        print(f"Total runtime (incl. I/O): {total_time:.4f}s")
        print(f"Total forward time: {total_forward:.4f}s")
        if self.device.type == 'cuda':
            print(f"Avg forward time per batch: {avg_forward * 1000:.4f}ms")
            peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            print(f"Peak CUDA memory: {peak_mem_gb:.4f}GB")
        else:
            std_forward = np.std(forward_times) if forward_times else 0
            print(f"Avg forward time per batch: {avg_forward * 1000:.4f}ms, std: {std_forward * 1000:.4f}ms")
        print("---------------------------\n")

    def validate(
        self, val_loader: DataLoader, save_predictions: bool = False, profile: bool = False
    ) -> Dict[str, float]:
        """Run validation."""
        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided")

        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        forward_times = []

        if profile and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()

        try:
            val_total = len(val_loader)
        except TypeError:
            val_total = None

        with torch.no_grad():
            val_pbar = tqdm(
                val_loader,
                desc='Validation',
                leave=True,
                dynamic_ncols=True,
                total=val_total,
                position=2,
            )
            for coords, features, labels in val_pbar:
                coords, features, labels = coords.to(self.device), features.to(self.device), labels.to(self.device)
                coords, feats, batch_ids, labels = self.batch_prep_fn(coords, features, labels)

                with self._get_autocast_context():
                    if profile:
                        t0 = time.time()
                    preds = self.model(coords, feats, batch_ids=batch_ids)
                    if profile:
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        forward_times.append(time.time() - t0)

                    loss = self.loss_fn(preds, labels)
                total_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})

        if profile:
            self._print_profiling_results(forward_times, time.time() - start_time)

        batch_count = len(all_preds)
        avg_val_loss = total_loss / max(1, batch_count)

        metrics = {'val_loss': avg_val_loss}

        if batch_count == 0:
            if save_predictions:
                print('Warning: No validation batches processed; skipping prediction export.')
            return metrics

        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        if save_predictions:
            predictions_file = self.save_dir / 'results.npz'
            np.savez(predictions_file, predictions=preds_tensor.float().numpy(), labels=labels_tensor.float().numpy())
            print(f"Saved predictions to: {predictions_file}")

        if self.metric_fn is not None:
            task_metrics = self.metric_fn(preds_tensor, labels_tensor)
            metrics.update({f'val_{k}': v for k, v in task_metrics.items()})

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'cfg': self.cfg
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Regular checkpoint
        if self.current_epoch % self.save_epochs == 0 or is_final:
            if is_final:
                checkpoint_path = self.checkpoint_dir / f'final-checkpoint-{time.strftime("%Y%m%d-%H%M%S")}.pt'
            else:
                checkpoint_path = self.checkpoint_dir / f'epoch-{self.current_epoch:02d}-checkpoint-{time.strftime("%Y%m%d-%H%M%S")}.pt'

            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

            if self.use_wandb:
                try:
                    import wandb
                    artifact_name = f"{wandb.run.name or wandb.run.id}-checkpoint"
                    artifact = wandb.Artifact(artifact_name, type='model', metadata={'epoch': self.current_epoch, **metrics})
                    artifact.add_file(str(checkpoint_path))
                    wandb.log_artifact(artifact)
                except Exception as e:
                    print(f"Could not save wandb artifact: {e}")

        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'best-checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f'Best model saved: {best_path}')

    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = False):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']

            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train model."""
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        epoch_pbar = tqdm(
            range(self.current_epoch, self.epochs),
            desc='Training Progress',
            position=0,
            dynamic_ncols=True
        )

        for epoch in epoch_pbar:
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            epoch_metrics = {**train_metrics, **val_metrics}
            self.log_metrics(epoch_metrics)

            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']

            self.save_checkpoint(epoch_metrics, is_best)
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_metrics["train_loss"]:.6f}',
                'Val Loss': f'{val_metrics["val_loss"]:.6f}'
            })

        self.save_checkpoint({'best_val_loss': self.best_val_loss}, is_final=True)
        print(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")

    def test(self, test_loader: DataLoader):
        """Run test evaluation."""
        print("Running test evaluation...")
        test_metrics = self.validate(test_loader, save_predictions=True, profile=True)

        print("Test Results:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.6f}")

        self.log_metrics(test_metrics)

        return test_metrics
