#!/usr/bin/env python
"""
Train NanoTabPFN Regressor on the train split.

Usage:
    uv run python train_nanotabpfn_regressor.py
"""

import torch
from pathlib import Path

from pfns.bar_distribution import FullSupportBarDistribution
from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed, make_global_bucket_edges

# Set seed for reproducibility
set_randomness_seed(2402)

device = get_default_device()
print(f"Using device: {device}")

# Configuration
priordump_path = "tfm_playground/50x3_1280k_regression_train.h5"
output_dir = Path("models/nanotabpfn")
output_dir.mkdir(parents=True, exist_ok=True)

weights_path = output_dir / "regressor.pt"
buckets_path = output_dir / "regressor_buckets.pt"
checkpoint_path = output_dir / "regressor_latest.pt"

print(f"Training NanoTabPFN Regressor")
print(f"  Input: {priordump_path}")
print(f"  Model Output: {weights_path}")
print(f"  Buckets Output: {buckets_path}")

# Compute bucket edges from the entire dataset
print("\nComputing bucket edges...")
bucket_edges = make_global_bucket_edges(
    filename=priordump_path,
    n_buckets=100,
    device=device,
)
torch.save(bucket_edges, buckets_path)
print(f"✓ Bucket edges saved to {buckets_path}")

# Create distribution
dist = FullSupportBarDistribution(bucket_edges)

# Load prior data
prior = PriorDumpDataLoader(
    filename=priordump_path,
    num_steps=25,        # batches per epoch
    batch_size=50,       # datasets per batch
    device=device,
)

# Create model
model = NanoTabPFNModel(
    num_attention_heads=6,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=100,     # number of buckets for quantile regression
)

# Train with checkpointing
def save_checkpoint(epoch, model, loss):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'loss': loss
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

class CheckpointCallback(ConsoleLoggerCallback):
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        super().on_epoch_end(epoch, epoch_time, loss, model, **kwargs)
        if (epoch + 1) % 10 == 0:  # Save checkpoint every 10 epochs
            save_checkpoint(epoch, model, loss)

callbacks = [CheckpointCallback()]

print(f"\nStarting training for 80 epochs...")
try:
    trained_model, loss = train(
        model=model,
        prior=prior,
        criterion=dist,
        epochs=80,
        device=device,
        callbacks=callbacks,
    )
    
    # Save final model
    torch.save(trained_model.to('cpu').state_dict(), weights_path)
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to {weights_path}")
    print(f"✓ Bucket edges saved to {buckets_path}")
    
except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user")
    if checkpoint_path.exists():
        print(f"Latest checkpoint saved at {checkpoint_path}")
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    raise
