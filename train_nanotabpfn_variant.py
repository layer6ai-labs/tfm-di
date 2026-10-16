#!/usr/bin/env python
"""
Parametrized training of NanoTabPFN variants for the (b) less-data and
(c) larger-arch sweep.

Saves to:
    models/nanotabpfn/<name>.pt
    models/nanotabpfn/<name>.arch.json
    models/nanotabpfn/<name>_latest.pt   (checkpoint, every 10 epochs)

Usage:
    uv run python train_nanotabpfn_variant.py \
        --name data25k --num-tables 25000 \
        --layers 6 --embed 192 --heads 6 --mlp 768 --epochs 80
"""

import argparse
import json
from pathlib import Path

import h5py
import torch

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="variant name (used in output filenames)")
    p.add_argument("--priordump", default="tfm_playground/50x3_3_100k_classification_train.h5")
    p.add_argument("--num-tables", type=int, default=None,
                   help="cap on number of tables to use from the prior dump (None = all)")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--embed", type=int, default=192)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp", type=int, default=768)
    p.add_argument("--num-outputs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--steps", type=int, default=25, help="batches per epoch")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2402)
    p.add_argument("--output-dir", default="models/nanotabpfn")
    return p.parse_args()


class SubsetPriorDumpDataLoader(PriorDumpDataLoader):
    """PriorDumpDataLoader that wraps after `max_index` instead of file size."""

    def __init__(self, *, max_index, **kwargs):
        super().__init__(**kwargs)
        self.max_index = max_index

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                # Cap end at max_index, wrap if needed.
                if end > self.max_index:
                    end = self.max_index

                num_features = f["num_features"][self.pointer:end].max()
                if self.has_num_datapoints:
                    num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                    max_seq_in_batch = int(num_datapoints_batch.max())
                else:
                    max_seq_in_batch = int(self.stored_max_seq_len)

                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                single_eval_pos = f["single_eval_pos"][self.pointer:end]

                self.pointer += self.batch_size
                if self.pointer >= self.max_index:
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),
                    single_eval_pos=single_eval_pos[0].item(),
                )


def main():
    args = parse_args()
    set_randomness_seed(args.seed)
    device = get_default_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / f"{args.name}.pt"
    checkpoint_path = output_dir / f"{args.name}_latest.pt"
    arch_path = output_dir / f"{args.name}.arch.json"

    arch = {
        "num_attention_heads": args.heads,
        "embedding_size": args.embed,
        "mlp_hidden_size": args.mlp,
        "num_layers": args.layers,
        "num_outputs": args.num_outputs,
    }
    train_meta = {
        **arch,
        "num_tables": args.num_tables,
        "epochs": args.epochs,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "priordump": args.priordump,
    }
    with arch_path.open("w") as f:
        json.dump(train_meta, f, indent=2)
    print(f"Wrote arch metadata: {arch_path}")
    print(f"Variant config: {train_meta}")

    if args.num_tables is None:
        prior = PriorDumpDataLoader(
            filename=args.priordump,
            num_steps=args.steps,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        prior = SubsetPriorDumpDataLoader(
            max_index=args.num_tables,
            filename=args.priordump,
            num_steps=args.steps,
            batch_size=args.batch_size,
            device=device,
        )

    model = NanoTabPFNModel(**arch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    def save_checkpoint(epoch, model, loss):
        torch.save({"epoch": epoch, "model": model.state_dict(), "loss": loss}, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} -> {checkpoint_path}", flush=True)

    class CheckpointCallback(ConsoleLoggerCallback):
        def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
            super().on_epoch_end(epoch, epoch_time, loss, model, **kwargs)
            if (epoch + 1) % 10 == 0:
                save_checkpoint(epoch, model, loss)

    print(f"Starting training for {args.epochs} epochs...")
    trained_model, loss = train(
        model=model,
        prior=prior,
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        callbacks=[CheckpointCallback()],
    )
    torch.save(trained_model.to("cpu").state_dict(), weights_path)
    print(f"Training complete. Saved final weights to {weights_path}")


if __name__ == "__main__":
    main()
