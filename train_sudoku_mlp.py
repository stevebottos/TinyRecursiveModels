"""
Distilled Sudoku training using TRM model with MLP variant.

Usage:
    python train_sudoku_mlp.py
"""

import os
import math
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import mlflow
from tqdm import tqdm

from dataset.sudoku import SudokuDataset, SudokuConfig
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.losses import ACTLossHead
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


def memops():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)


def cosine_schedule_with_warmup(
    step, base_lr, warmup_steps, total_steps, min_ratio=0.1
):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (
        min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def flatten_metrics(metrics, prefix=""):
    flat = {}
    for key, value in metrics.items():
        new_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, new_key))
        else:
            flat[new_key] = float(value)
    return flat


def train():
    # Enable memory optimizations
    memops()

    # Hyperparameters
    batch_size = 64
    epochs = 10000
    lr = 1e-4
    puzzle_emb_lr = 1e-4
    weight_decay = 1.0
    puzzle_emb_weight_decay = 1.0
    warmup_steps = 100
    eval_interval = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Datasets
    train_config = SudokuConfig(split="train", num_samples=1000, augment=True)
    test_config = SudokuConfig(split="test", num_samples=1000, augment=False)

    train_dataset = SudokuDataset(train_config)
    test_dataset = SudokuDataset(test_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    # Model config
    model_config = {
        "batch_size": batch_size,
        "seq_len": 81,
        "vocab_size": 11,
        "num_puzzle_identifiers": 1,  # All Sudoku puzzles share same embedding
        "puzzle_emb_ndim": 512,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4,
        "pos_encodings": "none",  # MLP variant uses no positional encodings
        "halt_max_steps": 8,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "bfloat16",
        "mlp_t": True,  # Use MLP instead of attention
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
    }

    # Build model
    with torch.device(device):
        base_model = TinyRecursiveReasoningModel_ACTV1(model_config)
        model = ACTLossHead(base_model, loss_type="stablemax_cross_entropy")

        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Optimizers
    puzzle_emb_opt = CastedSparseEmbeddingSignSGD_Distributed(
        model.model.puzzle_emb.buffers(),
        lr=0,
        weight_decay=puzzle_emb_weight_decay,
        world_size=1,
    )

    main_opt = torch.optim.AdamW(
        model.parameters(), lr=0, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    # Training state
    total_steps = epochs * len(train_loader)
    step = 0

    # Initialize carry before training (reused across all batches)
    dummy_batch = {
        "inputs": torch.zeros(batch_size, 81, dtype=torch.long, device=device),
        "labels": torch.zeros(batch_size, 81, dtype=torch.long, device=device),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device),
    }
    with torch.device(device):
        carry = model.initial_carry(dummy_batch)

    # Setup EMA
    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)

    # MLflow
    mlflow.set_experiment("sudoku")
    mlflow.start_run()
    mlflow.log_params(
        {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "puzzle_emb_lr": puzzle_emb_lr,
            "total_steps": total_steps,
            **model_config,
        }
    )

    model.train()  # type: ignore
    pbar = tqdm(total=total_steps, desc="Training")

    for epoch in range(epochs):
        for batch in train_loader:
            # Move to device
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            puzzle_ids = batch["puzzle_id"].to(device)

            batch_dict = {
                "inputs": inputs,
                "labels": targets,
                "puzzle_identifiers": puzzle_ids,
            }

            # Forward (carry is reused and updated with new batch data)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                carry, loss, metrics, _, _ = model(
                    carry=carry, batch=batch_dict, return_keys=[]
                )

            # Backward
            (loss / batch_size).backward()

            # Update LR
            lr_this_step = cosine_schedule_with_warmup(
                step, lr, warmup_steps, total_steps
            )
            puzzle_lr_this_step = cosine_schedule_with_warmup(
                step, puzzle_emb_lr, warmup_steps, total_steps
            )

            for param_group in main_opt.param_groups:
                param_group["lr"] = lr_this_step
            for param_group in puzzle_emb_opt.param_groups:
                param_group["lr"] = puzzle_lr_this_step

            # Step
            puzzle_emb_opt.step()
            main_opt.step()
            puzzle_emb_opt.zero_grad()
            main_opt.zero_grad()

            # Update EMA
            ema_helper.update(model)

            # Metrics
            count = max(metrics["count"].item(), 1)
            train_metrics = {
                f"train/{k}": (
                    v.item() / batch_size if k.endswith("loss") else v.item() / count
                )
                for k, v in metrics.items()
            }
            train_metrics["train/lr"] = lr_this_step

            mlflow.log_metrics(flatten_metrics(train_metrics), step=step)

            pbar.set_postfix(loss=f"{train_metrics['train/lm_loss']:.4f}")
            pbar.update(1)
            step += 1

            # Eval
            if step % eval_interval == 0:
                # Use EMA weights for evaluation (creates a copy)
                model_ema = ema_helper.ema_copy(model)
                model_ema.eval()
                eval_metrics = evaluate(model_ema, test_loader, device)
                mlflow.log_metrics(flatten_metrics(eval_metrics), step=step)
                del model_ema  # Clean up

    pbar.close()
    mlflow.end_run()


def evaluate(model, test_loader, device):
    all_metrics = {}
    total_count = 0

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            puzzle_ids = batch["puzzle_id"].to(device)

            batch_dict = {
                "inputs": inputs,
                "labels": targets,
                "puzzle_identifiers": puzzle_ids,
            }

            # Initialize carry
            with torch.device(device):
                carry = model.initial_carry(batch_dict)

            # Adaptive computation loop
            steps = 0
            while True:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    carry, loss, metrics, _, all_halted = model(
                        carry=carry, batch=batch_dict, return_keys=[]
                    )
                steps += 1
                if all_halted:
                    break

            # Accumulate
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = 0
                all_metrics[k] += v.item()
            total_count += metrics["count"].item()

    # Average - divide all metrics by count (not dataset size)
    count = max(total_count, 1)
    eval_metrics = {
        f"test/{k}": v / count
        for k, v in all_metrics.items()
        if k != "count"  # Don't include count itself
    }

    print(f"Test accuracy: {eval_metrics.get('test/accuracy', 0):.4f}")
    return eval_metrics


if __name__ == "__main__":
    train()
