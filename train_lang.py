"""
Distilled Sudoku training using TRM model.

Usage:
    python train_sudoku.py
"""

import math
import os

import mlflow
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dataset.synth_dataset import SynthDatasetSample
from models.ema import EMAHelper
from models.losses import ACTLossHead, ACTLossHeadLang
from models.recursive_reasoning.trm_lang import TinyRecursiveReasoningModel_ACTV1


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


def prepare_targets_for_loss(
    targets: torch.Tensor, pad_token_id: int, ignore_index: int = -100
) -> torch.Tensor:
    """
    Replaces all but the first pad_token_id in the target tensor with the ignore_index.
    """
    pad_mask = targets == pad_token_id
    targets_modified = targets.masked_fill(pad_mask, ignore_index)

    return targets_modified


def train():
    # Enable memory optimizations
    memops()

    # Hyperparameters
    batch_size = 16
    epochs = 100
    lr = 1e-4
    puzzle_emb_lr = 1e-4
    weight_decay = 1.0
    warmup_steps = 100
    eval_interval = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = SynthDatasetSample()

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )

    # Model config
    model_config = {
        "batch_size": 1,
        "seq_len": 256,
        "vocab_size": 11,
        "num_puzzle_identifiers": 0,  # All Sudoku puzzles share same embedding
        "puzzle_emb_ndim": 0,  # no puzzle stuff
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 768,
        "num_heads": 8,
        "expansion": 4,
        "pos_encodings": "rope",  # Just always use rope
        "halt_max_steps": 8,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "bfloat16",
        "mlp_t": False,
        "puzzle_emb_len": 0,  # No puzzle stuff
        "no_ACT_continue": True,
    }

    # Build model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    with torch.device(device):
        pretrained_embed_layer = model.transformer.wte.to(device)
        base_model = TinyRecursiveReasoningModel_ACTV1(
            model_config, pretrained_embed_layer=pretrained_embed_layer
        )
        model = ACTLossHeadLang(base_model, loss_type="stablemax_cross_entropy")

        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    main_opt = torch.optim.AdamW(
        model.parameters(), lr=0, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    # Training state
    total_steps = epochs * len(train_loader)
    step = 0

    # Initialize carry before training (reused across all batches)
    dummy_batch = {
        "inputs": torch.zeros(batch_size, 256, dtype=torch.long, device=device),
        "labels": torch.zeros(batch_size, 256, dtype=torch.long, device=device),
    }
    with torch.device(device):
        carry = model.initial_carry(dummy_batch)

    # Setup EMA
    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)

    # MLflow
    mlflow.set_experiment("lang")
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
            inputs = batch["query"]
            inputs = tokenizer(
                inputs,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )["input_ids"].to(device)

            targets = batch["synthetic_answer"]
            targets = tokenizer(
                targets,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )["input_ids"].to(device)
            targets = prepare_targets_for_loss(
                targets, pad_token_id=tokenizer.eos_token_id
            )
            batch_dict = {
                "inputs": inputs,
                "labels": targets,
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

            for param_group in main_opt.param_groups:
                param_group["lr"] = lr_this_step

            main_opt.step()
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
                continue
                # NOTE: Skipping eval until we have an actual eval set

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

            batch_dict = {
                "inputs": inputs,
                "labels": targets,
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
