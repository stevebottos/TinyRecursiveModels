"""
Distilled Sudoku training using TRM model.

Usage:
    python train_sudoku.py
"""

import os
import math
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import mlflow
from tqdm import tqdm

from dataset.sudoku import SudokuDataset, SudokuConfig
from models.recursive_reasoning.trm_lang import TinyRecursiveReasoningModel_ACTV1
from models.losses import ACTLossHead
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. Load Pre-trained Tokenizer and Model
# GPT2LMHeadModel includes the core transformer layers AND the final language modeling head (lm_head)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_w_embed = model.transformer.wte.weight.data
pretrained_embed_layer = model.transformer.wte
input_text = "the cat in the hat"
input_ids = tokenizer(
    input_text,
    add_special_tokens=True,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt",
)["input_ids"]

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

dummy_batch = torch.rand(1, 81)
batch = {"inputs": input_ids}
base_model = TinyRecursiveReasoningModel_ACTV1(
    model_config, pretrained_embed_layer=pretrained_embed_layer
)
carry = base_model.initial_carry(batch)

base_model(carry, batch)
# model = ACTLossHead(base_model, loss_type="stablemax_cross_entropy")
