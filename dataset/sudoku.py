"""
PyTorch Dataset for Sudoku with on-the-fly augmentation.

Usage:
    python dataset/sudoku.py
"""
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from typing import Literal, Optional
from dataclasses import dataclass


@dataclass
class SudokuConfig:
    source_repo: str = "sapientinc/sudoku-extreme"
    split: Literal["train", "test"] = "train"
    num_samples: Optional[int] = None
    augment: bool = True
    min_difficulty: Optional[int] = None
    seed: int = 42


class SudokuDataset(Dataset):
    def __init__(self, config: SudokuConfig):
        self.config = config
        self.puzzles = []
        self.solutions = []
        self.rng = np.random.default_rng(config.seed)
        self._load_data()

    def _load_data(self):
        csv_file = hf_hub_download(
            self.config.source_repo,
            f"{self.config.split}.csv",
            repo_type="dataset"
        )

        with open(csv_file, newline="") as f:
            reader = csv.reader(f)
            next(reader)

            for source, question, answer, rating in reader:
                if self.config.min_difficulty is not None:
                    if int(rating) < self.config.min_difficulty:
                        continue

                puzzle = np.frombuffer(
                    question.replace('.', '0').encode(),
                    dtype=np.uint8
                ).reshape(9, 9) - ord('0')

                solution = np.frombuffer(
                    answer.encode(),
                    dtype=np.uint8
                ).reshape(9, 9) - ord('0')

                self.puzzles.append(puzzle)
                self.solutions.append(solution)

        if self.config.num_samples is not None:
            total = len(self.puzzles)
            if self.config.num_samples < total:
                indices = self.rng.choice(total, size=self.config.num_samples, replace=False)
                self.puzzles = [self.puzzles[i] for i in indices]
                self.solutions = [self.solutions[i] for i in indices]

    def _augment_sudoku(self, puzzle: np.ndarray, solution: np.ndarray):
        """Apply validity-preserving augmentations."""
        digit_map = np.pad(self.rng.permutation(np.arange(1, 10)), (1, 0))
        transpose = self.rng.random() < 0.5

        bands = self.rng.permutation(3)
        row_perm = np.concatenate([b * 3 + self.rng.permutation(3) for b in bands])

        stacks = self.rng.permutation(3)
        col_perm = np.concatenate([s * 3 + self.rng.permutation(3) for s in stacks])

        mapping = np.array([
            row_perm[i // 9] * 9 + col_perm[i % 9]
            for i in range(81)
        ])

        def transform(grid):
            if transpose:
                grid = grid.T
            grid = grid.flatten()[mapping].reshape(9, 9)
            return digit_map[grid]

        return transform(puzzle), transform(solution)

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = self.puzzles[idx].copy()
        solution = self.solutions[idx].copy()

        if self.config.split == "train" and self.config.augment:
            puzzle, solution = self._augment_sudoku(puzzle, solution)

        puzzle_flat = puzzle.flatten()
        solution_flat = solution.flatten()

        return {
            'input': torch.from_numpy(puzzle_flat + 1).long(),
            'target': torch.from_numpy(solution_flat + 1).long(),
            'puzzle_id': 0,  # All Sudoku puzzles use same embedding
        }


def visualize_sudoku(grid: np.ndarray, title: str = ""):
    if title:
        print(f"\n{title}")
    print("╔═══╤═══╤═══╦═══╤═══╤═══╦═══╤═══╤═══╗")
    for i in range(9):
        row = "║ "
        for j in range(9):
            val = grid[i, j]
            row += ("." if val == 0 else str(val)) + " "
            if j % 3 == 2:
                row += "║ " if j < 8 else "║"
            else:
                row += "│ "
        print(row)
        if i == 8:
            print("╚═══╧═══╧═══╩═══╧═══╧═══╩═══╧═══╧═══╝")
        elif i % 3 == 2:
            print("╠═══╪═══╪═══╬═══╪═══╪═══╬═══╪═══╪═══╣")
        else:
            print("╟───┼───┼───╫───┼───┼───╫───┼───┼───╢")


if __name__ == "__main__":
    print("Testing SudokuDataset\n")

    train_config = SudokuConfig(split="train", num_samples=1000, augment=True)
    train_dataset = SudokuDataset(train_config)
    print(f"Train: {len(train_dataset)} samples")

    # Show augmentation diversity
    sample_idx = 0
    original = train_dataset.puzzles[sample_idx]
    visualize_sudoku(original, "Original")

    for i in range(3):
        sample = train_dataset[sample_idx]
        aug = (sample['input'].numpy() - 1).reshape(9, 9)
        visualize_sudoku(aug, f"Augmented #{i+1}")

    # Test DataLoader
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch shape: {batch['input'].shape}")
    print(f"Value range: [{batch['input'].min()}, {batch['input'].max()}]")

    # Test set
    test_config = SudokuConfig(split="test", num_samples=1000, augment=False)
    test_dataset = SudokuDataset(test_config)
    print(f"\nTest: {len(test_dataset)} samples")
