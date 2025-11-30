from dotenv import load_dotenv
from torch.utils.data import Dataset
from datasets import load_from_disk
import os

load_dotenv(".env")


class SynthDatasetSample(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data = self._load_data()

    def _load_data(self):
        cache_dir = os.getenv("DS_CACHE_DIR")
        if not cache_dir:
            raise ValueError("DS_CACHE_DIR environment variable not set.")

        load_path = os.path.join(cache_dir, "synth-sample", self.split)

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Dataset not found at {load_path}. Did you run 'invoke get-synth-dataset-sample'?"
            )

        return load_from_disk(load_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return {
            "input": torch.from_numpy(puzzle_flat + 1).long(),
            "target": torch.from_numpy(solution_flat + 1).long(),
            "puzzle_id": 0,  # All Sudoku puzzles use same embedding
        }


if __name__ == "__main__":
    ds = SynthDatasetSample()

    for e in iter(ds):
        print(e["language"])
