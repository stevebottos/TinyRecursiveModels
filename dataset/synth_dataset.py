from dotenv import load_dotenv
from torch.utils.data import Dataset
import os
import sqlite3
from pathlib import Path

load_dotenv(".env")


class SynthDatasetSample(Dataset):
    def __init__(self, split="train"):
        self.split = split
        # self.data = self._load_data() # Removed as data is loaded directly via cursor

        db_output_dir = os.getenv("DS_OUT_DIR")
        if not db_output_dir:
            print("Error: DS_OUT_DIR environment variable not set.")
            exit(1)
        self.db_path = Path(db_output_dir) / "filtered_data.db"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM synth_data")
        self._length = cursor.fetchone()[0]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Use LIMIT/OFFSET to get the row at the idx-th position

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT query, synthetic_reasoning,synthetic_answer FROM synth_data LIMIT 1 OFFSET ?",
            (idx,),
        )

        row = cursor.fetchone()

        if row is None:
            raise IndexError(f"Index {idx} out of bounds")

        query, synthetic_reasoning, synthetic_answer = row

        conn.close()
        return {
            "query": query,
            "synthetic_reasoning": synthetic_reasoning,
            "synthetic_answer": synthetic_answer,
            "puzzle_id": 0,  # All Sudoku puzzles use same embedding
        }


if __name__ == "__main__":
    ds = SynthDatasetSample()
    print(len(ds))
    # Example of accessing items
    if len(ds) > 0:
        first_item = ds[0]
        print(f"\nFirst item: {first_item}")

        # Try accessing a few more items
        if len(ds) > 5:
            fifth_item = ds[4]
            print(f"\nFifth item: {fifth_item}")

    for i in range(min(3, len(ds))):
        item = ds[i]
        print(f"\nItem {i} keys: {item.keys()}")
        print(f"Item {i} query: {item['query']}")
