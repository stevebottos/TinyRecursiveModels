from invoke.tasks import task
from dotenv import load_dotenv
import os

load_dotenv(".env")


@task
def get_synth_dataset_sample(c):
    from datasets import load_dataset
    from tqdm import tqdm

    output = os.getenv("DS_CACHE_DIR")
    assert os.path.exists(output)  # type: ignore

    print("Loading SYNTH dataset with streaming...")
    ds = load_dataset(
        "PleIAs/SYNTH",
        cache_dir=output,
        streaming=True,
    )

    print(f"Available splits: {list(ds.keys())}")

    print("\nDownloading 1M train samples...")
    train_samples = list(tqdm(ds["train"].take(1000), total=1000, desc="Train"))

    print("\nCreating test split from last 1K samples...")
    # Take 1000 more for test, skip the first 1M
    test_samples = list(
        tqdm(ds["train"].skip(1000).take(1000), total=1000, desc="Test")
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Save to disk
    from datasets import Dataset

    train_ds = Dataset.from_list(train_samples)
    test_ds = Dataset.from_list(test_samples)

    save_path = os.path.join(output, "synth-sample")
    os.makedirs(save_path, exist_ok=True)

    print(f"Saving to {save_path}...")
    train_ds.save_to_disk(os.path.join(save_path, "train"))
    test_ds.save_to_disk(os.path.join(save_path, "test"))

    print("Done!")
