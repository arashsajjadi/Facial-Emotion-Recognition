#!/usr/bin/env python
import os
import kagglehub
import torch
import gc
import shutil

# GPU memory clearing
torch.cuda.empty_cache()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
print("✅ GPU memory cleared.")

def download_fer2013(save_path="dataset"):
    """
    Download the FER-2013 dataset using KaggleHub if not already present.
    """
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        print(f"✅ Dataset already exists at '{save_path}'.")
        return save_path

    print("📥 Downloading FER-2013 dataset using KaggleHub...")

    try:
        # Download dataset using KaggleHub
        dataset_path = kagglehub.dataset_download("msambare/fer2013")
        print(f"✅ Dataset downloaded to: {dataset_path}")

        # Ensure the target folder exists
        os.makedirs(save_path, exist_ok=True)

        # Copy all files into `dataset/`
        for root, dirs, files in os.walk(dataset_path):
            for name in dirs:
                src = os.path.join(root, name)
                dest = os.path.join(save_path, name)
                if not os.path.exists(dest):
                    shutil.copytree(src, dest)  # Copy folders recursively

        print(f"✅ Dataset successfully copied to '{save_path}'")

    except Exception as e:
        raise Exception(f"❌ Failed to download dataset. Error: {e}")

    return save_path

if __name__ == "__main__":
    dataset_path = download_fer2013()

    # Ensure that `train/` and `test/` exist inside `dataset/`
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise Exception("❌ `train/` or `test/` folder missing! Check dataset structure.")

    print("✅ Dataset is ready for training!")
