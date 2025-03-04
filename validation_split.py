#!/usr/bin/env python
import os
import shutil
import random
import argparse


def split_dataset(train_dir, train_new_dir, val_new_dir, val_ratio=0.1):
    """
    Splits the images in train_dir into two subsets:
    - A training subset saved in train_new_dir.
    - A validation subset saved in val_new_dir.
    The split is performed per class folder to ensure balanced distribution.
    """
    # Create destination directories for train_new and val_new
    os.makedirs(train_new_dir, exist_ok=True)
    os.makedirs(val_new_dir, exist_ok=True)

    # Loop over each class folder in the original train directory
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create corresponding folders in train_new and val_new
        dest_train = os.path.join(train_new_dir, class_name)
        dest_val = os.path.join(val_new_dir, class_name)
        os.makedirs(dest_train, exist_ok=True)
        os.makedirs(dest_val, exist_ok=True)

        # Get list of all image files and shuffle them
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)

        # Determine split index
        num_val = max(1, int(len(files) * val_ratio))
        val_files = files[:num_val]
        train_files = files[num_val:]

        # Copy files to validation folder
        for f in val_files:
            src = os.path.join(class_dir, f)
            dst = os.path.join(dest_val, f)
            shutil.copy2(src, dst)

        # Copy files to training folder
        for f in train_files:
            src = os.path.join(class_dir, f)
            dst = os.path.join(dest_train, f)
            shutil.copy2(src, dst)

        print(f"Class '{class_name}': {len(train_files)} train, {len(val_files)} val images.")

    print("✅ Dataset split complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the training dataset into train and validation sets.")
    parser.add_argument("--train_dir", type=str, default="dataset/train", help="Original training directory.")
    parser.add_argument("--train_new_dir", type=str, default="dataset/train_new",
                        help="Destination directory for new training set.")
    parser.add_argument("--val_new_dir", type=str, default="dataset/val_new",
                        help="Destination directory for validation set.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1).")
    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        raise Exception(f"❌ Training directory '{args.train_dir}' does not exist.")

    split_dataset(args.train_dir, args.train_new_dir, args.val_new_dir, args.val_ratio)
