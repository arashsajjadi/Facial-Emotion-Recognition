#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import gc
import time

# GPU memory clearing
torch.cuda.empty_cache()
gc.collect()
if not torch.cuda.is_available():
    raise SystemError("❌ CUDA is not available. This training script requires a GPU.")
else:
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
print("✅ GPU memory cleared.")

# Import the advanced model
from advanced_pretrained import AdvancedPretrainedFERModel


def get_transforms():
    # Advanced model expects 3-channel, 224x224 images with ImageNet normalization.
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if (batch_idx + 1) % 100 == 0:
            print(f"[Epoch {epoch}/{total_epochs}] Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    elapsed = time.time() - start_time
    print(f"Epoch {epoch} Training: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc * 100:.2f}%, Time = {elapsed:.2f}s")
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch} Validation: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc * 100:.2f}%, Time = {elapsed:.2f}s")
    return epoch_loss, epoch_acc


def export_model(model, device, export_path="fer_model_advanced.onnx", input_shape=(1, 3, 224, 224)):
    dummy_input = torch.randn(input_shape).to(device)
    print("Exporting advanced model to ONNX with dummy input shape:", dummy_input.shape)
    torch.onnx.export(model, dummy_input, export_path,
                      input_names=["input"],
                      output_names=["logits", "embedding"],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "logits": {0: "batch_size"},
                                    "embedding": {0: "batch_size"}},
                      opset_version=12)
    print(f"Advanced model exported to ONNX format at {export_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = get_transforms()
    train_dir = os.path.join(args.dataset_path, "train_new")
    val_dir = os.path.join(args.dataset_path, "val_new")
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise Exception("❌ Train or validation directory not found. Please run validation_split.py first.")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = AdvancedPretrainedFERModel(num_classes=len(train_dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    total_epochs = args.epochs

    print("Starting advanced model training with early stopping and extended extra epochs (patience: {} epochs)".format(
        args.patience))
    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch, total_epochs)
        scheduler.step()

        improvement = best_val_loss - val_loss if best_val_loss != float('inf') else 0.0
        print(f"Epoch {epoch}: Current LR = {scheduler.get_last_lr()[0]:.6f}, Improvement = {improvement:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.output_checkpoint)
            print(f"✅ Saved best advanced model checkpoint at epoch {epoch} (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= args.patience:
                print(
                    f"🛑 Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}).")
                print("🔄 Continuing training for 10 extra epochs to confirm stopping decision...")
                extra_epochs = 10
                for extra in range(1, extra_epochs + 1):
                    epoch += 1
                    extra_train_loss, extra_train_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                                                    epoch, total_epochs)
                    extra_val_loss, extra_val_acc = validate_epoch(model, val_loader, criterion, device, epoch,
                                                                   total_epochs)
                    scheduler.step()
                    print(
                        f"Extra Epoch {extra}: Train Loss = {extra_train_loss:.4f}, Val Loss = {extra_val_loss:.4f}, Val Acc = {extra_val_acc * 100:.2f}%")
                    if extra_val_loss < best_val_loss:
                        best_val_loss = extra_val_loss
                        best_epoch = epoch
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), args.output_checkpoint)
                        print("✅ Improvement observed during extra epochs. Resuming normal training.")
                        break
                else:
                    print("❌ No improvement during extra epochs. Stopping training.")
                    model.load_state_dict(torch.load(args.output_checkpoint, map_location=device))
                    break

    print(f"🎉 Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}.")

    # Export the final advanced model to ONNX.
    export_model(model, device, export_path=args.output_onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced FER Fine-Tuning Script")
    parser.add_argument("--dataset_path", type=str, default="dataset",
                        help="Base dataset folder (should contain train_new/ and val_new/).")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training.")
    parser.add_argument("--step_size", type=int, default=10, help="Learning rate scheduler step size.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Learning rate scheduler gamma.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (in epochs).")
    parser.add_argument("--output_checkpoint", type=str, default="fer_model_advanced.pth",
                        help="Filename to save the fine-tuned advanced model checkpoint.")
    parser.add_argument("--output_onnx", type=str, default="fer_model_advanced.onnx",
                        help="Filename to export the advanced ONNX model.")
    args = parser.parse_args()
    main(args)
