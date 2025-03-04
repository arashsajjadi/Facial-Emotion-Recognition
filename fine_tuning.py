#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gc
import argparse

# GPU memory clearing
torch.cuda.empty_cache()
gc.collect()
if not torch.cuda.is_available():
    raise SystemError("❌ CUDA is not available. This training script requires a GPU.")
else:
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
print("✅ GPU memory cleared.")


# Define an advanced CNN model with deeper architecture, dropout, and batch normalization.
class AdvancedFERModel(nn.Module):
    def __init__(self, num_classes=7):
        super(AdvancedFERModel, self).__init__()
        self.features = nn.Sequential(
            # Conv Block 1: Input 1x48x48 -> 32x24x24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Block 2: 32x24x24 -> 64x12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),

            # Conv Block 3: 64x12x12 -> 128x6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),

            # Conv Block 4: 128x6x6 -> 256x3x3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2)
        )
        self.flatten_dim = 256 * 3 * 3  # For 48x48 input images
        self.fc_embedding = nn.Linear(self.flatten_dim, 128)  # 128D embedding
        self.dropout_fc = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc_embedding(x)
        embedding = self.dropout_fc(embedding)
        if return_embedding:
            return embedding
        logits = self.classifier(embedding)
        return logits, embedding


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, patience, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # Using integer labels from ImageFolder
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        current_lr = scheduler.get_last_lr()[0]
        improvement = best_val_loss - val_loss if best_val_loss != float('inf') else 0.0

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | LR: {current_lr:.6f} | Improvement: {improvement:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save the best model weights
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"🛑 Early stopping triggered at epoch {epoch} (best validation loss: {best_val_loss:.4f} at epoch {best_epoch}).")
                print("🔄 Continuing training for 10 more epochs to confirm early stopping decision.")
                extra_epochs = 10  # Updated extra epochs from 2 to 10
                for extra in range(extra_epochs):
                    epoch += 1
                    model.train()
                    extra_train_loss = 0.0
                    for images, labels in train_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs, _ = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        extra_train_loss += loss.item() * images.size(0)
                    scheduler.step()
                    extra_train_loss /= len(train_loader.dataset)

                    model.eval()
                    extra_val_loss = 0.0
                    correct_extra = 0
                    total_extra = 0
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs, _ = model(images)
                            loss = criterion(outputs, labels)
                            extra_val_loss += loss.item() * images.size(0)
                            preds = outputs.argmax(dim=1)
                            correct_extra += (preds == labels).sum().item()
                            total_extra += labels.size(0)
                    extra_val_loss /= len(val_loader.dataset)
                    extra_val_acc = correct_extra / total_extra
                    current_lr = scheduler.get_last_lr()[0]

                    print(
                        f"Extra Epoch {extra + 1}: Train Loss: {extra_train_loss:.4f} | Val Loss: {extra_val_loss:.4f} | Val Acc: {extra_val_acc * 100:.2f}% | LR: {current_lr:.6f}")

                    # If improvement happens during extra epochs, resume training normally.
                    if extra_val_loss < best_val_loss:
                        best_val_loss = extra_val_loss
                        best_epoch = epoch
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), "best_model.pth")
                        print("✅ Validation loss improved during extra epochs. Resuming training normally.")
                        break
                else:
                    print("❌ No improvement during extra epochs. Stopping training.")
                    model.load_state_dict(torch.load("best_model.pth", map_location=device))
                    return model
        # End of epoch loop
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    return model


def main(args):
    # Data transforms: convert to grayscale, resize, to tensor, and normalize.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load train and validation datasets from new directories
    train_dir = os.path.join(args.dataset_path, "train_new")
    val_dir = os.path.join(args.dataset_path, "val_new")
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise Exception("❌ Train or validation directory not found. Run validation_split.py first.")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedFERModel(num_classes=len(train_dataset.classes))

    # Optionally load a pretrained model for fine-tuning.
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print("Loading pretrained model for fine-tuning...")
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))

    print("Starting training with early stopping...")
    model = train_model(model, train_loader, val_loader,
                        num_epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        patience=args.patience,
                        device=device)

    # Save final model and export to ONNX (exporting the embedding branch)
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")

    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    torch.onnx.export(model, dummy_input, args.output_onnx, input_names=["input"], output_names=["embedding"],
                      dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}})
    print(f"Model exported to ONNX format at {args.output_onnx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom FER model with early stopping and validation split.")
    parser.add_argument("--dataset_path", type=str, default="dataset",
                        help="Base dataset folder (should contain train_new/ and val_new/).")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (in epochs).")
    parser.add_argument("--pretrained_model", type=str, default="",
                        help="Path to a pretrained model for fine-tuning (optional).")
    parser.add_argument("--output_model", type=str, default="fer_model.pth", help="Filename to save the trained model.")
    parser.add_argument("--output_onnx", type=str, default="fer_model.onnx", help="Filename to export the ONNX model.")
    args = parser.parse_args()
    model = main(args)
