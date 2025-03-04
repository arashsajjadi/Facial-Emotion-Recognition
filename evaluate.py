#!/usr/bin/env python
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import gc
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# GPU memory clearing
torch.cuda.empty_cache()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
print("✅ GPU memory cleared.")

# Use the updated model from fine_tuning.py (ensure your fine-tuning file defines AdvancedFERModel)
from fine_tuning import AdvancedFERModel


def compute_class_centroids(embeddings, labels, num_classes):
    centroids = {}
    for c in range(num_classes):
        inds = np.where(labels == c)[0]
        if len(inds) > 0:
            centroids[c] = np.mean(embeddings[inds], axis=0)
    return centroids


def evaluate_embeddings(embeddings, labels, num_classes):
    centroids = compute_class_centroids(embeddings, labels, num_classes)
    cosine_similarities = []
    for i, emb in enumerate(embeddings):
        c = labels[i]
        centroid = centroids[c]
        cos_sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8)
        cosine_similarities.append(cos_sim)
    avg_cos_sim = np.mean(cosine_similarities)
    print(f"Average Intra-class Cosine Similarity: {avg_cos_sim:.4f}")

    # Clustering evaluation using KMeans and ARI.
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(labels, cluster_labels)
    print(f"Clustering Adjusted Rand Index (ARI): {ari:.4f}")


def extract_embeddings(model, dataloader, device):
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images, return_embedding=True)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_embeddings, all_labels


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dir = os.path.join("dataset", "test")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedFERModel(num_classes=len(test_dataset.classes))
    model_path = "fer_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded.")
    else:
        print("Model weights not found. Exiting.")
        return

    # Extract embeddings and evaluate vector quality
    embeddings, labels = extract_embeddings(model, test_loader, device)
    evaluate_embeddings(embeddings, labels, num_classes=len(test_dataset.classes))

    # Compute classification accuracy and confusion matrix
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, _ = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    accuracy = np.mean(all_preds == all_true)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")

    # Compute confusion matrix and plot it
    cm = confusion_matrix(all_true, all_preds)
    print("Confusion Matrix:")
    print(cm)
    class_names = test_dataset.classes
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()
