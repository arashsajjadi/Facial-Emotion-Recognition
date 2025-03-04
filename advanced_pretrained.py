#!/usr/bin/env python
import torch
import torch.nn as nn
import timm  # For pretrained models
import argparse
import gc
import sys

# GPU memory clearing
torch.cuda.empty_cache()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
print("✅ GPU memory cleared.")


class AdvancedPretrainedFERModel(nn.Module):
    """
    Advanced FER model using a Swin-Base backbone pretrained on ImageNet.
    The backbone returns a tensor with shape [batch, 7, 7, 1024] (channels-last).
    We convert it to channels-first, apply adaptive average pooling to get [batch, 1024, 1, 1],
    then flatten to [batch, 1024], expand to 2304 using fc_pre, and finally map to a 128D embedding.
    The classifier head maps the 128D embedding to 7 emotion classes.

    Expected input shape: (batch, 3, 224, 224)
    """

    def __init__(self, num_classes=7):
        super(AdvancedPretrainedFERModel, self).__init__()
        # Create backbone with features_only=True.
        self.backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=True, features_only=True)
        # We'll use the last feature map. According to your printout, it's [batch, 7, 7, 1024]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Expand feature dimension from 1024 to 2304.
        self.fc_pre = nn.Linear(1024, 2304)
        # Map 2304 -> 128 to get a 128D embedding.
        self.fc_embedding = nn.Linear(2304, 128)
        # Classifier head: 128 -> num_classes.
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_embedding=False):
        # Get backbone output. Expecting shape [batch, 7, 7, 1024]
        features = self.backbone(x)[-1]
        #print("Backbone output shape (before permute):", features.shape)
        # Convert from channels-last to channels-first: [batch, 1024, 7, 7]
        features = features.permute(0, 3, 1, 2)
        #print("After permute:", features.shape)
        # Adaptive average pooling to [batch, 1024, 1, 1]
        pooled = self.avgpool(features)
        #print("After avgpool:", pooled.shape)
        # Flatten to [batch, 1024]
        pooled_flat = pooled.reshape(pooled.size(0), -1)
        #print("After flatten:", pooled_flat.shape)
        # Expand from 1024 -> 2304
        pre = self.fc_pre(pooled_flat)
        #print("After fc_pre (should be [batch, 2304]):", pre.shape)
        # Compute embedding [batch, 128]
        embedding = self.fc_embedding(pre)
        #print("Embedding shape:", embedding.shape)
        if return_embedding:
            return embedding
        logits = self.classifier(embedding)
        #print("Logits shape:", logits.shape)
        return logits, embedding


def export_model(model, device, export_path="fer_model_advanced.onnx", input_shape=(1, 3, 224, 224)):
    """
    Exports the model to ONNX format.
    The exported model outputs both logits and embedding.
    """
    dummy_input = torch.randn(input_shape).to(device)
    print("Exporting model to ONNX with dummy input shape:", dummy_input.shape)
    torch.onnx.export(model, dummy_input, export_path,
                      input_names=["input"],
                      output_names=["logits", "embedding"],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "logits": {0: "batch_size"},
                                    "embedding": {0: "batch_size"}},
                      opset_version=12)
    print(f"Model exported to ONNX format at {export_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)
    model = AdvancedPretrainedFERModel(num_classes=7)
    model.to(device)

    # If checkpoint provided, try to load it.
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint file {args.checkpoint} not found.")
            sys.exit(1)
        try:
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded checkpoint from {args.checkpoint}")
        except RuntimeError as e:
            print("Error loading checkpoint:")
            print(e)
            print("The provided checkpoint does not match the advanced model architecture.")
            sys.exit(1)

    # Save the advanced model checkpoint.
    torch.save(model.state_dict(), args.output_checkpoint)
    print(f"Advanced model checkpoint saved to {args.output_checkpoint}")

    # Export the model to ONNX.
    export_model(model, device, export_path=args.output_onnx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Pretrained FER Model Exporter (Swin-Base)")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to a pretrained advanced model checkpoint (optional).")
    parser.add_argument("--output_checkpoint", type=str, default="fer_model_advanced.pth",
                        help="Filename to save the advanced model checkpoint.")
    parser.add_argument("--output_onnx", type=str, default="fer_model_advanced.onnx",
                        help="Filename to export the ONNX model.")
    args = parser.parse_args()
    main(args)
