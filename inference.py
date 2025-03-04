#!/usr/bin/env python
import os
import sys
import cv2
import torch
import gc
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

# Try importing onnxruntime; if unavailable, set flag accordingly.
try:
    import onnxruntime as ort
    onnx_available = True
except ModuleNotFoundError:
    onnx_available = False

# GPU memory clearing
torch.cuda.empty_cache()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
print("✅ GPU memory cleared.")

# Define emotion names (same for both models)
emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to load the appropriate model based on model_type.
def load_model(model_type, num_classes):
    if model_type == "advanced":
        from advanced_pretrained import AdvancedPretrainedFERModel
        return AdvancedPretrainedFERModel(num_classes=num_classes)
    else:
        from fine_tuning import AdvancedFERModel
        return AdvancedFERModel(num_classes=num_classes)

# Get the appropriate transformation.
def get_transform(model_type):
    if model_type == "advanced":
        # Advanced model expects 3-channel 224x224 input with ImageNet normalization.
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # Default model: grayscale, 48x48.
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

# Load image using OpenCV.
def load_image_cv(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' not found.")
    return cv2.imread(image_path)

# Face detection using Haar Cascade with fallback.
def detect_largest_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    except AttributeError:
        cascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade XML file not found at {cascade_path}. Please download it.")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2]*rect[3])  # (x, y, w, h)

# Preprocess face image using PIL.
def preprocess_face(face_img, transform, model_type):
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    if model_type != "advanced":
        face_pil = face_pil.convert("L")
    return transform(face_pil).unsqueeze(0)

# Run inference using PyTorch.
def run_inference_pytorch(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs, embedding = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs, embedding.cpu().numpy()[0]

# Run inference using ONNX.
def run_inference_onnx(image_tensor, onnx_path):
    if not onnx_available:
        print("Error: ONNX Runtime not installed. Run without --use_onnx or install onnxruntime.")
        sys.exit(1)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_np = image_tensor.cpu().numpy()
    inputs = {session.get_inputs()[0].name: input_np}
    outputs = session.run(None, inputs)
    # Assume outputs[0] contains logits.
    logits = outputs[0]
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    embedding = outputs[1][0]
    return probs[0], embedding

def draw_top3_bars(image, face_rect, top3):
    """Draw a horizontal bar chart (top-3) above the face rectangle."""
    overlay = image.copy()
    x, y, w, h = face_rect
    start_y = max(y - 80, 0)
    bar_height = 25
    gap = 10
    max_width = w
    for i, (emo, percent) in enumerate(top3):
        bar_length = int((percent / 100.0) * max_width)
        y_bar = start_y + i * (bar_height + gap)
        cv2.rectangle(overlay, (x, y_bar), (x + max_width, y_bar + bar_height), (50, 50, 50), -1)
        cv2.rectangle(overlay, (x, y_bar), (x + bar_length, y_bar + bar_height), (255, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        cv2.putText(overlay, emo, (x + 5, y_bar + bar_height//2 + 8),
                    font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        text_right = f"{percent:.0f}%"
        text_size, _ = cv2.getTextSize(text_right, font, font_scale, thickness)
        cv2.putText(overlay, text_right, (x + max_width - text_size[0] - 5, y_bar + bar_height//2 + 8),
                    font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def main():
    parser = argparse.ArgumentParser(
        description="FER Inference: Detect face, classify emotions, and display sorted top-3 results."
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX Runtime for inference")
    parser.add_argument("--model_type", type=str, choices=["default", "advanced"], default="default",
                        help="Model type: 'default' (grayscale, 48x48) or 'advanced' (RGB, 224x224)")
    args = parser.parse_args()

    transform = get_transform(args.model_type)
    image = load_image_cv(args.image_path)
    orig_image = image.copy()

    face_rect = detect_largest_face(image)
    if face_rect is None:
        print("No face detected; using full image.")
        face_img = image
        face_rect = (0, 0, image.shape[1], image.shape[0])
    else:
        x, y, w, h = face_rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = image[y:y+h, x:x+w]

    try:
        face_tensor = preprocess_face(face_img, transform, args.model_type)
    except Exception as e:
        print(f"Error processing face: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_onnx:
        if args.model_type == "advanced":
            onnx_path = "fer_model_advanced.onnx"
        else:
            onnx_path = "fer_model.onnx"
        if not os.path.exists(onnx_path):
            print("ONNX model file not found for the selected model type.")
            sys.exit(1)
        probs, embedding = run_inference_onnx(face_tensor, onnx_path)
    else:
        model = load_model(args.model_type, num_classes=len(emotion_names))
        if args.model_type == "advanced":
            model_path = "fer_model_advanced.pth"
        else:
            model_path = "fer_model.pth"
        if not os.path.exists(model_path):
            print("Model weights not found for the selected model type.")
            sys.exit(1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        probs, embedding = run_inference_pytorch(face_tensor, model, device)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_emotions = [(emotion_names[i], probs[i] * 100) for i in sorted_indices]
    print("Sorted Emotion Probabilities:")
    for emo, percent in sorted_emotions:
        print(f"{emo}: {percent:.1f}%")

    top3 = sorted_emotions[:3]
    draw_top3_bars(orig_image, face_rect, top3)
    predicted_idx = int(np.argmax(probs))
    cv2.putText(orig_image, f"Predicted: {emotion_names[predicted_idx]}",
                (face_rect[0], max(face_rect[1]-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("FER Inference", orig_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
