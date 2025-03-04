#!/usr/bin/env python
import os
import cv2
import torch
import gc
import numpy as np
from PIL import Image
from torchvision import transforms
import time
import argparse

# GPU memory clearing
torch.cuda.empty_cache()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
print("✅ GPU memory cleared.")

# Define emotion names (common to both models)
emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# Model selection function
def load_model(model_type, num_classes):
    if model_type == "advanced":
        from advanced_pretrained import AdvancedPretrainedFERModel
        return AdvancedPretrainedFERModel(num_classes=num_classes)
    else:
        from fine_tuning import AdvancedFERModel
        return AdvancedFERModel(num_classes=num_classes)


def get_transform(model_type):
    if model_type == "advanced":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


def load_image_cv(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' not found.")
    return cv2.imread(image_path)


def detect_faces(gray):
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    except AttributeError:
        cascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade XML file not found at {cascade_path}. Please download it.")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces


def preprocess_face(face_img, transform, model_type):
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    if model_type != "advanced":
        face_pil = face_pil.convert("L")
    return transform(face_pil).unsqueeze(0)


def run_inference_pytorch(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs, embedding = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs, embedding.cpu().numpy()[0]


def run_inference_onnx(image_tensor, onnx_path):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("Error: ONNX Runtime not installed. Install it or run without --use_onnx.")
        exit(1)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else [
        'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_np = image_tensor.cpu().numpy()
    inputs = {session.get_inputs()[0].name: input_np}
    outputs = session.run(None, inputs)
    logits = outputs[0]
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    embedding = outputs[1][0]
    return probs[0], embedding


def draw_top3_bars(frame, face_rect, top3):
    overlay = frame.copy()
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
        cv2.putText(overlay, emo, (x + 5, y_bar + bar_height // 2 + 8),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        text_right = f"{percent:.0f}%"
        text_size, _ = cv2.getTextSize(text_right, font, font_scale, thickness)
        cv2.putText(overlay, text_right, (x + max_width - text_size[0] - 5, y_bar + bar_height // 2 + 8),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time FER: Process video stream/file, overlay emotion probabilities, and optionally save the output video."
    )
    parser.add_argument("--video", type=str, default=None, help="Path to video file. If omitted, webcam is used.")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX Runtime for inference")
    parser.add_argument("--model_type", type=str, choices=["default", "advanced"], default="default",
                        help="Model type: 'default' (1-channel, 48x48) or 'advanced' (3-channel, 224x224)")
    parser.add_argument("--save_output", type=str, default="",
                        help="Filename to save output video (e.g., output.mp4). If empty, video is not saved.")
    args = parser.parse_args()

    # Open video capture: either video file or webcam.
    cap = cv2.VideoCapture(args.video) if args.video else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit(1)

    # Initialize VideoWriter if save_output is specified.
    out = None
    if args.save_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25  # default fallback fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec if needed.
        out = cv2.VideoWriter(args.save_output, fourcc, fps, (frame_width, frame_height))
        print("Output video will be saved to:", args.save_output)

    transform = get_transform(args.model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.use_onnx:
        model = load_model(args.model_type, num_classes=len(emotion_names))
        model_path = "fer_model.pth" if args.model_type == "default" else "fer_model_advanced.pth"
        if not os.path.exists(model_path):
            print("Model weights not found for the selected model type.")
            exit(1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        onnx_path = "fer_model.onnx" if args.model_type == "default" else "fer_model_advanced.onnx"
        if not os.path.exists(onnx_path):
            print("ONNX model file not found for the selected model type.")
            exit(1)

    fps_start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face detection (using grayscale version)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
            .detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        all_probs = []
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y + h, x:x + w]
            try:
                face_tensor = preprocess_face(face_img, transform, args.model_type).to(device)
                if args.use_onnx:
                    probs, embedding = run_inference_onnx(face_tensor, onnx_path)
                else:
                    probs, embedding = run_inference_pytorch(face_tensor, model, device)
                all_probs.append(probs)
                pred_idx = int(np.argmax(probs))
                cv2.putText(frame, f"{emotion_names[pred_idx]}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                sorted_indices = np.argsort(probs)[::-1]
                sorted_emotions = [(emotion_names[i], probs[i] * 100) for i in sorted_indices]
                top3 = sorted_emotions[:3]
                draw_top3_bars(frame, (x, y, w, h), top3)
            except Exception as e:
                print(f"Error processing face: {e}")

        # Optional: compute group emotion if multiple faces are detected.
        if len(all_probs) > 1:
            avg_probs = np.mean(all_probs, axis=0)
            group_pred = emotion_names[int(np.argmax(avg_probs))]
            cv2.putText(frame, f"Group Emotion: {group_pred}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        frame_count += 1
        fps = frame_count / (time.time() - fps_start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Real-Time FER", frame)
        # Write frame to output video if enabled.
        if out:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
