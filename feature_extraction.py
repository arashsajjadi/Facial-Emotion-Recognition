import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os


def load_image(image_path, device):
    """
    Load an image using PIL and ensure it exists.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    image = Image.open(image_path).convert('RGB')
    return image


def extract_face_embedding(image_path, device):
    """
    Detects a face in the image and extracts a 512-d embedding.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device on which to run models.

    Returns:
        torch.Tensor: A tensor of shape [1, 512] with the face embedding.
    """
    # Initialize models on the correct device
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Load and process the image
    image = load_image(image_path, device)

    # Detect face and crop it; raise error if no face is detected
    face = mtcnn(image)
    if face is None:
        raise ValueError("No face detected in the image.")
    face = face.unsqueeze(0).to(device)  # add batch dimension

    # Extract and return the embedding
    embedding = resnet(face)
    return embedding


if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) != 2:
        print("Usage: python feature_extraction.py <image_path>")
        exit(1)
    image_path = sys.argv[1]
    try:
        embedding = extract_face_embedding(image_path, device)
        print("Extracted Face Embedding:")
        print(embedding)
    except Exception as e:
        print("Error:", e)
