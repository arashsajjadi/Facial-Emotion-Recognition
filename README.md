
# Facial Emotion Recognition (FER) System

*Developed by Arash Sajjadi*

Welcome to the Facial Emotion Recognition (FER) System – a home project crafted to explore, experiment, and advance the field of emotion recognition using deep learning. This project is designed not only to satisfy curiosity but also to serve as a playground for experimenting with state-of-the-art techniques in computer vision and neural network fine-tuning. 

## Table of Contents

- [Introduction](#introduction)
- [Core Features](#core-features)
- [Project Architecture & Files](#project-architecture--files)
- [Environment Setup](#environment-setup)
- [Training & Fine-Tuning](#training--fine-tuning)
- [Inference & Real-Time Video](#inference--real-time-video)
- [GPU Optimization & Deployment](#gpu-optimization--deployment)
- [Future Work](#future-work)
- [License & Acknowledgments](#license--acknowledgments)

## Introduction

In today’s world, understanding human emotions is crucial for applications ranging from user experience research and market analysis to robotics and healthcare. Traditional FER systems often rely on limited representations and simple classification schemes. Our system challenges that norm by:

- **Extracting Rich Embeddings:** Instead of merely classifying faces into discrete categories, the system produces a 128-dimensional embedding that captures the nuanced representation of facial expressions.
- **Using Advanced Pretrained Backbones:** By leveraging state-of-the-art architectures like Swin Transformer-based backbones, our model incorporates robust, high-level features learned from vast datasets (ImageNet), which are then fine-tuned for emotion recognition.
- **Implementing Modern Training Techniques:** We incorporate early stopping, validation-based regularization, and extended extra epochs to ensure that the model converges well without overfitting.
- **Supporting Real-Time Applications:** With GPU optimization and ONNX export options, the system is designed for real-time inference on both images and video streams.

## Core Features

- **Multi-Modal Dataset Handling:**  
  - Automatic dataset download and organization via a custom script.
  - Folder-based data loading (leveraging PyTorch’s ImageFolder) without reliance on CSV parsing.
  
- **Advanced Model Architecture:**  
  - Two model variants: the **default** (a custom CNN for FER2013) and the **advanced** model that uses a Swin-Base backbone.
  - The advanced model generates a 128-dimensional embedding from a 3-channel 224×224 input using adaptive pooling and additional linear layers.
  
- **Fine-Tuning & Early Stopping:**  
  - Custom training pipelines with detailed epoch reports.
  - Early stopping with extra epochs to confirm the stopping decision, ensuring robust convergence.
  
- **ONNX Export & GPU Optimization:**  
  - The trained models can be exported to ONNX format for deployment using GPU-accelerated inference (e.g., with onnxruntime-gpu or TensorRT).
  - Explicit GPU memory clearing and device management throughout the codebase.
  
- **Real-Time Video Processing:**  
  - Processes live video or pre-recorded files.
  - Overlays emotion probability bars (top three predictions) on detected faces.
  - Optionally saves the output video for review or further processing.

## Project Architecture & Files

- **environment.yml:**  
  A Conda environment file that installs all necessary dependencies including PyTorch with CUDA support, OpenCV, timm, ONNX, and other essential libraries.

- **download_dataset.py:**  
  Automatically downloads and organizes a real FER dataset (e.g., FER-2013) from an online source into a folder-based structure with `train/` and `test/` subfolders.

- **validation_split.py:**  
  Splits the training dataset into balanced `train_new/` and `val_new/` directories for robust training and evaluation.

- **fine_tuning.py:**  
  Contains the training pipeline for the default FER model (with 48×48 grayscale input).

- **advanced_pretrained.py:**  
  Defines the advanced model architecture using a Swin-Base backbone. This file is used both for model export (to checkpoint and ONNX) and as the basis for advanced fine-tuning.

- **advanced_fine_tuning.py:**  
  A fine-tuning script for the advanced model. It applies ImageNet-style transformations, detailed logging, early stopping with extra epochs, and finally saves the best checkpoint and exports an ONNX model.

- **inference.py:**  
  Runs inference on a single image. It loads either the default or advanced model (from checkpoint or ONNX) and displays sorted emotion probabilities.

- **real_time_video.py:**  
  Processes real-time video streams or video files, overlays emotion predictions (with clear, non-overlapping probability bars), and includes an option to save the output video.

## Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/FER-System.git
   cd FER-System
   ```

2. **Set Up the Conda Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate fer_system
   ```

3. **Install Additional Dependencies (if needed):**
   For ONNX GPU support:
   ```bash
   pip install onnxruntime-gpu
   ```

## Training & Fine-Tuning

### Advanced Model Fine-Tuning

To fine-tune the advanced model on your FER dataset, ensure your dataset is structured as follows:
```
dataset/
 ├── train_new/
 │    ├── Angry/
 │    ├── Disgust/
 │    ├── Fear/
 │    ├── Happy/
 │    ├── Sad/
 │    ├── Surprise/
 │    └── Neutral/
 └── val_new/
      ├── Angry/
      ├── Disgust/
      ├── Fear/
      ├── Happy/
      ├── Sad/
      ├── Surprise/
      └── Neutral/
```

Then, run the advanced fine-tuning script:
```bash
python advanced_fine_tuning.py --dataset_path dataset --epochs 30 --batch_size 32 --learning_rate 0.0001 --step_size 10 --gamma 0.5 --patience 3
```
The script prints detailed reports (training loss, validation loss, accuracy, learning rate, improvement) each epoch. It saves the best model checkpoint to **fer_model_advanced.pth** and exports the final model to **fer_model_advanced.onnx**.

## Inference & Real-Time Video

### Single Image Inference

Run the inference script for a single image:
```bash
python inference.py Test_files/IMG_test_1.jpg --model_type advanced
```
Add the `--use_onnx` flag to use the ONNX export instead:
```bash
python inference.py Test_files/IMG_test_1.jpg --use_onnx --model_type advanced
```

### Real-Time Video Processing & Saving Output

To process a video file (or webcam) and save the output, run:
```bash
python real_time_video.py --video Test_files/Video_Test_1.mp4 --model_type advanced --save_output output.mp4
```
This command overlays emotion probability bars on detected faces in real time and saves the annotated video to `output.mp4`.

## GPU Optimization & Deployment

- **GPU-Friendly Code:**  
  All scripts are written to move models and tensors to CUDA when available.  
- **ONNX Export:**  
  The advanced model can be exported to ONNX format for high-performance inference using frameworks such as TensorRT.
- **Real-Time Performance:**  
  With GPU acceleration, the system is optimized for real-time video processing even with complex architectures.

## Future Work

- **Further Fine-Tuning:**  
  Experiment with additional datasets and more extensive data augmentation to improve performance.
- **Lightweight Architectures:**  
  Consider exploring even lighter backbones for real-time applications on edge devices.
- **Enhanced UI:**  
  Improve the user interface for live video output and results visualization.
- **Multimodal Fusion:**  
  Integrate additional modalities (e.g., audio cues) for richer emotion recognition.

## License & Acknowledgments

This project is for educational and personal research purposes. Feel free to explore, modify, and extend it for your own experiments.

**License:** MIT License

**Acknowledgments:**
- **Arash Sajjadi:** Project lead and developer.
- Thanks to the contributors and open-source community for providing tools like PyTorch, timm, and ONNX Runtime.
- Inspiration and ideas provided by language models and research publications in facial emotion recognition.

---



## Contact

For questions, suggestions, or collaboration opportunities, please contact **Arash Sajjadi** at [arash.sajjadi@usask.ca](mailto:arash.sajjadi@usask.ca).

---

*Thank you for exploring this project. I hope it inspires you to delve deeper into the fascinating world of emotion recognition and deep learning!*
