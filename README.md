# Face-Gender-Recognition
A PyTorch-based Convolutional Neural Network (CNN) for gender classification from facial images. The application automatically scans the internet_test_images folder, classifies each image as Male or Female, and exports the results to a CSV file.
<div align="center">

# 👨 Gender Recognition using Convolutional Neural Networks 👩

### Deep Learning Project for Automatic Gender Classification from Facial Images

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green?logo=nvidia)
![Windows](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)
![License](https://img.shields.io/badge/License-MIT-yellow)

*A custom Convolutional Neural Network (CNN) built with PyTorch for binary gender classification from facial images.*

</div>

---

# 📖 Overview

This project implements a **Convolutional Neural Network (CNN)** capable of classifying facial images into two categories:

class_0 - 👨 **Male**
class_1 - 👩 **Female**

Unlike webcam-based applications, this project processes **all images stored inside a folder**, making it useful for batch image classification.

The program automatically:

- 📂 Reads every image from the `internet_test_images` directory.
- 🖼 Converts each image to grayscale.
- 📏 Resizes images to **224×224 pixels**.
- 🧠 Performs inference using a trained CNN model.
- 📊 Calculates prediction probabilities.
- 💾 Saves all results into a CSV report.

---

# ✨ Features

- Binary gender classification
- Custom CNN architecture
- GPU acceleration using CUDA
- Automatic CPU fallback
- Batch image processing
- CSV export of predictions
- Automatic image discovery
- Support for JPG, PNG, BMP, TIFF and WEBP images
- Checkpoint model loading

---

# 🧠 CNN Architecture

```
Input Image
224 × 224 × 1

        │

Conv2D (4 filters)

        │

Max Pooling

        │

Conv2D (8 filters)

        │

Max Pooling

        │

Conv2D (16 filters)

        │

Max Pooling

        │

Conv2D (32 filters)

        │

Max Pooling

        │

Conv2D (64 filters)

        │

Max Pooling

        │

Flatten

        │

Fully Connected (512)

        │

Fully Connected (256)

        │

Fully Connected (128)

        │

Fully Connected (64)

        │

Output Layer

        │

Male / Female
```

---

# 📂 Project Structure

```
Gender-Recognition-CNN/

│

├── train.py
├── classify.py

├── cnn_model.pth

├── internet_test_images/

│      image1.jpg
│      image2.png
│      image3.jpg

├── classification_results.csv

├── requirements.txt

├── README.md

├── LICENSE

└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Gender-Recognition-CNN.git
```

Open the project

```bash
cd Gender-Recognition-CNN
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

- Python 3.11+
- PyTorch
- Pillow
- NumPy

---

# ▶ Training

Train the CNN model

```bash
python train.py
```

The trained model will be saved as:

```
cnn_model.pth
```

---

# 📷 Image Classification

Copy your facial images into:

```
internet_test_images/
```

Run the classifier:

```bash
python classify.py
```

The application will automatically classify every image inside the folder.

---

# 📄 Output

After processing all images, the application creates:

```
classification_results.csv
```

Example:

| Image | Prediction | Confidence |
|--------|------------|------------|
| image1.jpg | Male | 99.84% |
| image2.jpg | Female | 98.71% |
| image3.jpg | Male | 99.21% |

---

# Supported Image Formats

- JPG
- JPEG
- PNG
- BMP
- TIFF
- WEBP

---

# ⚡ GPU Support

The application automatically detects whether CUDA is available.

If a supported NVIDIA GPU is found:

- CUDA is enabled automatically.
- Inference runs on GPU.

Otherwise:

- CPU inference is used.

---

# 📊 Workflow

```
User Images

       │

       ▼

internet_test_images

       │

       ▼

Image Preprocessing

(Grayscale + Resize)

       │

       ▼

CNN Model

       │

       ▼

Prediction

(Male / Female)

       │

       ▼

classification_results.csv
```

---

# 🖼 Screenshots

## Folder with Images

*(Add screenshot here)*

---

## Console Output

*(Add screenshot here)*

---

## Classification Results

*(Add screenshot here)*

---

# 🛠 Technologies

- Python
- PyTorch
- CUDA
- Pillow
- NumPy

---

# 🔮 Future Improvements

- Support for more classes
- Model optimization
- ONNX export
- TensorRT support
- Graphical User Interface (GUI)
- Cross-platform compatibility

---

# 👨‍💻 Author

Developed by **Yevhen Martyniuk**

Deep Learning • Computer Vision • PyTorch

---

# 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

</div>
