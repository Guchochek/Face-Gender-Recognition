# -*- coding: utf-8 -*-
"""
Класифікатор PyTorch CUDA для моделі CNN-ANN 2 класи, 224x224x1.

Що робить скрипт:
1. Завантажує навчену модель з cnn_model.pth.
2. Читає фотографії з теки internet_test_images.
3. Перетворює кожне фото у grayscale 224x224x1.
4. Виконує класифікацію на CUDA або CPU.
5. Формує CSV-файл classification_results.csv з параметрами класифікації.

Очікувана структура:
    ваш_проєкт/
        cnn_model.pth
        classify_internet_test_images_CUDA.py
        internet_test_images/
            image1.jpg
            image2.png
            ...

Запуск у Spyder або Anaconda Prompt:
    conda activate tf_env
    python classify_internet_test_images_CUDA.py
"""

import os
import csv
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# =========================
# НАЛАШТУВАННЯ
# =========================
MODEL_PATH = 'cnn_model.pth'
IMAGE_DIR = 'internet_test_images'
CSV_PATH = 'classification_results.csv'

IMAGE_H = 224
IMAGE_W = 224
IMAGE_C = 1
NUM_CLASSES = 2

# За потреби змініть назви класів під ваш датасет.
# Важливо: індекси мають відповідати міткам під час навчання: 0 і 1.
CLASS_NAMES = {
    0: 'class_0',
    1: 'class_1'
}

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


# =========================
# DEVICE: CUDA АБО CPU
# =========================
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('🎉 PyTorch працює на CUDA!')
        print('GPU:', torch.cuda.get_device_name(0))
        print('CUDA version in PyTorch:', torch.version.cuda)
    else:
        device = torch.device('cpu')
        print('❌ CUDA недоступна. Класифікація буде на CPU.')
    return device


# =========================
# МОДЕЛЬ CNN-ANN
# Архітектура має точно збігатися з моделлю, яку навчали.
# =========================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        x = self.pool(F.relu(self.conv4(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv5(x)))  # 14 -> 7

        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# =========================
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# =========================
def load_trained_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError('Не знайдено файл моделі: {}'.format(model_path))

    model = CNNModel().to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # Варіант 1: checkpoint зі словником model_state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)
        print('Модель завантажена з checkpoint:', model_path)
        if epoch is not None:
            print('Епоха checkpoint:', epoch + 1)
        if loss is not None:
            print('Loss checkpoint:', loss)

    # Варіант 2: якщо було збережено тільки state_dict
    else:
        model.load_state_dict(checkpoint)
        print('Модель завантажена як state_dict:', model_path)

    model.eval()
    return model


# =========================
# ПОШУК ФОТОГРАФІЙ
# =========================
def collect_image_files(image_dir):
    if not os.path.exists(image_dir):
        raise FileNotFoundError('Не знайдено теку з фото: {}'.format(image_dir))

    image_files = []

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                image_files.append(os.path.join(root, filename))

    image_files.sort()
    return image_files


# =========================
# ПІДГОТОВКА ЗОБРАЖЕННЯ
# =========================
def preprocess_image(image_path):
    """
    Повертає:
        tensor: shape [1, 1, 224, 224]
        original_width, original_height
    """
    with Image.open(image_path) as img:
        original_width, original_height = img.size

        # Grayscale, бо модель навчалась на 1 каналі.
        img = img.convert('L')

        # Якщо потрібно строго як у train bin 224x224 — просто resize.
        img = img.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)

        # PIL -> torch tensor вручну без torchvision, щоб код був простіший для Spyder.
        img_bytes = img.tobytes()
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(img_bytes))
        tensor = tensor.float().reshape(IMAGE_H, IMAGE_W) / 255.0

        # Нормалізація така сама, як у CUDA-версії навчання: (x - 0.5) / 0.5
        tensor = (tensor - 0.5) / 0.5

        # [H, W] -> [1, 1, H, W]
        tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor, original_width, original_height


# =========================
# КЛАСИФІКАЦІЯ 1 ФОТО
# =========================
def classify_one_image(model, image_path, device):
    tensor, original_width, original_height = preprocess_image(image_path)
    tensor = tensor.to(device, non_blocking=True)

    start_time = time.time()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    elapsed_ms = (time.time() - start_time) * 1000.0

    logits_cpu = logits.detach().cpu().numpy()[0]
    probs_cpu = probabilities.detach().cpu().numpy()[0]

    predicted_class = int(predicted_class.item())
    confidence = float(confidence.item())

    result = {
        'filename': os.path.basename(image_path),
        'filepath': image_path,
        'original_width': original_width,
        'original_height': original_height,
        'input_width': IMAGE_W,
        'input_height': IMAGE_H,
        'predicted_class': predicted_class,
        'class_name': CLASS_NAMES.get(predicted_class, 'class_{}'.format(predicted_class)),
        'confidence': confidence,
        'confidence_percent': confidence * 100.0,
        'prob_class_0': float(probs_cpu[0]),
        'prob_class_1': float(probs_cpu[1]),
        'logit_class_0': float(logits_cpu[0]),
        'logit_class_1': float(logits_cpu[1]),
        'device': str(device),
        'time_ms': elapsed_ms
    }

    return result


# =========================
# ЗБЕРЕЖЕННЯ CSV
# =========================
def save_results_to_csv(results, csv_path):
    fieldnames = [
        'filename',
        'filepath',
        'original_width',
        'original_height',
        'input_width',
        'input_height',
        'predicted_class',
        'class_name',
        'confidence',
        'confidence_percent',
        'prob_class_0',
        'prob_class_1',
        'logit_class_0',
        'logit_class_1',
        'device',
        'time_ms'
    ]

    # utf-8-sig — щоб Excel нормально відкривав українські символи.
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for row in results:
            writer.writerow(row)


# =========================
# ГОЛОВНА ФУНКЦІЯ
# =========================
def main():
    print('==========================================')
    print('Класифікація фото з теки:', IMAGE_DIR)
    print('Модель:', MODEL_PATH)
    print('CSV результат:', CSV_PATH)
    print('Час старту:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('==========================================')

    device = get_device()
    model = load_trained_model(MODEL_PATH, device)

    image_files = collect_image_files(IMAGE_DIR)
    print('Знайдено фото:', len(image_files))

    if len(image_files) == 0:
        print('У теці немає зображень для класифікації.')
        return

    results = []

    for index, image_path in enumerate(image_files, start=1):
        try:
            result = classify_one_image(model, image_path, device)
            results.append(result)

            print(
                '[{}/{}] {} -> class={} ({}) | confidence={:.2f}% | p0={:.4f} | p1={:.4f}'.format(
                    index,
                    len(image_files),
                    result['filename'],
                    result['predicted_class'],
                    result['class_name'],
                    result['confidence_percent'],
                    result['prob_class_0'],
                    result['prob_class_1']
                )
            )

        except Exception as e:
            print('[ПОМИЛКА] {}: {}'.format(image_path, e))

    save_results_to_csv(results, CSV_PATH)

    print('==========================================')
    print('Готово. CSV файл створено:', CSV_PATH)
    print('Кількість успішно класифікованих фото:', len(results))
    print('==========================================')


if __name__ == '__main__':
    main()
