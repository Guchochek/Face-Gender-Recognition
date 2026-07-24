# -*- coding: utf-8 -*-
"""
CUDA-версія CNN-ANN для 2 класів, 224x224x1, conv5 -> 7x7.
Працює в Anaconda / Spyder / Windows 10-11.

ЗМІНИ:
1. Якщо нема test_data.bin, програма автоматично шукає:
   - test_batch.bin
   - test_blur_batch.bin
2. Якщо нема train_batch_1.bin ... train_batch_5.bin,
   програма автоматично шукає:
   - train_blur_batch_1.bin ... train_blur_batch_5.bin
3. DATA_DIR = "." — шукає файли в тій самій папці, де лежить скрипт.
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# =========================
# НАЛАШТУВАННЯ
# =========================
EPOCH = 100
DATA_DIR = '.'
MODEL_PATH = 'cnn_model.pth'
OPTIMIZER_PATH = 'cnn_optimizer.pth'
FILTERS_JSON = 'cnn_filters.json'

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_CLASSES = 2
IMAGE_H = 224
IMAGE_W = 224
IMAGE_C = 1

NUM_WORKERS = 0


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
        print('❌ CUDA недоступна. PyTorch працює на CPU.')
    return device


# =========================
# ПОШУК І ЗАВАНТАЖЕННЯ .BIN ДАНИХ
# =========================
def find_train_files(data_dir):
    # Спочатку шукаємо звичайні train_batch
    normal_train = []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, 'train_batch_{}.bin'.format(i))
        if os.path.exists(file_path):
            normal_train.append(file_path)

    if normal_train:
        print('Знайдено train_batch файли:')
        for f in normal_train:
            print('  ', f)
        return normal_train

    # Якщо не знайдено — шукаємо blur train файли
    blur_train = []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, 'train_blur_batch_{}.bin'.format(i))
        if os.path.exists(file_path):
            blur_train.append(file_path)

    if blur_train:
        print('Знайдено train_blur_batch файли:')
        for f in blur_train:
            print('  ', f)
        return blur_train

    raise FileNotFoundError(
        'Не знайдено train-файлів.\n'
        'Очікувались train_batch_1.bin ... train_batch_5.bin\n'
        'або train_blur_batch_1.bin ... train_blur_batch_5.bin'
    )


def find_test_file(data_dir):
    candidates = [
        'test_data.bin',
        'test_batch.bin',
        'test_blur_batch.bin',
    ]

    for name in candidates:
        file_path = os.path.join(data_dir, name)
        if os.path.exists(file_path):
            print('Знайдено test-файл:', file_path)
            return file_path

    raise FileNotFoundError(
        'Не знайдено test-файлу.\n'
        'Очікувався один з цих файлів:\n'
        'test_data.bin\n'
        'test_batch.bin\n'
        'test_blur_batch.bin'
    )


def load_samples_from_bin(data_dir):
    image_size = IMAGE_H * IMAGE_W * IMAGE_C
    train_data = []
    train_labels = []

    train_files = find_train_files(data_dir)

    for file_path in train_files:
        with open(file_path, 'rb') as f:
            batch_data = np.frombuffer(f.read(), dtype=np.uint8)

        batch_data = batch_data.reshape(-1, image_size + 1)
        train_labels.append(batch_data[:, 0])
        train_data.append(batch_data[:, 1:])

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    test_data_path = find_test_file(data_dir)

    with open(test_data_path, 'rb') as f:
        test_data = np.frombuffer(f.read(), dtype=np.uint8)

    test_data = test_data.reshape(-1, image_size + 1)
    test_labels = test_data[:, 0]
    test_data = test_data[:, 1:]

    train_data = train_data.reshape(-1, IMAGE_C, IMAGE_H, IMAGE_W).astype(np.float32) / 255.0
    test_data = test_data.reshape(-1, IMAGE_C, IMAGE_H, IMAGE_W).astype(np.float32) / 255.0

    return train_data, train_labels, test_data, test_labels


# =========================
# МОДЕЛЬ CNN-ANN
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
# JSON-ФІЛЬТРИ
# =========================
def save_filters_to_json(model, filename):
    model_cpu = model.cpu()
    filters = {
        'conv1': model_cpu.conv1.weight.detach().numpy().tolist(),
        'conv2': model_cpu.conv2.weight.detach().numpy().tolist(),
        'conv3': model_cpu.conv3.weight.detach().numpy().tolist(),
        'conv4': model_cpu.conv4.weight.detach().numpy().tolist(),
        'conv5': model_cpu.conv5.weight.detach().numpy().tolist(),
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(filters, f)


def load_filters_from_json(model, filename, device):
    with open(filename, 'r', encoding='utf-8') as f:
        filters = json.load(f)

    model.conv1.weight = nn.Parameter(torch.tensor(filters['conv1'], dtype=torch.float32, device=device))
    model.conv2.weight = nn.Parameter(torch.tensor(filters['conv2'], dtype=torch.float32, device=device))
    model.conv3.weight = nn.Parameter(torch.tensor(filters['conv3'], dtype=torch.float32, device=device))
    model.conv4.weight = nn.Parameter(torch.tensor(filters['conv4'], dtype=torch.float32, device=device))
    model.conv5.weight = nn.Parameter(torch.tensor(filters['conv5'], dtype=torch.float32, device=device))


# =========================
# CHECKPOINT
# =========================
def save_model_and_optimizer(model, optimizer, epoch, loss, model_path, optimizer_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, model_path)

    torch.save(optimizer.state_dict(), optimizer_path)


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_model_and_optimizer(model, optimizer, model_path, optimizer_path, device):
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                move_optimizer_to_device(optimizer, device)
            elif os.path.exists(optimizer_path):
                try:
                    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                    move_optimizer_to_device(optimizer, device)
                except Exception as e:
                    print('WARNING: optimizer checkpoint пошкоджений або не підходить.')
                    print('Причина:', e)
                    print('Продовжую без завантаження optimizer.')

            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', None)
            print('Модель завантажена з {}. Продовження з епохи {}.'.format(model_path, epoch + 2))
            return epoch + 1, loss

        except Exception as e:
            print('WARNING: файл моделі знайдено, але його не вдалося прочитати.')
            print('Файл:', model_path)
            print('Причина:', e)
            print('Ймовірно cnn_model.pth пошкоджений або був збережений не повністю.')
            print('Починаю навчання з нуля.')
            print('Порада: можна видалити або перейменувати cnn_model.pth і cnn_optimizer.pth.')

    print('Файл робочої моделі не знайдено. Початок тренування з нуля.')
    return 0, None


# =========================
# DATALOADER
# =========================
def make_dataloaders(train_data, train_labels, test_data, test_labels, device):
    train_data = (torch.tensor(train_data, dtype=torch.float32) - 0.5) / 0.5
    test_data = (torch.tensor(test_data, dtype=torch.float32) - 0.5) / 0.5

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    use_pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory
    )

    return train_loader, test_loader


# =========================
# НАВЧАННЯ
# =========================
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# =========================
# ТЕСТУВАННЯ
# =========================
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# =========================
# ГОЛОВНА ФУНКЦІЯ
# =========================
def train_model():
    device = get_device()

    print('Завантаження даних з:', DATA_DIR)
    train_data, train_labels, test_data, test_labels = load_samples_from_bin(DATA_DIR)

    print('Train samples:', len(train_labels))
    print('Test samples:', len(test_labels))
    print('Batch size:', BATCH_SIZE)

    train_loader, test_loader = make_dataloaders(
        train_data, train_labels, test_data, test_labels, device
    )

    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    start_epoch, last_loss = load_model_and_optimizer(
        model, optimizer, MODEL_PATH, OPTIMIZER_PATH, device
    )

    for epoch in range(start_epoch, EPOCH):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        test_loss, test_acc = evaluate_model(
            model, test_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        print(
            'Epoch {}/{} | train_loss: {:.6f} | train_acc: {:.2f}% | test_loss: {:.6f} | test_acc: {:.2f}% | time: {:.2f}s'.format(
                epoch + 1, EPOCH, train_loss, train_acc, test_loss, test_acc, epoch_time
            )
        )

        save_model_and_optimizer(
            model, optimizer, epoch, train_loss, MODEL_PATH, OPTIMIZER_PATH
        )

    save_filters_to_json(model, FILTERS_JSON)
    print('Фільтри збережено у файл:', FILTERS_JSON)

    print('Готово.')


if __name__ == '__main__':
    start = time.time()
    train_model()
    end = time.time()

    elapsed = end - start
    print('Час виконання: {:.5f} секунд'.format(elapsed))
    print('Час виконання: {:.2f} хвилин'.format(elapsed / 60.0))
