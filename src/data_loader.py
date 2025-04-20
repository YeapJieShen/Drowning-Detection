# src/data_loader.py

import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# --- Custom Augmentation ---
class CustomTransformation:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        img_hsv = np.array(img.convert("RGB"))
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)
        v_blurred = cv2.blur(v, (7, 7))
        img_hsv_blurred = cv2.merge([h, s, v_blurred])

        return Image.fromarray(img_hsv_blurred)

# --- Augmented Dataset Generator ---
def generate_augmented_images(augmentation_path, categories, max_images_per_class):
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
    ])

    augmented_images = {cat: [] for cat in categories}

    for i, folder_path in enumerate(augmentation_path):
        category = categories[i]
        label = i + 1
        current_image_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        required_images = max_images_per_class - current_image_count

        while len(augmented_images[category]) < required_images:
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(folder_path, filename)
                    img = Image.open(img_path)
                    aug_img = augmentation_transforms(img)
                    augmented_images[category].append((aug_img, label))
                    if len(augmented_images[category]) >= required_images:
                        break
    return augmented_images

# --- Combined Dataset Class ---
class CombinedDataset(Dataset):
    def __init__(self, original_dataset, augmented_images, transform=None):
        self.original_images = original_dataset.imgs
        self.transform = transform
        self.augmented_data = []
        for img_list in augmented_images.values():
            self.augmented_data.extend(img_list)

    def __len__(self):
        return len(self.original_images) + len(self.augmented_data)

    def __getitem__(self, idx):
        if idx < len(self.original_images):
            img_path, label = self.original_images[idx]
            img = Image.open(img_path)
        else:
            img, label = self.augmented_data[idx - len(self.original_images)]

        if self.transform:
            img = self.transform(img)

        return img, label

# --- Dataset Loader ---
def load_dataset(dataset_path, aug_path, aug_categories, max_images_per_aug_class, batch_size=32, image_size=(128, 128)):
    augmented_images = generate_augmented_images(aug_path, aug_categories, max_images_per_aug_class)
    
    transformations = transforms.Compose([
        transforms.Resize(image_size),
        CustomTransformation(),
        transforms.ToTensor()
    ])

    original_dataset = ImageFolder(dataset_path, transform=None)
    full_dataset = CombinedDataset(original_dataset, augmented_images, transformations)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Visualise batch
def show_batch(dl):
    for images, _  in dl:
        _ ,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break
