"""
    @author: Van Toan <damtoan321@gmail.com>
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
import cv2


class myAnimalsDataset(Dataset):
    def __init__(self, root="Dataset/animals", train=True, transform=None):
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]

        self.transform = transform
        self.image_paths = []
        self.labels = []

        mode = "train" if train else "test"
        data_file = os.path.join(root, mode)
        for category in self.categories:
            category_path = os.path.join(data_file, category)
            for image_path in os.listdir(category_path):
                self.image_paths.append(os.path.join(category_path, image_path))
                self.labels.append(self.categories.index(category))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label