import numpy as np
import torch
import torchvision
from torch.nn import Linear, ReLU, Dropout, Hardswish, CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd



class DRdataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2]+'.jpeg')
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
images_dir = os.path.join('Dataset','DR_dataset_70k', 'resized_train_cropped','resized_train_cropped')
annot_path = os.path.join('Dataset','DR_dataset_70k', 'trainLabels_cropped.csv')
dataset = DRdataset(annotations_file=annot_path, img_dir=images_dir)




BATCH_SIZE = 4
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


