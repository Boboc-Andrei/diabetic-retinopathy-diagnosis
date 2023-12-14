import numpy as np
import torch
import torchvision
from torch.nn import Linear, ReLU, Dropout, Hardswish, CrossEntropyLoss, Conv2d, Module
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.models import vgg16_bn
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    
transforms = v2.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # convert PIL image to torch tensor - float32 [0,1]
    v2.Resize((500,500), interpolation=v2.InterpolationMode.NEAREST),
    ])

class vgg_DR(Module):
    def __init__(self):
        super(vgg_DR, self).__init__()
        self.model = vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT)
        self.model.classifier[-1] = Linear(self.model.classifier[-1].in_features, out_features=4, bias=True)

    def forward(self,x):
        return torch.softmax(self.model(x),dim=1)
    
classes_dict = {
    0: 'No Dr',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative DR'
}

t2img = v2.ToPILImage()
img2t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
 
images_dir = os.path.join('Dataset','DR_dataset_70k', 'resized_train_cropped','resized_train_cropped')
annot_path = os.path.join('Dataset','DR_dataset_70k', 'trainLabels_cropped.csv')
dataset = DRdataset(annotations_file=annot_path, img_dir=images_dir, transform=transforms)


BATCH_SIZE = 4
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

import itertools
train_dataloader = itertools.islice(train_dataloader,20)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)




def train_model(model, loss_func, optimizer, scheduler, num_epochs):

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            img, label = batch
            img = img.to(device)

            #   FORWARD
            output = model(img)
            loss = loss_func(output,label)

            #   BACKWARD

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(loss.item())
    
    return model


batch,label = next(iter(train_dataloader))
batch = batch.to(device)

NUM_EPOCHS = 1
LEARNING_RATE = 1e-4




model = vgg_DR().to(device)


loss_func = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

out = model(batch).cpu()
print(out)
print(label)
print(loss_func(out,label).item())

#model = train_model(model, loss_func, optimizer, scheduler, NUM_EPOCHS)

model_save_path = os.path.join('Models', 'vgg16_bn.pth')
torch.save(model.state_dict(), model_save_path)