import time
import os
import copy

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

# Pennylane
from pennylane import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Plotting
import matplotlib.pyplot as plt
from qnn import DressedQuantumNet
# OpenMP: number of parallel threads.

os.environ["OMP_NUM_THREADS"] = "1"

batch_size = 4              # Number of samples for each training step
q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
start_time = time.time()    # Start of the computation timer
data_dir = "../data/hymenoptera_data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    "train": transforms.Compose(
        [
            # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
            # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalize input channels using mean values and standard deviations of ImageNet.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

image_datasets = {
    x if x == "train" else "validation": datasets.ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
class_names = image_datasets["train"].classes

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
     'validation':torch.utils.data.DataLoader(image_datasets['validation'], batch_size=batch_size, shuffle=True)}

# Get a batch of training data
inputs, classes = next(iter(dataloaders["validation"]))
criterion = nn.CrossEntropyLoss()
model = torch.load('hybrid_qnn_model.pt')
model.eval()

running_loss = 0.0
running_corrects = 0

n_batches = dataset_sizes['validation'] // batch_size

for images,labels in dataloaders['test']:
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, labels)
    running_loss += loss.item() * batch_size
    batch_corrects = torch.sum(preds == labels.data).item()
    running_corrects += batch_corrects


accuracy = running_corrects / dataset_sizes['validation']
loss = running_loss / dataset_sizes['validation']

print('Test Accuracy', accuracy)
print('Test Loss', loss)


