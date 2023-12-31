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
from aider import *
from torch.utils.data import DataLoader
# Pennylane
from pennylane import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Plotting
import matplotlib.pyplot as plt
from qnn import DressedQuantumNet
# OpenMP: number of parallel threads.
os.environ["OMP_NUM_THREADS"] = "1"

n_qubits = 4                # Number of qubits
step = 0.0004               # Learning rate
batch_size = 4              # Number of samples for each training step
num_epochs = 20             # Number of training epochs
q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.01              # Initial spread of random quantum weights
start_time = time.time()    # Start of the computation timer
data_dir = "../data/hymenoptera_data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transformed_dataset = AIDER("aider_labels.csv", 'AIDER/', transform=aider_transforms)

total_count = 6432
train_count = int(0.5 * total_count)
valid_count = int(0.2 * total_count)
test_count = total_count - train_count - valid_count
train_set, valid_set, test_set = torch.utils.data.random_split(transformed_dataset,
                                                                (train_count, valid_count, test_count))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)



dataset_sizes = {"train": train_count, "validation": valid_count}
# class_names = image_datasets["train"].classes


dataloaders = {
    'train': train_loader,
     'validation': valid_loader}

# Get a batch of training data
inputs, classes = next(iter(dataloaders["validation"]))


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0  # Large arbitrary number
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number
    print("Training started:")

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ["train", "validation"]:
            if phase == "train":
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = dataset_sizes[phase] // batch_size
            it = 0
            for inputs, labels in dataloaders[phase]:
                since_batch = time.time()
                batch_size_ = len(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # Track/compute gradient and make an optimization step only when training
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Print iteration results
                running_loss += loss.item() * batch_size_
                batch_corrects = torch.sum(preds == labels.data).item()
                running_corrects += batch_corrects
                print(
                    "Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}".format(
                        phase,
                        epoch + 1,
                        num_epochs,
                        it + 1,
                        n_batches + 1,
                        time.time() - since_batch,
                    ),
                    end="\r",
                    flush=True,
                )
                it += 1

            # Print epoch results
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(
                "Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}        ".format(
                    "train" if phase == "train" else "validation  ",
                    epoch + 1,
                    num_epochs,
                    epoch_loss,
                    epoch_acc,
                )
            )

            # Check if this is the best model wrt previous epochs
            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
            if phase == "train" and epoch_acc > best_acc_train:
                best_acc_train = epoch_acc
            if phase == "train" and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss

            # Update learning rate
            if phase == "train":
                scheduler.step()

    # Print final results
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print(
        "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
    )
    print("Best test loss: {:.4f} | Best test accuracy: {:.4f}".format(best_loss, best_acc))
    return model


if __name__ == '__main__':

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model_hybrid = torchvision.models.resnet18(weights=weights)

    for param in model_hybrid.parameters():
        param.requires_grad = False


    # Notice that model_hybrid.fc is the last layer of ResNet18
    model_hybrid.fc = DressedQuantumNet()

    # Use CUDA or CPU according to the "device" object.
    model_hybrid = model_hybrid.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_hybrid = optim.Adam(model_hybrid.fc.parameters(), lr=step)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_hybrid, step_size=10, gamma=gamma_lr_scheduler
    )

    model_hybrid = train_model(model_hybrid, criterion, optimizer_hybrid, exp_lr_scheduler, num_epochs=num_epochs)

    torch.save(model_hybrid.state_dict(), 'hybrid_qnn_model.pt')