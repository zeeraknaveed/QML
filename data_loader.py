import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset and filter classes
selected_classes = ['cat', 'dog']
full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Filter dataset to include only 'cat' and 'dog' classes
filtered_data = [(img, label) for img, label in zip(full_dataset.data, full_dataset.targets) if full_dataset.classes[label] in selected_classes]
filtered_targets = [full_dataset.classes.index(cls) for cls in selected_classes for _, label in filtered_data]

# Create Dataset objects
filtered_dataset = torch.utils.data.TensorDataset(torch.tensor(filtered_data), torch.tensor(filtered_targets))

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(filtered_dataset))
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

# Create DataLoader instances for train, validation, and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Load CIFAR-10 testing dataset and filter classes
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Filter dataset to include only 'cat' and 'dog' classes
test_dataset.data = [img for img, label in zip(test_dataset.data, test_dataset.targets) if test_dataset.classes[label] in selected_classes]
test_dataset.targets = [test_dataset.classes.index(cls) for cls in selected_classes for label in test_dataset.targets]

# Create DataLoader for the test set
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)