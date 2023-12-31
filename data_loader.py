import torch
import torchvision
import torchvision.transforms as transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 training dataset and filter classes
selected_classes = ['cat', 'dog']
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Filter dataset to include only 'cat' and 'dog' classes
train_dataset.data = [img for img, label in zip(train_dataset.data, train_dataset.targets) if train_dataset.classes[label] in selected_classes]
train_dataset.targets = [train_dataset.classes.index(cls) for cls in selected_classes for label in train_dataset.targets]

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

# Load CIFAR-10 testing dataset and filter classes
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Filter dataset to include only 'cat' and 'dog' classes
test_dataset.data = [img for img, label in zip(test_dataset.data, test_dataset.targets) if test_dataset.classes[label] in selected_classes]
test_dataset.targets = [test_dataset.classes.index(cls) for cls in selected_classes for label in test_dataset.targets]

# Create DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
