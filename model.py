import torch
import torch.nn as nn
import torch.nn.functional as F

class MyClassicalFC(nn.Module):
    def __init__(self):
        super(MyClassicalFC, self).__init__()
        
        # First linear layer with 512 input and 64 output
        self.fc1 = nn.Linear(512, 64)
        
        # Last linear layer with 64 input and 2 output
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        # Forward pass with softmax activation
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
