import torch
import torch.nn as nn

class SimpleFCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Fully connected layer with 16 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # Output layer (binary classification)
        self.sigmoid = nn.Sigmoid()  # Ensures output is between 0 and 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
