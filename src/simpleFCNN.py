import torch
import torch.nn as nn

class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # ✅ Apply sigmoid correctly
        return x


# Teacher Model (16 neurons)
def get_teacher_model(input_size):
    return SimpleFCNN(input_size, 16)

# Student Model (8 neurons)
def get_student_model_8(input_size):
    return SimpleFCNN(input_size, 8)

# Student Model (4 neurons)
def get_student_model_4(input_size):
    return SimpleFCNN(input_size, 4)
