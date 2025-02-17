import torch

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from torch.utils.data import DataLoader
from src.simpleFCNN import SimpleFCNN
from src.adultIncome import AdultIncomeDataset

# Load dataset
dataset = AdultIncomeDataset("data/raw/adult.csv")
_, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

test_loader = DataLoader(test_dataset, batch_size=32)

# Detect the best available device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Silicon GPU (Metal Performance Shaders)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
else:
    device = torch.device("cpu")   # Default to CPU

model = SimpleFCNN(input_size=dataset.X.shape[1]).to(device)
model.load_state_dict(torch.load("/Users/ethan3048/Documents/school/winter25/ece594bbEthics/compressionFairness/models/baseline/fcnn_model_adult_income.pth"))
model.eval()

# Evaluate model
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
