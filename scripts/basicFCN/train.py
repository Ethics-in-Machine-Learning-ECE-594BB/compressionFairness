import torch
import torch.optim as optim
import torch.nn as nn
import sys
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset to use to train", choices=['compas', 'adult'])
args = parser.parse_args()


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from torch.utils.data import DataLoader
from src.simpleFCNN import SimpleFCNN
from src.adultIncome import AdultIncomeDataset
from src.compas import CompasDataset

# Detect the best available device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Silicon GPU (Metal Performance Shaders)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
else:
    device = torch.device("cpu")   # Default to CPU

print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Load dataset
if args.dataset == 'Adult':
    dataset = AdultIncomeDataset("data/raw/adult.csv")
else:
    dataset = CompasDataset("data/raw/compas.csv")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model setup
model = SimpleFCNN(input_size=dataset.X.shape[1], hidden_size=16).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), f"models/baseline/fcnn_model_{args.dataset}_income.pth")
print("Model saved successfully!")
