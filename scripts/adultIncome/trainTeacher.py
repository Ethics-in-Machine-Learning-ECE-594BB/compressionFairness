import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from torch.utils.data import DataLoader
from src.simpleFCNN import SimpleFCNN, get_teacher_model, get_principle_model
from src.adultIncome import AdultIncomeDataset

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
dataset = AdultIncomeDataset("data/raw/adult.csv")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = get_teacher_model(input_size=dataset.X.shape[1]).to(device)
model = get_principle_model(input_size=dataset.X.shape[1]).to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train loop
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

# Save teacher model
torch.save(model.state_dict(), "models/baseline/teacher_model_adult.pth")
print("Teacher model saved successfully!")
