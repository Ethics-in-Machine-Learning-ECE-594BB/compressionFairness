import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch import quantization
import torch.utils
from torch.utils.data import DataLoader
import os 
import sys 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset to use to train", choices=['compas', 'adult'])
args = parser.parse_args()
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.simpleFCNN import SimpleFCNN
from src.adultIncome import AdultIncomeDataset
from src.compas import CompasDataset
from src.quantization import fuse_modules
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
TEMPERATURE = 3.0
ALPHA = 0.5

# Load dataset
if args.dataset == 'Adult':
    dataset = AdultIncomeDataset("data/raw/adult.csv")
else:
    dataset = CompasDataset("data/raw/compas.csv")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Set up model for quantize aware training
model = SimpleFCNN(input_size=dataset.X.shape[1], hidden_size = 16, q=True)
model.qconfig = quantization.get_default_qat_qconfig('x86') # NOTE: Might need to change for ARM 

fuse_modules(model)
quantization.prepare_qat(model, inplace=True)
model.to(device)

# train 
criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("Starting Training")
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
        total_loss +=loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

model.eval()
quantized_model = quantization.convert(model.to('cpu'))
torch.save(quantized_model.state_dict(), f"../../models/quantized/quant_aware_train_{args.dataset}.pth")
print("Quantized Aware Training Model Saved")