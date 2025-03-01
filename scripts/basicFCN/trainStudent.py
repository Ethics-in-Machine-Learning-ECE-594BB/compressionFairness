import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset to use to train", choices=['compas', 'adult'])
args = parser.parse_args()
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from torch.utils.data import DataLoader
from src.simpleFCNN import SimpleFCNN, get_teacher_model, get_student_model_8, get_student_model_4
from src.adultIncome import AdultIncomeDataset
from src.distillation import DistillationLoss
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

# Load teacher model
teacher_model = get_teacher_model(input_size=dataset.X.shape[1]).to(device)
teacher_model.load_state_dict(torch.load("/Users/ethan3048/Documents/school/winter25/ece594bbEthics/compressionFairness/models/baseline/teacher_model_adult.pth"))
teacher_model.eval()

# Choose student model (8 or 4 neurons)
# student_model = get_student_model_8(input_size=dataset.X.shape[1]).to(device)
student_model = get_student_model_4(input_size=dataset.X.shape[1]).to(device)

# Loss & Optimizer
criterion = DistillationLoss(alpha=ALPHA, temperature=TEMPERATURE)
optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

# Train loop
for epoch in range(EPOCHS):
    student_model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(inputs)

        optimizer.zero_grad()
        student_logits = student_model(inputs)

        loss = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

# Save student model
torch.save(student_model.state_dict(), f"/Users/ethan3048/Documents/school/winter25/ece594bbEthics/compressionFairness/models/distillation/fcnn_student_model_{args.dataset}_income_4.pth")
print("Student model (8 neurons) saved successfully!")
