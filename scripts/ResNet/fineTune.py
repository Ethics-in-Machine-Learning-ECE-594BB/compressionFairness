import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
# https://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf

import sys
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset to use", choices=['CelebA'] )
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TEMPERATURE = 3.0
ALPHA = 0.5

# load dataset
if args.dataset == "CelebA":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])
    train_dataset = ImageFolder(root='../../data/images/celeb_train_set/', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# load model and add fc layer
model = models.resnet18(pretrained=True)
if args.dataset =='CelebA':
    model.fc = nn.Linear(model.fc.in_features, 2) # 2 classes for CelebA dataset
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# freeze all but last layer 
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
model.to(device)

# Train Loop
for epoch in range(EPOCHS):
    total_loss = 0 
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}")

print("Saving Model")
torch.save(model.state_dict(), f"../../models/ResNet18/base")
