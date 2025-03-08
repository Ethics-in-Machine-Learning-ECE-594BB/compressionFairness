import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
# https://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf

import sys
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset to use", choices=['FairFace'], default='FairFace')
parser.add_argument("--size", type=int, help="Select Resnet 18 or 50", choices=[18,50], default=50)
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

BATCH_SIZE = 512
EPOCHS = 2
LEARNING_RATE = 0.001
ALPHA = 0.5

# load dataset
if args.dataset == "FairFace":
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomPerspective(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])
    train_dataset = ImageFolder(root='../../data/images/train/', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# load model and add fc layer
if args.size == 50:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
else:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# change final output layer to fit classification task 
if args.dataset =='FairFace':
    model.fc = nn.Linear(model.fc.in_features, 2) # 2 classes for CelebA dataset
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
# include fair regularizer here in future
criterion = nn.CrossEntropyLoss()

# freeze all but last layer 
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
model.to(device)

prev_loss = float('inf')
# Train Loop
for epoch in tqdm(range(EPOCHS)):
    total_loss = 0 
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}") 
    # if total_loss/len(train_loader) - prev_loss <= 0.0001:
    #     print("Terminating Early")
    #     break
print("Saving Model")
torch.save(model.state_dict(), f"../../models/baseline/ResNET{args.size}_Base.pth")
