import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto 
import copy 
import sys 
import os
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.quantization import QUANT_METHODS
# https://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf

import sys
import os 
import argparse

torch.manual_seed(80)
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="Dataset to use", choices=['gender', 'face'])
parser.add_argument("--qat", type=bool, default=False)
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
EPOCHS = 6
LEARNING_RATE = 0.001
ALPHA = 0.5
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    # transforms.RandomAffine(degrees=45),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])
# load dataset
if args.task == "face":
    train_dataset = ImageFolder(root='../../data/images/rebalanced_train/', transform=transform)
else: 
    train_dataset = ImageFolder('../../data/images/gender_train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# load model and add fc layer
if args.size == 50:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
else:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# change final output layer to fit classification task 
model.fc = nn.Linear(model.fc.in_features, 2) # 2 classes for bianry
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

if args.qat == True:
    def forward_loop(model):
        for image, _ in calib_loader:
            model(image)
    if args.dataset == 'FairFace':
        calib_dataset = ImageFolder(root='../../data/images/calib', transform=transform)
        calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()

# freeze all but last last layer and FC
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
prev_loss = float('inf')
if args.qat == True: # QAT Loop  
    for quant_method in QUANT_METHODS[:2]:
        quant_model = copy.deepcopy(model)
        quant_model = mtq.quantize(quant_model, quant_method[0], forward_loop)
        for name, param in quant_model.named_parameters():
            if not 'layer4' in name:
                param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        quant_model.train()
        quant_model.to(device)
        for epoch in tqdm(range(EPOCHS)):
            total_loss = 0 
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = quant_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss+=loss.item()
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}")
        mto.save(quant_model, f'../../models/quantized/QAT_ResNET{args.size}_{quant_method[1]}.pth')
else: # Regular Train Loop 
    model.to(device)
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        total_loss = 0 
        correct = 0
        samples = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            samples +=labels.size(0)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            output = torch.argmax(outputs, dim=1)
            correct += (output==labels).sum().item()
            total_loss+=loss.item()
        avg_loss = total_loss/samples
        acc = correct/samples
        print(f"Epoch acc{epoch}, Acc: {correct/samples}")
        print(f"Epoch {epoch}, Loss: {total_loss/samples}") 
        # if total_loss/len(train_loader) - prev_loss <= 0.0001:
        #     print("Terminating Early")
        #     break
    print("Saving Model")
    torch.save(model.state_dict(), f"../../models/baseline/{args.task}_ResNET{args.size}_Base.pth")
