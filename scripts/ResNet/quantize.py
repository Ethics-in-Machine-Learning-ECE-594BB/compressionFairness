import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import sys, os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.quantization import static_quantization
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="Dataset to use", choices=['gender', 'face'])
parser.add_argument("--size", type=int, choices=[50,18])
parser.add_argument("--seed", type=int)
args = parser.parse_args()

print("Loading Data and Model")
if args.size == 50: 
    base_model = models.resnet50(weights=None)
else:
    base_model = models.resnet18(weights=None)
base_model.fc = nn.Linear(base_model.fc.in_features, 2) # 2 classes for CelebA dataset
quant_model = base_model
quant_model.load_state_dict(torch.load(f'../../models/baseline/{args.task}_ResNET{args.size}_Base.pth'))
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])
if args.size == 18:
    calib_dataset = ImageFolder(root='../../data/images/18_calib/', transform=transform)
    calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True)
else:
    calib_dataset = ImageFolder(root='../../data/images/50_calib/', transform=transform)
    calib_loader = DataLoader(calib_dataset, batch_size=512, shuffle=True)
def forward_loop(model):
    for image, _ in tqdm(calib_loader):
        model(image)
print("Starting Quantization")
# for quant_method in QUANT_METHODS
static_quantization(quant_model,forward_loop, f'../../models/quantized/Balanced/seed_{args.seed}/{args.task}_ResNET{args.size}_')