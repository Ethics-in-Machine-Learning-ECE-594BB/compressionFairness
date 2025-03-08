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

print("Loading Data and Model")
base_model = models.resnet18()
base_model.fc = nn.Linear(base_model.fc.in_features, 2) # 2 classes for CelebA dataset
quant_model = base_model
quant_model.load_state_dict(torch.load('../../models/baseline/ResNET18_Base.pth'))
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])
calib_dataset = ImageFolder(root='../../data/images/calib/', transform=transform)

calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True)
def forward_loop(model):
    for image, _ in calib_loader:
        model(image)
print("Starting Quantization")
static_quantization(quant_model,forward_loop, '../../models/quantized/ResNET_')