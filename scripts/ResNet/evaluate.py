import torch 
import argparse
import copy
import os
import sys
import pandas as pd 
import torch.nn as nn 
import modelopt.torch.opt as mto
import torchvision.transforms as transforms
from glob import glob
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, choices=[18,50])
args = parser.parse_args()
# gender label 0 Male, 1 Female
# race 0 East Asian, 1 Indian, 2 Black, 3 White, 4 Middle Eastern, 5 Latino, 6 Southeast Asian 


sens_attr_labels = pd.read_csv('../../data/raw/FairFace_test_attr.csv')
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Silicon GPU (Metal Performance Shaders)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
else:
    device = torch.device("cpu")   # Default to CPU

# LOAD MODELS and EVALUATE
if args.size == 18:
    model_paths = glob('../../models/quantized/*ResNET18*.pth')
    base = models.resnet18(weights=None)
    base.fc = nn.Linear(base.fc.in_features, 2)
else:
    model_paths = glob('../../models/quantized/ResNET50*.pth')
    base = models.resnet50(weights=None)
    base.fc = nn.Linear(base.fc.in_features, 2)

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomPerspective(),
        transforms.RandomAffine(degrees=45),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])
test_dataset = ImageFolder('../../data/images/test/', transform=transform)
indices = [i for i in range(5999)]
faces_data = Subset(test_dataset, indices)
test_loader = DataLoader(faces_data, batch_size=512, shuffle=False)
attributes = pd.read_csv('../../data/raw/FairFace_test_attr.csv')

for model in model_paths:
    print(f'ON {model}')
    o_m = copy.deepcopy(base)
    o_m = mto.restore(o_m, model).to(device)
    o_m.eval()
    predictions = []
    with torch.no_grad():
        for image, _ in test_loader:
            image = image.to(device)
            output = o_m(image)
            predicted = torch.argmax(output, dim=1).cpu().numpy()
            predictions.extend(predicted)
    pred_df = pd.DataFrame({"prediction": predictions})
    results = pd.concat([attributes, pred_df], axis=1)
    path = os.path.basename(model)[:-4]
    results.to_csv(f'../../results/ResFair/{path}')
# SAVE METRICS TO RESULTS FOLDER 