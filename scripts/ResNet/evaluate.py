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
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, choices=[18,50])
parser.add_argument("--version", type=str, choices=['base', 'quant', 'pruned'])
parser.add_argument('--task', type=str, choices=['face', 'gender'])
args = parser.parse_args()
# gender label 1 Male, 0 Female
# race 0 East Asian, 1 Indian, 2 Black, 3 White, 4 Middle Eastern, 5 Latino, 6 Southeast Asian 


sens_attr_labels = pd.read_csv('../../data/raw/FairFace_test_attr.csv')
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Silicon GPU (Metal Performance Shaders)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
else:
    device = torch.device("cpu")   # Default to CPU

# LOAD MODELS and EVALUATE
if args.version == 'quant':
    if args.size == 18:
        model_paths = glob(f"../../models/quantized/{args.task}_ResNET18*.pth")
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, 2)
    else:
        model_paths = glob(f"../../models/quantized/{args.task}_ResNET50*.pth")
        base = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, 2)
elif args.version == 'base':
    if args.size == 18:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(f"../../models/baseline/{args.task}_ResNET18_Base.pth"))
    else:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(f"../../models/baseline/{args.task}_ResNET50_Base.pth"))
    model.to(device)

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomPerspective(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])
if args.task == 'face':
    test_dataset = ImageFolder('../../data/images/test/', transform=transform)
    indices = [i for i in range(5999)]
    test_dataset = Subset(test_dataset, indices)
    attributes = pd.read_csv('../../data/raw/FairFace_test_attr.csv')
else:
    test_dataset = ImageFolder('../../data/images/gender_test/', transform=transform)
    attributes = pd.read_csv('../../data/raw/gender_class.csv')
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

if args.version == 'quant':
    for model in tqdm(model_paths):
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
        results.to_csv(f'../../results/ResFair/{path}.csv')
elif args.version == 'base':
    model.eval()
    predictions = []
    with torch.no_grad():
        correct = 0
        sample = 0
        for image, labels in test_loader:
            image = image.to(device)
            labels = labels.to(device)
            sample +=labels.size(0)
            output = model(image)
            predicted = torch.argmax(output, dim=1)
            correct += (predicted==labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
        pred_df = pd.DataFrame({"prediction": predictions})
        results = pd.concat([attributes, pred_df], axis=1)
        results.to_csv(f'../../results/ResFair/{args.task}_Base_ResNET{args.size}.csv') 
        print(f"Acc {correct/sample}")
# SAVE METRICS TO RESULTS FOLDER 