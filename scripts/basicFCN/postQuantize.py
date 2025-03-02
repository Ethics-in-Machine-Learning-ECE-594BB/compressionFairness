import torch
from torch.utils.data import DataLoader
import sys
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset to use to train", choices=['compas', 'adult'])
args = parser.parse_args()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.quantization import static_quantization
from src.adultIncome import AdultIncomeDataset
from src.simpleFCNN import SimpleFCNN
from src.compas import CompasDataset
 
# Load dataset
if args.dataset == 'Adult':
    dataset = AdultIncomeDataset("data/raw/adult.csv")
else:
    dataset = CompasDataset("data/raw/compas.csv")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset)
# print("Pre-Quantization Size in Memory: ", os.path.getsize("../../models/baseline/teacher_model_adult.pth")/1e3, " KB")
# model = SimpleFCNN(dataset.X.shape[1], 16)
# model.load_state_dict(torch.load(f'../../models/baseline/teacher_model_{args.dataset}.pth', weights_only=True))
print("Starting Quantization")
quantized_model = SimpleFCNN(14, 16, q=True)
quantized_model.load_state_dict(torch.load(f'../../models/baseline/teacher_model_{args.dataset}.pth', weights_only=True))
static_quantization(quantized_model,test_loader, f'../../models/quantized/static_quantization_{args.dataset}.pth')

# dynamic quantization done at runtime so need to wait for full evaluation code
