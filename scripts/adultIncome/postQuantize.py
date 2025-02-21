import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.quantization import static_quantization
from src.adultIncome import AdultIncomeDataset
from src.simpleFCNN import SimpleFCNN

# TODO finish writing out script here 
dataset = AdultIncomeDataset("../../data/raw/adult.csv")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset)
# print("Pre-Quantization Size in Memory: ", os.path.getsize("../../models/baseline/teacher_model_adult.pth")/1e3, " KB")
model = SimpleFCNN(dataset.X.shape[1], 16)
model.load_state_dict(torch.load('../../models/baseline/teacher_model_adult.pth', weights_only=True))
print("Starting Quantization")
static_quantization(model, test_loader, '../../models/quantized/static_quantization.pth')

# Dynamic 