import torch
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from torch.utils.data import DataLoader
from src.simpleFCNN import get_teacher_model, get_student_model_8, get_student_model_4
from src.adultIncome import AdultIncomeDataset

# Model paths
MODEL_PATHS = {
    "Teacher (16 neurons)": "models/baseline/teacher_model.pth",
    "Student (8 neurons)": "models/distillation/fcnn_student_model_adult_income_8.pth",
    "Student (4 neurons)": "models/distillation/fcnn_student_model_adult_income_4.pth"
}

# Load dataset
dataset = AdultIncomeDataset("data/raw/adult.csv")
_, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

test_loader = DataLoader(test_dataset, batch_size=32)

# Detect the best available device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Silicon GPU (Metal Performance Shaders)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
else:
    device = torch.device("cpu")   # Default to CPU

# Function to evaluate a model
def evaluate_model(model, model_name):
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"{model_name} Accuracy: {accuracy:.4f}")

# Load and evaluate each model
models = {
    "Teacher (16 neurons)": get_teacher_model(input_size=dataset.X.shape[1]),
    "Student (8 neurons)": get_student_model_8(input_size=dataset.X.shape[1]),
    "Student (4 neurons)": get_student_model_4(input_size=dataset.X.shape[1])
}

for model_name, model in models.items():
    if os.path.exists(MODEL_PATHS[model_name]):  # Ensure model file exists
        model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=device))
        evaluate_model(model, model_name)
    else:
        print(f"Warning: Model file not found for {model_name}. Skipping.")
