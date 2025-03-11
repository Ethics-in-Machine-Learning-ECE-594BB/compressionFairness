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
import torch.nn.functional as F
from torch.autograd import Function

'''
To train a ResNet50 with adversarial debiasing for gender fairness:
python fineTune.py --size 50 --debiasing adversarial --protected_attr 0 --adv_weight 0.3
To train with fairness constraints for demographic parity:
python fineTune.py --size 50 --debiasing fairness_constraint --fairness_type demographic_parity --protected_attr 0 --lambda_fair 0.4
'''

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.quantization import QUANT_METHODS
# https://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf

import sys
import os 
import argparse

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for adversarial fairness training
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DiscriminatorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(DiscriminatorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class FairResNet(nn.Module):
    """ResNet wrapper for adversarial fairness training"""
    def __init__(self, base_model):
        super(FairResNet, self).__init__()
        self.base_model = base_model
        
        # Remove the final layer to get features
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Task classifier (using the original fc layer)
        self.task_classifier = base_model.fc
        
    def forward(self, x):
        # Get features (before the final fc layer)
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Task prediction
        task_output = self.task_classifier(features)
        
        return task_output, features

class ImprovedFairnessLoss(nn.Module):
    def __init__(self, fairness_type='demographic_parity', lambda_fair=0.5):
        """
        Fairness-aware loss function for image classification
        
        Args:
            fairness_type: Type of fairness constraint ('demographic_parity', 'equalized_odds')
            lambda_fair: Weight for the fairness term
        """
        super(ImprovedFairnessLoss, self).__init__()
        self.base_criterion = nn.CrossEntropyLoss(reduction='none')
        self.fairness_type = fairness_type
        self.lambda_fair = lambda_fair
        
    def forward(self, outputs, targets, protected_attrs):
        # Base loss
        base_loss = self.base_criterion(outputs, targets)
        
        # Get predicted classes
        _, preds = torch.max(outputs, 1)
        
        # For binary protected attribute
        mask_protected = (protected_attrs > 0.5).float()
        mask_unprotected = 1 - mask_protected
        
        # Ensure we have samples from both groups
        has_protected = torch.sum(mask_protected) > 0
        has_unprotected = torch.sum(mask_unprotected) > 0
        
        # If either group is missing, return only the base loss
        if not (has_protected and has_unprotected):
            return torch.mean(base_loss)
        
        if self.fairness_type == 'demographic_parity':
            # Safe mean calculation for prediction rates
            pred_protected = safe_mean((preds == 1).float(), mask_protected)
            pred_unprotected = safe_mean((preds == 1).float(), mask_unprotected)
            fairness_loss = torch.abs(pred_protected - pred_unprotected)
            
        elif self.fairness_type == 'equalized_odds':
            # Positive and negative ground truth masks
            pos_mask = (targets == 1).float()
            neg_mask = 1 - pos_mask
            
            # Group masks
            prot_pos = mask_protected * pos_mask
            prot_neg = mask_protected * neg_mask
            unprot_pos = mask_unprotected * pos_mask
            unprot_neg = mask_unprotected * neg_mask
            
            # Calculate TPR and FPR for each group
            tpr_prot = safe_mean((preds == 1).float(), prot_pos)
            tpr_unprot = safe_mean((preds == 1).float(), unprot_pos)
            fpr_prot = safe_mean((preds == 1).float(), prot_neg)
            fpr_unprot = safe_mean((preds == 1).float(), unprot_neg)
            
            # Combined fairness loss
            fairness_loss = torch.abs(tpr_prot - tpr_unprot) + torch.abs(fpr_prot - fpr_unprot)
        else:
            raise ValueError(f"Unknown fairness type: {self.fairness_type}")
        
        # Combine losses
        total_loss = torch.mean(base_loss) + self.lambda_fair * fairness_loss
        return total_loss

def safe_mean(tensor, mask):
    """Safely compute mean even when mask has no positive values"""
    sum_values = torch.sum(tensor * mask)
    count = torch.sum(mask)
    if count > 0:
        return sum_values / count
    else:
        return torch.tensor(0.0, device=tensor.device)

# Define training functions for different fairness approaches
def train_standard(model, train_loader, criterion, optimizer, epochs, device):
    """Standard training without fairness constraints"""
    model.to(device)
    for epoch in tqdm(range(epochs)):
        total_loss = 0 
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
    return model

def train_with_fairness_constraint(model, train_loader, epochs, device, 
                                   protected_attr_idx=0,
                                   fairness_type='demographic_parity',
                                   initial_lambda=0.1, final_lambda=0.5):
    """Training with fairness constraints"""
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        
        # Calculate current lambda using linear schedule
        lambda_fair = initial_lambda + (final_lambda - initial_lambda) * (epoch / epochs)
        criterion = ImprovedFairnessLoss(fairness_type=fairness_type, lambda_fair=lambda_fair)
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # For FairFace, extract protected attributes (assuming they're encoded in the labels)
            # This needs to be adapted to your specific dataset structure
            protected_attrs = (labels == protected_attr_idx).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute loss with fairness constraints
            loss = criterion(outputs, labels, protected_attrs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss with Fairness: {total_loss/len(train_loader):.4f}, "
              f"Lambda: {lambda_fair:.4f}")
    
    return model

def train_adversarial(model, train_loader, epochs, device, protected_attr_idx=0,
                     initial_adv_weight=0.1, final_adv_weight=0.5):
    """Training with adversarial debiasing"""
    # Wrap the model to get features
    fair_model = FairResNet(model)
    fair_model.to(device)
    
    # Create discriminator for the protected attribute
    # Get the feature dimension from the model
    feature_dim = model.fc.in_features
    discriminator = DiscriminatorNetwork(input_size=feature_dim).to(device)
    
    # Optimizers
    model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, fair_model.parameters()), lr=LEARNING_RATE)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE*0.5)
    
    # Loss functions
    task_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.BCELoss()
    
    for epoch in tqdm(range(epochs)):
        fair_model.train()
        discriminator.train()
        total_task_loss = 0
        total_adv_loss = 0
        
        # Calculate current adversarial weight
        adv_weight = initial_adv_weight + (final_adv_weight - initial_adv_weight) * (epoch / epochs)
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Extract protected attributes (assuming they're encoded in the labels)
            # This needs to be adapted to your specific dataset structure
            protected_attrs = (labels == protected_attr_idx).float().unsqueeze(1)
            
            # Phase 1: Train discriminator less frequently
            if batch_idx % 3 == 0:
                disc_optimizer.zero_grad()
                
                # Get features without tracking gradients for main model
                with torch.no_grad():
                    _, features = fair_model(inputs)
                    
                disc_output = discriminator(features)
                disc_loss = adv_criterion(disc_output, protected_attrs)
                disc_loss.backward()
                disc_optimizer.step()
            
            # Phase 2: Train main model
            model_optimizer.zero_grad()
            task_output, features = fair_model(inputs)
            
            # Task loss
            task_loss = task_criterion(task_output, labels)
            task_loss.backward(retain_graph=True)
            
            # Adversarial component - train model to confuse discriminator
            disc_output = discriminator(features)
            adv_loss = adv_criterion(disc_output, protected_attrs)
            
            # Apply negative gradient for adversarial component
            (-adv_weight * adv_loss).backward()
            
            model_optimizer.step()
            
            total_task_loss += task_loss.item()
            total_adv_loss += adv_loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Task Loss: {total_task_loss/len(train_loader):.4f}, "
              f"Adv Loss: {total_adv_loss/len(train_loader):.4f}, "
              f"Adv Weight: {adv_weight:.4f}")
    
    # Return the base model (without wrapper)
    return model

torch.manual_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset to use", choices=['FairFace'], default='FairFace')
parser.add_argument("--qat", type=bool, default=False)
parser.add_argument("--size", type=int, help="Select Resnet 18 or 50", choices=[18,50], default=50)
# Add these arguments for fairness mitigation
parser.add_argument("--debiasing", type=str, default="none", 
                   choices=['none', 'adversarial', 'fairness_constraint'],
                   help="Debiasing method to use")
parser.add_argument("--fairness_type", type=str, default="demographic_parity",
                   choices=['demographic_parity', 'equalized_odds'],
                   help="Type of fairness constraint to apply")
parser.add_argument("--protected_attr", type=int, default=0,
                   help="Index of the protected attribute in the data (typically gender or race)")
parser.add_argument("--lambda_fair", type=float, default=0.5,
                   help="Weight for fairness term in loss function")
parser.add_argument("--adv_weight", type=float, default=0.2,
                   help="Initial weight for adversarial loss term")
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

BATCH_SIZE = 512
EPOCHS = 7
LEARNING_RATE = 0.001
ALPHA = 0.5

# load dataset
if args.dataset == "FairFace":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        # transforms.RandomAffine(degrees=45),
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

if args.qat == True:
    def forward_loop(model):
        for image, _ in calib_loader:
            model(image)
    if args.dataset == 'FairFace':
        calib_dataset = ImageFolder(root='../../data/images/calib', transform=transform)
        calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()

# freeze all but last last layer and FC
for name, param in model.named_parameters():
    if not 'layer4' in name:
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
# else: # Regular Train Loop 
#     model.to(device)
#     for epoch in tqdm(range(EPOCHS)):
#         total_loss = 0 
#         for inputs, labels in tqdm(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss+=loss.item()
#         print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}") 
#         # if total_loss/len(train_loader) - prev_loss <= 0.0001:
#         #     print("Terminating Early")
#         #     break
#     print("Saving Model")
#     torch.save(model.state_dict(), f"../../models/baseline/ResNET{args.size}_Base.pth")
else:  # Regular Train Loop with fairness options
    if args.debiasing == 'adversarial':
        print("Using Adversarial Debiasing for fairness")
        model = train_adversarial(
            model, 
            train_loader, 
            EPOCHS, 
            device, 
            protected_attr_idx=args.protected_attr,
            initial_adv_weight=0.1, 
            final_adv_weight=args.adv_weight
        )
        torch.save(model.state_dict(), f"../../models/debiased/adv_ResNET{args.size}_{args.protected_attr}.pth")
    
    elif args.debiasing == 'fairness_constraint':
        print(f"Using Fairness Constraint: {args.fairness_type}")
        model = train_with_fairness_constraint(
            model, 
            train_loader, 
            EPOCHS, 
            device, 
            protected_attr_idx=args.protected_attr,
            fairness_type=args.fairness_type, 
            initial_lambda=0.1, 
            final_lambda=args.lambda_fair
        )
        torch.save(model.state_dict(), 
                  f"../../models/debiased/fair_{args.fairness_type}_ResNET{args.size}_{args.protected_attr}.pth")
    
    else:  # Default baseline training without debiasing
        print("Using standard training (no debiasing)")
        model = train_standard(model, train_loader, criterion, optimizer, EPOCHS, device)
        torch.save(model.state_dict(), f"../../models/baseline/ResNET{args.size}_Base.pth")
