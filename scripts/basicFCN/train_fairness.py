import torch
import torch.optim as optim
import torch.nn as nn
import sys
import os
import argparse 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.quantization import QuantStub, DeQuantStub

'''
To train with this script:
Baseline - 
python train_fairness.py --dataset adult --debiasing none --protected_attr sex

Adversarial debiasing - 
python train_fairness.py --dataset adult --debiasing adversarial --protected_attr sex --adv_weight 0.5

Fairness constraint debiasing - 
python train_fairness.py --dataset adult --debiasing fairness_constraint --protected_attr sex --fairness_type demographic_parity --lambda_fair 0.3

'''

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for Fair Representation Learning
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

class FairRepresentationFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, feature_size=12, alpha=1.0):
        super(FairRepresentationFCNN, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, feature_size),
            nn.ReLU()
        )
        
        # Task predictor (main classifier)
        self.task_predictor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Gradient reversal for protected attribute classifier
        self.gradient_reversal = GradientReversalLayer(alpha=alpha)
        
        # Protected attribute predictor
        self.protected_attr_predictor = nn.Sequential(
            nn.Linear(feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Task prediction (main objective)
        task_output = self.task_predictor(features)
        
        # Protected attribute prediction (adversarial objective)
        reversed_features = self.gradient_reversal(features)
        protected_attr_output = self.protected_attr_predictor(reversed_features)
        
        return task_output, protected_attr_output

def train_fair_representation(model, train_loader, test_loader, protected_attr_idx, device, 
                             epochs=10, learning_rate=0.001, task_weight=1.0, adv_weight=1.0):
    """
    Train with fair representation learning using gradient reversal
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    task_criterion = nn.BCELoss()
    protected_criterion = nn.BCELoss()
    
    # Find min and max values of protected attribute for normalization
    attr_min, attr_max = find_protected_attr_range(train_loader, protected_attr_idx)
    
    for epoch in range(epochs):
        model.train()
        total_task_loss = 0
        total_protected_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Extract protected attributes and normalize to [0,1]
            protected_attrs = inputs[:, protected_attr_idx].unsqueeze(1)
            protected_attrs = normalize_protected_attr(protected_attrs, attr_min, attr_max)
            
            optimizer.zero_grad()
            
            # Forward pass
            task_output, protected_attr_output = model(inputs)
            
            # Calculate losses
            task_loss = task_criterion(task_output, labels)
            protected_loss = protected_criterion(protected_attr_output, protected_attrs)
            
            # Combined loss - no need to negate protected_loss due to gradient reversal
            loss = task_weight * task_loss + adv_weight * protected_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_task_loss += task_loss.item()
            total_protected_loss += protected_loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Task Loss: {total_task_loss/len(train_loader):.4f}, "
              f"Protected Attr Loss: {total_protected_loss/len(train_loader):.4f}")
        
        # Evaluate on test set every few epochs
        if (epoch + 1) % 2 == 0:
            evaluate_fair_representation(model, test_loader, protected_attr_idx, attr_min, attr_max, device)
    
    return model

def evaluate_fair_representation(model, test_loader, protected_attr_idx, attr_min, attr_max, device):
    """Evaluate fair representation model on test set"""
    model.eval()
    task_correct = 0
    protected_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            protected_attrs = inputs[:, protected_attr_idx].unsqueeze(1)
            protected_attrs = normalize_protected_attr(protected_attrs, attr_min, attr_max)
            
            task_output, protected_attr_output = model(inputs)
            
            # Task prediction accuracy
            task_preds = (task_output >= 0.5).float()
            task_correct += (task_preds == labels).sum().item()
            
            # Protected attribute prediction accuracy
            # Lower accuracy is better here (means model can't predict protected attributes)
            protected_preds = (protected_attr_output >= 0.5).float()
            protected_correct += (protected_preds == protected_attrs).sum().item()
            
            total += labels.size(0)
    
    task_acc = task_correct / total
    protected_acc = protected_correct / total
    
    print(f"  Test Evaluation:")
    print(f"  - Task Accuracy: {task_acc:.4f}")
    print(f"  - Protected Attribute Accuracy: {protected_acc:.4f} (lower is better)")
    print(f"  - Ideal Protected Attr Accuracy: 0.5 (random guessing)")

class DiscriminatorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=8):
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

class AdversarialFCNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdversarialFCNN, self).__init__()
        # Use the same layer names as the FCNN class
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Create a separate feature extractor function but keep the same architecture
        # as the original FCNN for state_dict compatibility
    
    def forward(self, x):
        # Regular forward pass - identical to the FCNN
        x = self.fc1(x)
        features = self.relu(x)  # Store features after activation
        x = self.fc2(features)
        x = self.sigmoid(x)
        return x, features
    
    def get_compatible_state_dict(self):
        """Returns a state_dict compatible with standard FCNN loading"""
        return {
            'fc1.weight': self.fc1.weight,
            'fc1.bias': self.fc1.bias,
            'fc2.weight': self.fc2.weight,
            'fc2.bias': self.fc2.bias
        }

def find_protected_attr_range(data_loader, protected_attr_idx):
    """Find the min and max values of the protected attribute in the dataset"""
    min_val = float('inf')
    max_val = float('-inf')
    
    for inputs, _ in data_loader:
        protected_attrs = inputs[:, protected_attr_idx]
        batch_min = protected_attrs.min().item()
        batch_max = protected_attrs.max().item()
        
        min_val = min(min_val, batch_min)
        max_val = max(max_val, batch_max)
    
    return min_val, max_val

def normalize_protected_attr(protected_attrs, attr_min, attr_max):
    """Normalize protected attributes to [0,1] range"""
    if attr_min == attr_max:  # Handle binary case
        return (protected_attrs > attr_min).float()
    return (protected_attrs - attr_min) / (attr_max - attr_min + 1e-8)

# Improved adversarial training function
def train_adversarial(model, discriminator, train_loader, test_loader, 
                      protected_attr_idx, device, epochs=10, 
                      learning_rate=0.001):
    """
    Training with adversarial debiasing - fixed implementation
    """
    # Task optimizer
    task_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Discriminator optimizer
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate*0.5)  # Slower discriminator
    
    # Loss functions
    task_criterion = nn.BCELoss()
    adv_criterion = nn.BCELoss()
    
    # Find min and max values of protected attribute for normalization
    attr_min = float('inf')
    attr_max = float('-inf')
    for inputs, _ in train_loader:
        protected_attrs = inputs[:, protected_attr_idx]
        batch_min = protected_attrs.min().item()
        batch_max = protected_attrs.max().item()
        attr_min = min(attr_min, batch_min)
        attr_max = max(attr_max, batch_max)
    
    print(f"Protected attribute range: [{attr_min}, {attr_max}]")
    
    for epoch in range(epochs):
        model.train()
        discriminator.train()
        total_task_loss = 0
        total_adv_loss = 0
        
        # KEY CHANGE: Use schedule for adversarial weight
        adv_weight = 0.1 + 0.4 * (epoch / epochs)  # Start at 0.1, max 0.5
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Extract and normalize protected attributes consistently
            protected_attrs = inputs[:, protected_attr_idx].unsqueeze(1)
            if attr_min == attr_max:  # Handle binary case
                protected_attrs = (protected_attrs > attr_min).float()
            else:
                protected_attrs = (protected_attrs - attr_min) / (attr_max - attr_min)
            
            # KEY CHANGE: Train discriminator less frequently
            if batch_idx % 3 == 0:  # Train discriminator every 3 batches
                disc_optimizer.zero_grad()
                
                # Get features without tracking gradients for main model
                with torch.no_grad():
                    _, features = model(inputs)
                    
                disc_output = discriminator(features)
                disc_loss = adv_criterion(disc_output, protected_attrs)
                disc_loss.backward()
                disc_optimizer.step()
            
            # Train model with task objective
            task_optimizer.zero_grad()
            outputs, features = model(inputs)
            
            # Task loss
            task_loss = task_criterion(outputs, labels)
            
            # KEY CHANGE: Separate gradient flows
            task_loss.backward(retain_graph=True)
            
            # Adversarial component with a separate backward pass
            disc_output = discriminator(features)
            adv_loss = adv_criterion(disc_output, protected_attrs)
            
            # Apply negative gradient for adversarial component
            (-adv_weight * adv_loss).backward()
            
            # Update model
            task_optimizer.step()
            
            total_task_loss += task_loss.item()
            total_adv_loss += adv_loss.item()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}], Task Loss: {total_task_loss/len(train_loader):.4f}, "
              f"Adv Loss: {total_adv_loss/len(train_loader):.4f}, "
              f"Adv Weight: {adv_weight:.4f}")
    
    return model

class ImprovedFairnessBCELoss(nn.Module):
    def __init__(self, fairness_type='demographic_parity', lambda_fair=0.5):
        """
        Improved fairness-aware loss function with more robust calculations
        """
        super(ImprovedFairnessBCELoss, self).__init__()
        self.base_criterion = nn.BCELoss(reduction='none')
        self.fairness_type = fairness_type
        self.lambda_fair = lambda_fair
        
    def forward(self, outputs, targets, protected_attrs):
        # Base BCE loss
        base_loss = self.base_criterion(outputs, targets)
        
        # Compute per-group metrics
        mask_protected = (protected_attrs > 0.5).float()
        mask_unprotected = 1 - mask_protected
        
        # Ensure we have samples from both groups
        has_protected = torch.sum(mask_protected) > 0
        has_unprotected = torch.sum(mask_unprotected) > 0
        
        # If either group is missing, return only the base loss
        if not (has_protected and has_unprotected):
            return torch.mean(base_loss)
        
        if self.fairness_type == 'demographic_parity':
            # Safe mean calculation
            pred_protected = safe_mean(outputs, mask_protected)
            pred_unprotected = safe_mean(outputs, mask_unprotected)
            fairness_loss = torch.abs(pred_protected - pred_unprotected)
            
        elif self.fairness_type == 'equalized_odds':
            # Positive and negative ground truth masks
            pos_mask = (targets > 0.5).float()
            neg_mask = 1 - pos_mask
            
            # Group masks
            prot_pos = mask_protected * pos_mask
            prot_neg = mask_protected * neg_mask
            unprot_pos = mask_unprotected * pos_mask
            unprot_neg = mask_unprotected * neg_mask
            
            # Safe calculations for TPR and FPR
            tpr_prot = safe_mean(outputs, prot_pos)
            tpr_unprot = safe_mean(outputs, unprot_pos)
            fpr_prot = safe_mean(outputs, prot_neg)
            fpr_unprot = safe_mean(outputs, unprot_neg)
            
            # Combined fairness loss
            fairness_loss = torch.abs(tpr_prot - tpr_unprot) + torch.abs(fpr_prot - fpr_unprot)
        else:
            raise ValueError(f"Unknown fairness type: {self.fairness_type}")
        
        # Combine losses with dynamic weighting
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

# Improved training with fairness constraints
def train_with_improved_fairness(model, train_loader, test_loader, 
                               protected_attr_idx, device, 
                               fairness_type='demographic_parity',
                               initial_lambda=0.1, final_lambda=1.0,
                               epochs=10, learning_rate=0.001):
    """
    Improved training with fairness constraints and dynamic lambda adjustment
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Find min and max values of protected attribute for normalization
    attr_min, attr_max = find_protected_attr_range(train_loader, protected_attr_idx)
    print(f"Protected attribute range: [{attr_min}, {attr_max}]")
    
    # Set up early stopping
    best_dp_diff = float('inf')
    patience = 3
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Calculate current lambda using linear schedule
        lambda_fair = initial_lambda + (final_lambda - initial_lambda) * (epoch / epochs)
        criterion = ImprovedFairnessBCELoss(fairness_type=fairness_type, lambda_fair=lambda_fair)
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Extract protected attributes and normalize to [0,1]
            protected_attrs = inputs[:, protected_attr_idx].unsqueeze(1)
            protected_attrs = normalize_protected_attr(protected_attrs, attr_min, attr_max)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute loss with fairness constraints
            loss = criterion(outputs, labels, protected_attrs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss with Fairness: {total_loss/len(train_loader):.4f}, "
              f"Lambda: {lambda_fair:.4f}")
        
        # Evaluate fairness metrics and implement early stopping
        fairness_metrics = evaluate_fairness_metrics(model, test_loader, protected_attr_idx, attr_min, attr_max, device)
        dp_diff = fairness_metrics['dp_diff']
        
        print(f"  - Demographic Parity Difference: {dp_diff:.4f}")
        print(f"  - TPR Difference: {fairness_metrics['tpr_diff']:.4f}")
        print(f"  - FPR Difference: {fairness_metrics['fpr_diff']:.4f}")
        
        # Early stopping based on fairness metric
        if dp_diff < best_dp_diff:
            best_dp_diff = dp_diff
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"  - New best fairness metric: {dp_diff:.4f}")
        else:
            patience_counter += 1
            print(f"  - No improvement in fairness. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("Early stopping due to no improvement in fairness metrics")
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    # Return best model if early stopping occurred
    if best_model_state is not None and patience_counter >= patience:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_fairness_metrics(model, test_loader, protected_attr_idx, attr_min, attr_max, device):
    """Evaluate various fairness metrics on test set"""
    model.eval()
    
    # Initialize counters for each group
    protected_pos_preds = 0
    protected_pos_total = 0
    unprotected_pos_preds = 0
    unprotected_pos_total = 0
    
    # For equalized odds
    protected_tp = 0
    protected_pos_labels = 0
    unprotected_tp = 0
    unprotected_pos_labels = 0
    
    protected_fp = 0
    protected_neg_labels = 0
    unprotected_fp = 0
    unprotected_neg_labels = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Extract protected attributes and normalize
            protected_attrs = inputs[:, protected_attr_idx].unsqueeze(1)
            protected_attrs = normalize_protected_attr(protected_attrs, attr_min, attr_max)
            
            # Model prediction
            if isinstance(model, AdversarialFCNN) or isinstance(model, FairRepresentationFCNN):
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            
            # Binary predictions
            preds = (outputs >= 0.5).float()
            
            # Get masks for protected and unprotected groups
            protected_mask = (protected_attrs >= 0.5).view(-1)
            unprotected_mask = ~protected_mask
            
            # Count positive predictions for each group (for demographic parity)
            protected_pos_preds += torch.sum(preds[protected_mask]).item()
            protected_pos_total += protected_mask.sum().item()
            
            unprotected_pos_preds += torch.sum(preds[unprotected_mask]).item()
            unprotected_pos_total += unprotected_mask.sum().item()
            
            # For equalized odds (TPR, FPR)
            # True positives and positives in each group
            pos_labels = (labels >= 0.5).view(-1)
            protected_pos_labels += (protected_mask & pos_labels).sum().item()
            unprotected_pos_labels += (unprotected_mask & pos_labels).sum().item()
            
            protected_tp += ((preds.view(-1) == 1) & protected_mask & pos_labels).sum().item()
            unprotected_tp += ((preds.view(-1) == 1) & unprotected_mask & pos_labels).sum().item()
            
            # False positives and negatives in each group
            neg_labels = ~pos_labels
            protected_neg_labels += (protected_mask & neg_labels).sum().item()
            unprotected_neg_labels += (unprotected_mask & neg_labels).sum().item()
            
            protected_fp += ((preds.view(-1) == 1) & protected_mask & neg_labels).sum().item()
            unprotected_fp += ((preds.view(-1) == 1) & unprotected_mask & neg_labels).sum().item()
    
    # Calculate demographic parity (difference in positive prediction rates)
    if protected_pos_total > 0 and unprotected_pos_total > 0:
        protected_pos_rate = protected_pos_preds / protected_pos_total
        unprotected_pos_rate = unprotected_pos_preds / unprotected_pos_total
        dp_diff = abs(protected_pos_rate - unprotected_pos_rate)
    else:
        dp_diff = 0.0
    
    # Calculate TPR for each group
    if protected_pos_labels > 0 and unprotected_pos_labels > 0:
        protected_tpr = protected_tp / protected_pos_labels
        unprotected_tpr = unprotected_tp / unprotected_pos_labels
        tpr_diff = abs(protected_tpr - unprotected_tpr)
    else:
        tpr_diff = 0.0
    
    # Calculate FPR for each group
    if protected_neg_labels > 0 and unprotected_neg_labels > 0:
        protected_fpr = protected_fp / protected_neg_labels
        unprotected_fpr = unprotected_fp / unprotected_neg_labels
        fpr_diff = abs(protected_fpr - unprotected_fpr)
    else:
        fpr_diff = 0.0
    
    # Return metrics
    return {
        'dp_diff': dp_diff,
        'tpr_diff': tpr_diff,
        'fpr_diff': fpr_diff
    }

# Update debug function to check protected attribute values
def debug_protected_attributes(train_loader, protected_attr_idx):
    """
    Enhanced debug function to check protected attribute values in the dataset
    """
    samples = 0
    unique_values = set()
    min_val = float('inf')
    max_val = float('-inf')
    
    # Count distribution for binary attributes
    count_0 = 0
    count_1 = 0
    
    for inputs, _ in train_loader:
        # Extract protected attributes
        protected_attrs = inputs[:, protected_attr_idx].numpy()
        
        # Update stats
        samples += len(protected_attrs)
        unique_values.update(protected_attrs.flatten())
        min_val = min(min_val, protected_attrs.min())
        max_val = max(max_val, protected_attrs.max())
        
        # Count binary distribution
        if len(unique_values) <= 2:
            for val in protected_attrs:
                if val == min_val:
                    count_0 += 1
                elif val == max_val:
                    count_1 += 1
        
        # Only check a few batches
        if samples >= 500:
            break
    
    print("\nProtected Attribute Debug Info:")
    print(f"- Attribute index: {protected_attr_idx}")
    print(f"- Min value: {min_val}")
    print(f"- Max value: {max_val}")
    print(f"- Unique values: {sorted(unique_values)}")
    print(f"- Is binary: {len(unique_values) <= 2}")
    print(f"- Is in [0,1] range: {min_val >= 0 and max_val <= 1}")
    
    if len(unique_values) <= 2:
        print(f"- Group distribution: {count_0} samples with value {min_val}, {count_1} samples with value {max_val}")
        print(f"- Group ratio: {count_0/samples:.2f} vs {count_1/samples:.2f}")
    
    print("")
    
    # Return information about whether normalization is needed
    return {
        "is_binary": len(unique_values) <= 2,
        "needs_normalization": min_val < 0 or max_val > 1,
        "min_val": min_val,
        "max_val": max_val
    }

class AdultIncomeDataset(Dataset):
    def __init__(self, file_path):
        # Load dataset
        column_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                        "marital-status", "occupation", "relationship", "race", "sex",
                        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

        df = pd.read_csv(file_path, names=column_names, skiprows=1)

        # Drop missing values
        df = df.replace(" ?", pd.NA).dropna()

        # Encode categorical features
        categorical_features = ["workclass", "education", "marital-status", "occupation", 
                                "relationship", "race", "sex", "native-country"]

        for col in categorical_features:
            df[col] = LabelEncoder().fit_transform(df[col])

        # Convert income to binary label
        df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

        # Separate features and labels
        X = df.drop(columns=["income"])
        y = df["income"]

        # Normalize numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CompasDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        # filter down desired features
        cols_to_keep = ['sex', 'age', 'race', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
                'c_charge_degree', 'decile_score', 'score_text', 'two_year_recid']
        df = df[cols_to_keep]
        df = df.dropna()

        # Encode categorical features
        df['sex'] = LabelEncoder().fit_transform(df['sex'])
        df['score_text'] = LabelEncoder().fit_transform(df['score_text'])

        # Filter to include only white and African-American offenders
        df = df[(df['race'] == 'African-American') | (df['race'] == 'Caucasian')]
        df['race'] = LabelEncoder().fit_transform(df['race'])
        
        X = df.drop(columns=['two_year_recid'])
        y = df['two_year_recid']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, q=False):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.q = q 
        if q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) 
        if self.q:
            x = self.dequant(x)
        return x

# Teacher Model (16 neurons)
def get_principle_model(input_size):
    return SimpleFCNN(input_size, 128)

# Teacher Model (16 neurons)
def get_teacher_model(input_size):
    return SimpleFCNN(input_size, 16)

# Student Model (8 neurons)
def get_student_model_8(input_size):
    return SimpleFCNN(input_size, 8)

# Student Model (4 neurons)
def get_student_model_4(input_size):
    return SimpleFCNN(input_size, 4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to use to train", choices=['compas', 'adult'])
    parser.add_argument("--debiasing", type=str, default="none", 
                      choices=['none', 'adversarial', 'fairness_constraint'],
                      help="Debiasing method to use")
    parser.add_argument("--fairness_type", type=str, default="demographic_parity",
                      choices=['demographic_parity', 'equalized_odds'],
                      help="Type of fairness constraint to apply")
    parser.add_argument("--protected_attr", type=str, default="sex",
                      help="Protected attribute to use for debiasing")
    parser.add_argument("--lambda_fair", type=float, default=0.5,
                      help="Weight for fairness term in loss function")
    parser.add_argument("--adv_weight", type=float, default=0.8,
                      help="Weight for adversarial loss term")
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of epochs to train")
    args = parser.parse_args()

    # Add the project root to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    # Detect the best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Apple Silicon GPU (Metal Performance Shaders)
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU
    else:
        device = torch.device("cpu")   # Default to CPU

    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = args.epochs
    LEARNING_RATE = 0.001

    # Load dataset
    if args.dataset == 'adult':
        dataset = AdultIncomeDataset("../../data/raw/adult.csv")
    else:
        dataset = CompasDataset("../../data/raw/compas.csv")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Map protected attribute name to column index
    protected_attr_map = {
        'adult': {
            'sex': 9,  # Index of sex in Adult dataset features
            'race': 8   # Index of race in Adult dataset features
        },
        'compas': {
            'sex': 0,  # Index of sex in COMPAS dataset features
            'race': 2   # Index of race in COMPAS dataset features
        }
    }

    # Get protected attribute index
    protected_attr_idx = protected_attr_map[args.dataset][args.protected_attr]
    print(f"Using protected attribute: {args.protected_attr} (index: {protected_attr_idx})")

    # Debug protected attributes to understand distribution
    attr_info = debug_protected_attributes(train_loader, protected_attr_idx)
    print(f"Protected attribute requires normalization: {attr_info['needs_normalization']}")

    # Create directory for model saving
    os.makedirs("../../models/debiased", exist_ok=True)
    os.makedirs("../../models/baseline", exist_ok=True)

    # Model setup based on debiasing method
    if args.debiasing == 'adversarial':
        print("Using Improved Adversarial Debiasing")
    
        # Create model and discriminator
        input_size = dataset.X.shape[1]
        model = AdversarialFCNN(input_size=input_size, hidden_size=16).to(device)
        
        # The discriminator operates on features
        discriminator = DiscriminatorNetwork(input_size=16).to(device)
        
        # Use the new training function instead of train_adversarial_compatible
        model = train_adversarial(
            model, discriminator, train_loader, test_loader,
            protected_attr_idx=protected_attr_idx, device=device,
            epochs=EPOCHS, learning_rate=LEARNING_RATE
        )
        
        # Save models
        torch.save(model.get_compatible_state_dict(), 
                f"../../models/debiased/adv_fcnn_model_{args.dataset}_{args.protected_attr}.pth")
        
        print("Models saved successfully!")
        
    elif args.debiasing == 'fairness_constraint':
        print(f"Using Improved Fairness Constraint: {args.fairness_type}")
        model = SimpleFCNN(input_size=dataset.X.shape[1], hidden_size=16).to(device)
        
        # Train with improved fairness constraints
        model = train_with_improved_fairness(
            model, train_loader, test_loader,
            protected_attr_idx=protected_attr_idx, device=device,
            fairness_type=args.fairness_type, 
            initial_lambda=0.1, final_lambda=args.lambda_fair,
            epochs=EPOCHS, learning_rate=LEARNING_RATE
        )
        
        # Save model
        torch.save(model.state_dict(), 
                 f"../../models/debiased/fair_{args.fairness_type}_fcnn_model_{args.dataset}_{args.protected_attr}.pth")
        
    else:  # Default baseline training without debiasing
        print("Using standard training (no debiasing)")
        model = SimpleFCNN(input_size=dataset.X.shape[1], hidden_size=16).to(device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")
        
        # Save model
        torch.save(model.state_dict(), f"../../models/debiased/fcnn_model_{args.dataset}_income.pth")

    print("Model saved successfully!")

    # Final fairness evaluation for all models
    attr_min, attr_max = find_protected_attr_range(train_loader, protected_attr_idx)
    fairness_metrics = evaluate_fairness_metrics(model, test_loader, protected_attr_idx, attr_min, attr_max, device)
    
    print("\nFinal Fairness Evaluation:")
    print(f"- Demographic Parity Difference: {fairness_metrics['dp_diff']:.4f}")
    print(f"- TPR Difference: {fairness_metrics['tpr_diff']:.4f}")
    print(f"- FPR Difference: {fairness_metrics['fpr_diff']:.4f}")

if __name__ == "__main__":
    main()