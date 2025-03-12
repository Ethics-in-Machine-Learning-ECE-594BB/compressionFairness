import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import argparse
import os
import copy
import sys
from pathlib import Path
import importlib.util

# Define model architectures from train_fairness.py
class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleFCNN, self).__init__()
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
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        features = self.relu(x)
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

def calculate_sparsity(model):
    """Calculate the sparsity (percentage of zeros) in the model parameters"""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only count weights, not biases
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
    
    return zero_params / total_params if total_params > 0 else 0

def prune_model(model, amount, model_type='standard'):
    """
    Prune a model by setting weights to zero
    
    Args:
        model: PyTorch model to prune
        amount (float): Amount to prune (between 0 and 1)
        model_type (str): Type of model ('standard', 'adversarial', 'fairness')
        
    Returns:
        model: Pruned model
    """
    # Make a copy of the model to avoid modifying the original
    pruned_model = copy.deepcopy(model)
    
    # Apply L1 unstructured pruning to the weights of all Linear layers
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    # Calculate sparsity after pruning
    sparsity = calculate_sparsity(pruned_model)
    print(f"Model sparsity after pruning: {sparsity:.2%}")
    
    return pruned_model

def make_pruning_permanent(model):
    """Make pruning permanent by removing the reparameterization"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    return model

def import_from_file(module_name, file_path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_data(data_path, train_fairness_path=None, batch_size=32):
    """
    Load the Adult Income dataset for fine-tuning
    Try to import the dataset class from train_fairness.py if available
    """
    try:
        # First try to import directly if train_fairness.py is in the path
        try:
            from train_fairness import AdultIncomeDataset
            print("Imported AdultIncomeDataset from train_fairness module")
        except ImportError:
            # If that fails, try to import from file path
            if train_fairness_path:
                train_fairness = import_from_file("train_fairness", train_fairness_path)
                AdultIncomeDataset = train_fairness.AdultIncomeDataset
                print(f"Imported AdultIncomeDataset from {train_fairness_path}")
            else:
                # Look for train_fairness.py in current and parent directories
                script_dir = Path(__file__).parent
                candidates = [
                    script_dir / "train_fairness.py",
                    script_dir.parent / "train_fairness.py"
                ]
                
                for path in candidates:
                    if path.exists():
                        train_fairness = import_from_file("train_fairness", str(path))
                        AdultIncomeDataset = train_fairness.AdultIncomeDataset
                        print(f"Imported AdultIncomeDataset from {path}")
                        break
                else:
                    raise ImportError("Could not find train_fairness.py")
        
        # Load dataset
        dataset = AdultIncomeDataset(data_path)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )
        
        print(f"Loaded dataset with {len(dataset)} samples")
        print(f"Train: {train_size} samples, Validation: {val_size} samples")
        
        return train_loader, val_loader
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: For fine-tuning to work, the train_fairness.py script must be available")
        return None, None

def fine_tune_model(model, train_loader, val_loader=None, epochs=5, learning_rate=0.001, device='cpu', model_type='standard'):
    """
    Fine-tune a pruned model
    
    Args:
        model: PyTorch model to fine-tune
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on
        model_type (str): Type of model ('standard', 'adversarial', 'fairness')
        
    Returns:
        model: Fine-tuned model
    """
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass depends on model type
            if model_type == 'adversarial':
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass depends on model type
                    if model_type == 'adversarial':
                        outputs, _ = model(inputs)
                    else:
                        outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    predicted = (outputs >= 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            
            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best model (val loss: {best_val_loss:.4f})")
    
    # Load best model if validation was used
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Prune fairness-aware models')
    parser.add_argument('--models_dir', type=str, default='../../models/debiased',
                        help='Directory containing the models')
    parser.add_argument('--output_dir', type=str, default='../../models/pruned',
                        help='Directory to save pruned models')
    parser.add_argument('--input_size', type=int, default=14,
                        help='Input size for the models (default: 14 for Adult dataset)')
    parser.add_argument('--hidden_size', type=int, default=16,
                        help='Hidden layer size (default: 16)')
    parser.add_argument('--prune_levels', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7',
                        help='Comma-separated list of pruning levels (default: 0.1,0.2,0.3,0.4,0.5,0.6,0.7)')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune the pruned models')
    parser.add_argument('--data_path', type=str, default='../../data/raw/adult.csv',
                        help='Path to dataset for fine-tuning')
    parser.add_argument('--train_fairness_path', type=str, default=None,
                        help='Path to train_fairness.py script (optional)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for fine-tuning')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated list of model names to process, or "all" for all models')
    args = parser.parse_args()
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse pruning levels
    prune_levels = [float(x) for x in args.prune_levels.split(',')]
    prune_levels.sort()  # Ensure they're in ascending order
    
    # Model configurations - map name patterns to types
    model_configs = {
        'adv_fcnn_model_adult_sex.pth': {
            'type': 'adversarial',
            'class': AdversarialFCNN,
            'description': 'Adversarial Debiased Model'
        },
        'fair_demographic_parity_fcnn_model_adult_sex.pth': {
            'type': 'fairness',
            'class': SimpleFCNN,
            'description': 'Fairness Constraint Model'
        },
        'fcnn_model_adult_income.pth': {
            'type': 'standard',
            'class': SimpleFCNN,
            'description': 'Standard Model'
        }
    }
    
    # Filter models if specific ones were requested
    if args.models != 'all':
        requested_models = args.models.split(',')
        model_configs = {k: v for k, v in model_configs.items() if k in requested_models}
    
    # Setup data for fine-tuning if needed
    train_loader, val_loader = None, None
    if args.fine_tune:
        train_loader, val_loader = load_data(
            args.data_path, 
            args.train_fairness_path, 
            args.batch_size
        )
        if train_loader is None:
            print("Warning: Could not load dataset for fine-tuning")
            print("Proceeding without fine-tuning")
            args.fine_tune = False
    
    # Process each model
    for model_name, config in model_configs.items():
        model_path = os.path.join(args.models_dir, model_name)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found, skipping")
            continue
        
        print(f"\nProcessing model: {model_name} ({config['description']})")
        
        # Load model
        try:
            model_class = config['class']
            model_type = config['type']
            
            model = model_class(args.input_size, args.hidden_size).to(device)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
            print(f"Successfully loaded model: {model_name}")
            
            # Calculate initial sparsity
            initial_sparsity = calculate_sparsity(model)
            print(f"Initial model sparsity: {initial_sparsity:.2%}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        # Process each pruning level
        for amount in prune_levels:
            print(f"\nPruning {model_name} by {amount*100:.1f}%")
            
            # Create pruned model
            pruned_model = prune_model(model, amount, model_type)
            
            # Fine-tune if requested
            if args.fine_tune and train_loader is not None:
                print(f"Fine-tuning pruned model...")
                pruned_model = fine_tune_model(
                    pruned_model, train_loader, val_loader,
                    epochs=args.epochs, device=device, 
                    model_type=model_type
                )
            
            # Make pruning permanent
            pruned_model = make_pruning_permanent(pruned_model)
            
            # Calculate final sparsity
            final_sparsity = calculate_sparsity(pruned_model)
            print(f"Final model sparsity after pruning: {final_sparsity:.2%}")
            
            # Save pruned model
            output_name = f"{os.path.splitext(model_name)[0]}_pruned_{int(amount*100)}pct.pth"
            if args.fine_tune:
                output_name = f"{os.path.splitext(model_name)[0]}_pruned_{int(amount*100)}pct_finetuned.pth"
            
            output_path = os.path.join(args.output_dir, output_name)
            
            # For adversarial models, save compatible state dict
            if model_type == 'adversarial':
                torch.save(pruned_model.get_compatible_state_dict(), output_path)
            else:
                torch.save(pruned_model.state_dict(), output_path)
            
            print(f"Saved pruned model to: {output_path}")
    
    print("\nAll models processed successfully!")

if __name__ == "__main__":
    main()