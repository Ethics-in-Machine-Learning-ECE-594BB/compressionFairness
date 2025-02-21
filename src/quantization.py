import torch
import torch.quantization as quantization
import os 
from src.simpleFCNN import SimpleFCNN
# load quantized model 
def q_load_model(quantized_model, model):
    # state_dict = model.state_dict()
    # model = model.to('cpu')
    quantized_model.load_state_dict(model.state_dict())

# torch optimization for fusing quantized layers together, helps with inference speed
# dependent on model architecture needs changing for other models 
def fuse_modules(model): 
    quantization.fuse_modules(model, [['fc1', 'relu']],  inplace=True)

def compute_model_size_in_memory(model):
    total_size = sum(p.element_size() * p.numel() for p in model.parameters())
    return total_size  # Convert to MB

def static_quantization(unquantized_model, test_loader, path):
    quantized_model = SimpleFCNN(14, 16, q=True)
    quantized_model.qconfig = quantization.default_qconfig

  
    q_load_model(quantized_model, unquantized_model)
    # fuse_modules(unquantized_model)
    
    quantization.prepare(quantized_model, inplace=True)
    # need to pass some test data through model to allow torch to calibrate quantization 
    with torch.no_grad():
        for input, _ in test_loader:
            quantized_model(input)
    quantization.convert(quantized_model, inplace=True)
    print("Post-Train Quantization Complete")
    torch.save(quantized_model.state_dict(), path)
    orig_size = compute_model_size_in_memory(unquantized_model)
    quantized_size = compute_model_size_in_memory(quantized_model.to('cpu'))
    print(f"Original Model Size in RAM: {orig_size:.2f} MB")
    print(f"Quantized Model Size in RAM: {quantized_size:.2f} MB")
# https://gist.github.com/martinferianc/d6090fffb4c95efed6f1152d5fde079d 