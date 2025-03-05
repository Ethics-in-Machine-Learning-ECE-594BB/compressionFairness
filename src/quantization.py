import torch
import torch.quantization as quantization
import os 
from src.simpleFCNN import SimpleFCNN
from torch.ao.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver

class myObserver(MinMaxObserver):
    def __init__(self, bit_width=8, **kwargs):
        super().__init__(**kwargs)
        self.bit_width = bit_width
    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        qmin, qmax = 0, (2**self.bit_width) - 1
        return scale, zero_point.clamp(qmin, qmax)

# torch optimization for fusing quantized layers together, helps with inference speed
# dependent on model architecture needs changing for other models 
def fuse_modules(model): 
    quantization.fuse_modules(model, [['fc1', 'relu']],  inplace=True)

def compute_model_size_in_memory(model, bit_width):
    param_size = 0
    # iterate through params and buffers summing up each
    for param in model.parameters():
        param_size += param.nelement()* param.element_size()
    buffer_size = 0
    try:
        for buffer in model.buffer():
            buffer_size += buffer.nelement()*buffer.element_size()
    except:
        # do nothing
        buffer_size = 0
    return (param_size+buffer_size)/(1024*1024)  # Convert to MB

def static_quantization(quantized_model,test_loader, path, bit_width=8):
    quantized_model.qconfig = quantization.QConfig(activation=myObserver.with_args(bit_width=bit_width, ), weight=quantization.default_weight_observer)
    # quantized_model.qconfig = quantization.default_qconfig

    # fuse_modules(unquantized_model)
    quantization.prepare(quantized_model, inplace=True)
    # need to pass some test data through model to allow torch to calibrate quantization 
    with torch.no_grad():
        for input, _ in test_loader:
            quantized_model(input)
    quantization.convert(quantized_model, inplace=True)
    print("Post-Train Quantization Complete")
    torch.save(quantized_model.state_dict(), path)
    quantized_size = compute_model_size_in_memory(quantized_model.to('cpu'))
    print(f"Quantized size: {quantized_size}")
# https://gist.github.com/martinferianc/d6090fffb4c95efed6f1152d5fde079d 