import torch
import os 
import copy
import torch.quantization as quantization
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from tqdm import tqdm 
from src.simpleFCNN import SimpleFCNN

INT16_CFG = [{
    "quant_cfg": {
    "*weight_quantizer": {"num_bits": 16, "axis": 0},
    "*input_quantizer": {"num_bits": 16, "axis": None},
    "*lm_head*": {"enable": False},
    "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
    "*router*": {"enable": False},  # Skip the MOE router
    "default": {"enable": False},
    },
    "algorithm": "max",
}, 'INT16']
INT8_CFG = [{
    "quant_cfg": {
    "*weight_quantizer": {"num_bits": 16, "axis": 0},
    "*input_quantizer": {"num_bits": 16, "axis": None},
    "*lm_head*": {"enable": False},
    "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
    "*router*": {"enable": False},  # Skip the MOE router
    "default": {"enable": False},
    },
    "algorithm": "max",
}, 'INT8']
INT4_CFG = [{
    "quant_cfg": {
    "*weight_quantizer": {"num_bits": 4, "axis": 0},
    "*input_quantizer": {"num_bits": 4, "axis": None},
    "*lm_head*": {"enable": False},
    "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
    "*router*": {"enable": False},  # Skip the MOE router
    "default": {"enable": False},
    },
    "algorithm": "max",
}, 'INT4']
FP8_DEFAULT_CFG = [{
    "quant_cfg": {
    "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
    "*input_quantizer": {"num_bits": (4, 3), "axis": None},
    "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
    "*router*": {"enable": False},  # Skip the MOE router
    "default": {"enable": False},
    },
    "algorithm": "max",
}, 'FP8']

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

def static_quantization(model_to_quant,forward_loop, path):
#     quantized_model.eval()
#     quantized_model.qconfig = quantization.QConfig(activation=myObserver.with_args(bit_width=bit_width), weight=quantization.default_weight_observer)
#     # quantized_model.qconfig = quantization.default_qconfig

#     # fuse_modules(unquantized_model)
#     quantization.prepare(quantized_model, inplace=True)
#     # need to pass some test data through model to allow torch to calibrate quantization 
#     with torch.no_grad():
#         for input, _ in tqdm(test_loader):
#             quantized_model(input)
#     quantization.convert(quantized_model, inplace=True)
#     print("Post-Train Quantization Complete")
#     torch.save(quantized_model.state_dict(), path)
#     quantized_size = compute_model_size_in_memory(quantized_model.to('cpu'), bit_width)
#     print(f"Quantized size: {quantized_size}")
    print("Quantizing model, note this may take a few minutes")
    quant_methods = [INT16_CFG, INT8_CFG, INT4_CFG, FP8_DEFAULT_CFG]
    for quant_method in quant_methods:
        quantized_model = copy.deepcopy(model_to_quant)
        print(f"Method: {quant_method[1]}")
        quantized_model = mtq.quantize(quantized_model, quant_method[0], forward_loop)
        quantized_model.to(torch.device('cpu')) # should be put on cpu otherwise errors will arise during load
        mto.save(quantized_model, f"{path}{quant_method[1]}.pth")

def load_quantized(base_model, path_to_quant):
    base_model.to(torch.device('cpu'))
    quant_model = mto.restore(base_model, path_to_quant)
    return quant_model