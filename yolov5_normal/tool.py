import tensorrt as trt
import numpy as np
import torch

logger = trt.Logger(trt.Logger.ERROR)

class HostDeviceMemory:
    def __init__(self, host, device):
        self.host = host
        self.device = device

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    

def cosine_similarity(tensor1, tensor2):
    if isinstance(tensor1, np.ndarray):
        tensor1 = torch.from_numpy(tensor1)
    if isinstance(tensor2, np.ndarray):
        tensor2 = torch.from_numpy(tensor2)

    tensor1 = tensor1.cpu()
    tensor2 = tensor2.cpu()
    return torch.mean(torch.cosine_similarity(tensor1, tensor2, dim=0))