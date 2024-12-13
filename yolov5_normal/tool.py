import tensorrt as trt
import numpy as np
import torch
import pycuda.driver as cuda
import os
import pycuda.autoinit 
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

class CalibrationDataLoader:
    def __init__(self, batch_size=1, num_batches=10):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_batch = 0

    def get_batch(self):
        if self.current_batch >= self.num_batches:
            return None

        # Generate calibration data (replace this with your real dataset)
        data = np.random.randn(self.batch_size, 3, 640, 640).astype(np.float32)
        self.current_batch += 1
        return [data]

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.dataloader = dataloader

        # Allocate memory for calibration data
        self.d_input = cuda.mem_alloc(dataloader.batch_size * 3 * 640 * 640 * np.dtype(np.float32).itemsize)
        self.stream = cuda.Stream()

    def get_batch_size(self):
        return self.dataloader.batch_size

    def get_batch(self, names):
        batch = self.dataloader.get_batch()
        if batch is None:
            return None

        cuda.memcpy_htod_async(self.d_input, batch[0], self.stream)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists('calibration.cache'):
            with open('calibration.cache', 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open('calibration.cache', 'wb') as f:
            f.write(cache)
