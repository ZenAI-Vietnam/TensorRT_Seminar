import torch 
import onnxruntime as ort
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

from tool import load_engine, HostDeviceMemory, cosine_similarity

img = torch.randn((1, 3, 640 ,640))

#TORCH
#___________________________________________________________________________________________!
model_torch = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt")
model_torch.eval()

image_torch = img.to("cuda")

start_time = time.time()
result_torch = model_torch(image_torch)
end_time = time.time()
print(f"Time taken for torch inference: {end_time - start_time} seconds")
print(result_torch.shape)
#___________________________________________________________________________________________!

#TENSORRT
#___________________________________________________________________________________________!


#infer with trt
# TensorRT Optimized with FP16
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
    
model_trt = load_engine("yolov5s.engine")

context = model_trt.create_execution_context()
stream = cuda.Stream()


inputs, outputs, bindings = [], [], []
batch_size = img.shape[0]
height = img.shape[2]
width = img.shape[3]

for binding in model_trt:
    print(binding)
    idx = model_trt.get_binding_index(binding)
    shape = context.get_binding_shape(idx)
    if -1 in shape:  
        shape = list(shape)
        shape[0] = batch_size  
        if model_trt.binding_is_input(binding):
            shape[2] = height
            shape[3] = width
        context.set_binding_shape(idx, shape)
        
    size = trt.volume(shape)
    dtype = trt.nptype(model_trt.get_binding_dtype(idx))
    
    shape = tuple(shape)
        
    host_mem = cuda.pagelocked_empty(shape, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    
    if model_trt.binding_is_input(binding):
        inputs.append(HostDeviceMemory(host_mem, device_mem))
    else:
        outputs.append(HostDeviceMemory(host_mem, device_mem))

if model_trt.has_implicit_batch_dimension:
    context.set_binding_shape(0, img.shape)

inputs[0].host = img.numpy()
output_trt = []

for inp in inputs:
    cuda.memcpy_htod_async(inp.device, inp.host, stream)

start_time = time.time()
context.execute_async(bindings=bindings, stream_handle=stream.handle)
end_time = time.time()
print(f"Time taken for trt inference: {end_time - start_time} seconds")

for out in outputs:
    cuda.memcpy_dtoh_async(out.host, out.device, stream)
    output_trt.append(out.host)

stream.synchronize()
print(output_trt[0].shape)
#___________________________________________________________________________________________!

#Compare cosine similarity between 2 tensors
print("Compare torch and trt")
print(cosine_similarity(result_torch, output_trt[0]))
#___________________________________________________________________________________________!
