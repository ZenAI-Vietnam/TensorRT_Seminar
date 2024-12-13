import tensorrt as trt
import torch

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)

config = builder.create_builder_config()

#set memory pool limit
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

#set fp16 mode
config.set_flag(trt.BuilderFlag.FP16)

# #Dynamic Batch Size
# flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file("yolov5s.onnx")
for error in range(parser.num_errors):
    print(parser.get_error(error))

inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]

#Show the inputs and outputs
print("Inputs:")
for input in inputs:
    print(input.name, input.dtype, input.shape)
print("Outputs:")
for output in outputs:
    print(output.name, output.dtype, output.shape)

# dynamic = False
# if dynamic:
#     profile = builder.create_optimization_profile()
#     for inp in inputs:
#         profile.set_shape(inp.name, (1, 3, 640, 640), (8, 3, 640, 640), (32, 3, 640, 640))
#     config.add_optimization_profile(profile)

build = builder.build_serialized_network(network, config)  

with open("yolov5s.engine", "wb") as t:
    t.write(build)

print("Engine saved to yolov5s_dynamic.engine")





