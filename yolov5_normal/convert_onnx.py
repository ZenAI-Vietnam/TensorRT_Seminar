import torch

# Load YOLOv5 model from ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the model to evaluation mode
model.eval()

# Create dummy input for export (batch size = 1, 3 channels, 640x640 resolution)
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX format
torch.onnx.export(
    model,                # Model to export
    dummy_input,          # Dummy input tensor
    "yolov5s.onnx",         # File path to save the ONNX model
    input_names=["input"],  # Name of the input tensor
    output_names=["output"],  # Name of the output tensor
    opset_version=11,     # ONNX opset version
)

print("Model successfully exported to ONNX with dynamic batch size.")


# dynamic_axes={
#     #"input": {0: "batch_size"},  # Dynamic batch size for input
#     #"input": {0: "batch_size", 2: "height", 3: "width"},
#     "output": {0: "batch_size"}  # Dynamic batch size for output
# }