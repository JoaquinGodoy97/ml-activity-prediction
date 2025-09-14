import onnx
model = onnx.load("onnx_model_quantized/model_quantized.onnx")
for output in model.graph.output:
    print(output.name)