import onnx
model = onnx.load("onnx_model_quantized/model_quantized.onnx")
for output in model.graph.output:
    print(output.name)

# from sentence_transformers import SentenceTransformer
# from optimum.onnxruntime import ORTModelForFeatureExtraction
# model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
# model.save_pretrained("onnx_model_quantized")
# ORTModelForFeatureExtraction.from_pretrained(
#     "onnx_model_quantized",
#     export=True,
#     file_name="model_quantized.onnx"
# ).save_pretrained("onnx_model_quantized")