FROM python:3.12-slim
RUN apt-get update && apt-get install -y git-lfs && git lfs install
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt
COPY optimized_pipeline.py app.py .
COPY onnx_model_quantized ./onnx_model_quantized/
COPY main_model.joblib slot_model.joblib slot_encoders.joblib ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]