from fastapi import FastAPI
from optimized_pipeline import ActivityPredictor
from pydantic import BaseModel
from typing import List

app = FastAPI()
predictor = ActivityPredictor(onnx_model_path='onnx_model_quantized', model_file='model_quantized.onnx')
predictor.load_models()

class Task(BaseModel):
    name: str
    duration: float

class TaskInput(BaseModel):
    task_list: List[Task]

@app.post("/predict_activities")
async def predict(task_input: TaskInput):
    results = []
    for task in task_input.task_list:
        result = predictor.predict_complete_task(task.name, task.duration)
        results.append(result)
    return {"predictions": results}