from fastapi import FastAPI
from optimized_pipeline import ActivityPredictor
from pydantic import BaseModel
from predict_activities import main

app = FastAPI()

class TaskInput(BaseModel):
    task_list: list

@app.post("/predict_activities")
async def predict(task_input: TaskInput):
    result = main(task_input.task_list)
    return result