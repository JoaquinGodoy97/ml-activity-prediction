from optimized_pipeline import ActivityPredictor
import pandas as pd
import os

# Local use
def main():
    """Main execution function"""
    predictor = ActivityPredictor(onnx_model_path='onnx_model_quantized', model_file='model_quantized.onnx')
    
    # Check if models exist, if not train them
    if not all(os.path.exists(f) for f in ["main_model.joblib", "slot_model.joblib", "slot_encoders.joblib"]):
        print("Training models...")
        predictor.train_all()
    else:
        print("Loading existing models...")
        predictor.load_models()
    
    # Example predictions
    activity_list = [
    {"name": "Catch a butterfly outdoors", "duration": 45}
    ]

    original_df = pd.read_csv("activity_dataset.csv")

    def save_fixed_progress(df, checked_task):
        return pd.concat([df, checked_task], ignore_index=False)

    print("\nMaking predictions...")

    original_df = pd.read_csv("activity_dataset.csv")

    results = []
    for example in activity_list:
        result = predictor.predict_complete_task(example["name"], example["duration"])
        results.append(result)
        # print()
        print(f"Task: {result['task_name']}")
        # print(f"  Duration: {result['duration']} min")
        # print(f"  Type: {result['task_type']}, Mental: {result['mental_load']}, Physical: {result['physical_load']}")
        # print(f"  Ideal Slot: {result['ideal_slot']}")
        # print()

        checked_task = pd.DataFrame([result])
        original_df = save_fixed_progress(original_df, checked_task)
    
    original_df.to_csv("activity_dataset.csv", index=False)

    return results

if __name__ == "__main__":
    main()