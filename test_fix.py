"""
Test script to verify the fix works
"""

import os
import sys

def test_training():
    """Test if training works without errors"""
    print("Testing training...")
    try:
        # Import and run training
        from optimized_pipeline import ActivityPredictor
        
        predictor = ActivityPredictor()
        predictor.train_all()
        print("✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_prediction():
    """Test if prediction works without errors"""
    print("Testing prediction...")
    try:
        from optimized_pipeline import ActivityPredictor
        
        predictor = ActivityPredictor()
        predictor.load_models()
        
        # Test prediction
        result = predictor.predict_complete_task("Write a report", 30)
        print(f"✅ Prediction successful: {result}")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def main():
    print("Running tests...")
    
    # Check if models exist
    model_files = ["main_model.joblib", "slot_model.joblib", "slot_encoders.joblib"]
    models_exist = all(os.path.exists(f) for f in model_files)
    
    if not models_exist:
        print("Models don't exist, training first...")
        if not test_training():
            return
    else:
        print("Models exist, testing prediction...")
    
    # Test prediction
    test_prediction()

if __name__ == "__main__":
    main()
