"""
Simple script to train all models
Run this first before using prediction scripts
"""

from optimized_pipeline import ActivityPredictor

def main():
    print("Starting model training...")
    predictor = ActivityPredictor()
    predictor.train_all()
    print("Training completed! You can now run prediction scripts.")

if __name__ == "__main__":
    main()
