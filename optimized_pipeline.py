"""
Optimized ML Pipeline for Activity Prediction
This script combines training and prediction in an efficient manner
"""

import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings('ignore')

class ActivityPredictor:
    def __init__(self, embedding_model_name='sentence-transformers/paraphrase-MiniLM-L3-v2', onnx_model_path='onnx_model'):
        """Initialize the predictor with a single embedding model"""

        # self.embed_model = SentenceTransformer(embedding_model_name)
        #Onnx to shrink model size
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embed_model = ORTModelForFeatureExtraction.from_pretrained(onnx_model_path)

        self.main_model = None
        self.slot_model = None
        self.encoders = {}
        self.is_trained = False
        
    def load_data(self, csv_path="activity_dataset.csv"):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def preprocess_data(self):
        """Preprocess data with optimizations"""
        print("Preprocessing data...")
        
        # Encode categorical features
        for col in ["task_type", "mental_load", "physical_load", "ideal_slot"]:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le
        
        # Generate embeddings in batch (much faster)
        print("Generating embeddings...")
        task_names = self.df['task_name'].tolist()
        # embeddings = self.embed_model.encode(task_names, show_progress_bar=True, batch_size=32)
        embeddings = self._generate_embeddings(task_names)
        self.df['task_embedding'] = pd.Series(list(embeddings))
        
        return self.df
    
    def _generate_embeddings(self, texts):
        """Generate embeddings using ONNX model"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.embed_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    def train_main_model(self):
        """Train the main model for task classification"""
        print("Training main model...")
        
        # Prepare features efficiently
        durations = self.df["duration"].values.reshape(-1, 1)
        embeddings = np.array(self.df["task_embedding"].tolist())
        X = np.hstack([embeddings, durations])
        
        y = self.df[["task_type", "mental_load", "physical_load"]]
        
        # Train with optimized parameters
        self.main_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        
        self.main_model.fit(X, y)
        
        # Save model
        dump(self.main_model, "main_model.joblib")
        print("Main model trained and saved")
        
    def train_slot_model(self):
        """Train the slot prediction model"""
        print("Training slot model...")
        
        # Prepare features
        X = np.hstack([
            np.array(self.df["task_embedding"].tolist()),
            self.df[["task_type", "duration", "mental_load", "physical_load"]].values
        ])
        
        y = self.df["ideal_slot"]
        
        # Train with optimized parameters
        self.slot_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        
        self.slot_model.fit(X, y)
        
        # Save models
        dump(self.slot_model, "slot_model.joblib")
        dump(self.encoders, "slot_encoders.joblib")
        print("Slot model trained and saved")
        
    def train_all(self, csv_path="activity_dataset.csv"):
        """Train both models"""
        self.load_data(csv_path)
        self.preprocess_data()
        self.train_main_model()
        self.train_slot_model()
        self.is_trained = True
        print("All models trained successfully!")
        
    def load_models(self):
        """Load pre-trained models"""
        if not os.path.exists("main_model.joblib"):
            raise FileNotFoundError("Main model not found. Please train first.")
        if not os.path.exists("slot_model.joblib"):
            raise FileNotFoundError("Slot model not found. Please train first.")
        if not os.path.exists("slot_encoders.joblib"):
            raise FileNotFoundError("Encoders not found. Please train first.")
            
        self.main_model = load("main_model.joblib")
        self.slot_model = load("slot_model.joblib")
        self.encoders = load("slot_encoders.joblib")
        self.is_trained = True
        print("Models loaded successfully!")
        
    def predict_task_features(self, task_name, duration):
        """Predict task_type, mental_load, physical_load"""
        if not self.is_trained:
            raise ValueError("Models not trained or loaded")
            
        # Embed task name
        # task_vec = self.embed_model.encode([task_name])
        task_vec = self._generate_embeddings([task_name])

        combined_input = np.concatenate([task_vec, [[duration]]], axis=1)
        
        # Predict
        prediction = self.main_model.predict(combined_input)[0]
        
        # Decode the predictions back to original string values
        decoded_task_type = self.encoders["task_type"].inverse_transform([prediction[0]])[0]
        decoded_mental_load = self.encoders["mental_load"].inverse_transform([prediction[1]])[0]
        decoded_physical_load = self.encoders["physical_load"].inverse_transform([prediction[2]])[0]
        
        return decoded_task_type, decoded_mental_load, decoded_physical_load
        
    def predict_ideal_slot(self, task_name, duration, task_type, mental_load, physical_load):
        """Predict ideal time slot"""
        if not self.is_trained:
            raise ValueError("Models not trained or loaded")
            
        # Encode categorical features
        encoded_type = self.encoders["task_type"].transform([task_type])[0]
        encoded_mental = self.encoders["mental_load"].transform([mental_load])[0]
        encoded_physical = self.encoders["physical_load"].transform([physical_load])[0]
        
        # Embed task name
        # task_vec = self.embed_model.encode([task_name])
        task_vec = self._generate_embeddings([task_name])
        
        # Combine features
        input_vector = np.concatenate([
            task_vec,
            [[encoded_type, duration, encoded_mental, encoded_physical]]
        ], axis=1)
        
        # Predict
        prediction = self.slot_model.predict(input_vector)[0]
        
        # Decode the prediction back to the original string value
        decoded_prediction = self.encoders["ideal_slot"].inverse_transform([prediction])[0]
        return decoded_prediction
        
    def predict_complete_task(self, task_name, duration):
        """Complete prediction pipeline"""
        # First predict task features
        task_type, mental_load, physical_load = self.predict_task_features(task_name, duration)
        
        # Then predict ideal slot
        ideal_slot = self.predict_ideal_slot(task_name, duration, task_type, mental_load, physical_load)
        
        return {
            "task_name": task_name,
            "duration": duration,
            "task_type": task_type,
            "mental_load": mental_load,
            "physical_load": physical_load,
            "ideal_slot": ideal_slot
        }


