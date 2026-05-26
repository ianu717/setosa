import pandas as pd

from pathlib import Path
from catboost import CatBoostClassifier
from utils import load_json, BASE_DIR
from preprocessing import preprocess

class SeverityInferenceService:

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = CatBoostClassifier()
        self.model.load_model(BASE_DIR / 'model' / 'severity_classifier.cbm')
        self.model_features = load_json(Path(BASE_DIR / 'data' / 'selected_features.json'))

    def predict_from_context(self, casualty_df: pd.DataFrame, vehicle_df: pd.DataFrame,collision_df: pd.DataFrame) -> dict:
        processed_df = preprocess(casualty=casualty_df, vehicle=vehicle_df, collision=collision_df)
        return self.predict(processed_df)

    def predict(self,model_input: pd.DataFrame) -> dict:
        model_input = model_input[
            self.model_features
        ]
        probabilities = self.model.predict_proba(model_input)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        return {
            'severity_probability': float(probabilities[0]),
            'prediction': int(predictions[0]),
            'threshold': self.threshold
        }

    def set_threshold(self,threshold: float):
        self.threshold = threshold