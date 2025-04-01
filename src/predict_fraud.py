import argparse
import pandas as pd
import joblib
from typing import Union, List, Dict

class FraudPredictionCLI:
    def __init__(self, model_path):
        # Load the trained model
        self.model = joblib.load(model_path)
    
    def _preprocess_input(self, args):
        """Preprocess user input into a format suitable for the ML model."""
        # Create a DataFrame from user input
        data = {
            'amt': [args.amount],
            'city_pop': [args.city_pop],
            'age': [args.age],
            'trans_hour': [args.hour],
            'trans_day': [args.day],
            'lat': [args.lat],
            'long': [args.long],
            'merch_lat': [args.merch_lat],
            'merch_long': [args.merch_long]
        }
        
        df = pd.DataFrame(data)
        
        return df
    
    def predict_single(self, args):
        """Predict fraud probability using the trained model."""
        # Pre-process input data
        input_data = self._preprocess_input(args)
        # Predict probabilities
        probabilities = self.model.predict_proba(input_data)[:, 1]  # Get fraud probability (class 1)
        return probabilities[0]

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict Credit Card Fraud')
    
    # Define command-line arguments for transaction details
    parser.add_argument('--amount', type=float, required=True, help='Transaction amount')
    parser.add_argument('--city-pop', type=int, required=True, help='City population')
    parser.add_argument('--age', type=int, required=True, help='Customer age')
    parser.add_argument('--hour', type=int, required=True, help='Transaction hour (0-23)')
    parser.add_argument('--day', type=int, required=True, help='Transaction day (0=Monday, ..., 6=Sunday)')
    parser.add_argument('--lat', type=float, required=True, help='Customer latitude')
    parser.add_argument('--long', type=float, required=True, help='Customer longitude')
    parser.add_argument('--merch-lat', type=float, required=True, help='Merchant latitude')
    parser.add_argument('--merch-long', type=float, required=True, help='Merchant longitude')
    
    # Path to the trained model
    parser.add_argument('--model-path', default='fraud_model.pkl', help='Path to the trained ML model')
    
    args = parser.parse_args()
    
    # Initialize prediction CLI
    predictor = FraudPredictionCLI(args.model_path)
    
    # Predict fraud probability
    fraud_prob = predictor.predict_single(args)
    
    print(f"Fraud Probability: {fraud_prob * 100:.2f}%")
