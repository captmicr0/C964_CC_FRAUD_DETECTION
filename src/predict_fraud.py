import argparse
import pandas as pd
import joblib

class FraudPredictionCLI:
    def __init__(self, model_path):
        """Initialize with trained model"""
        self.model = joblib.load(model_path)
        self.required_features = [
            'amt', 'city_pop', 'age', 'trans_hour', 'trans_day',
            'lat', 'long', 'merch_lat', 'merch_long'
        ]

    def _validate_input(self, input_data):
        """Ensure all required features are present"""
        missing = set(self.required_features) - set(input_data.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def _preprocess_input(self, args):
        """Handle both single and batch input preprocessing"""
        # Handle batch CSV input
        if 'input_file' in args and args['input_file']:
            return self._preprocess_batch(args['input_file'])
        
        # Handle single transaction
        return self._preprocess_single(args)

    def _preprocess_single(self, transaction):
        """Process individual transaction"""
        data = {
            'amt': [transaction.get('amount')],
            'city_pop': [transaction.get('city_pop')],
            'age': [transaction.get('age')],
            'trans_hour': [transaction.get('hour')],
            'trans_day': [transaction.get('day')],
            'lat': [transaction.get('lat')],
            'long': [transaction.get('long')],
            'merch_lat': [transaction.get('merch_lat')],
            'merch_long': [transaction.get('merch_long')]
        }
        return pd.DataFrame(data)

    def _preprocess_batch(self, file_path):
        """Process CSV file with multiple transactions"""
        df = pd.read_csv(file_path, parse_dates=['trans_date_trans_time'])
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
        df['age'] = (pd.to_datetime('today') - df['dob']).dt.days // 365
        return df[self.required_features]

    def predict(self, args):
        """Unified prediction method for single/batch inputs"""
        # Convert namespace to dict if needed
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        
        # Preprocess input data
        input_data = self._preprocess_input(args)
        
        # Validate input data
        self._validate_input(input_data)
        
        # Predict fraud probabilities
        probabilities = self.model.predict_proba(input_data)[:, 1]
        
        # Return single or batch predictions
        return probabilities[0] if len(probabilities) == 1 else probabilities.tolist()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Credit Card Fraud Prediction System')
    
    # Single transaction arguments
    single_group = parser.add_argument_group('Single transaction')
    single_group.add_argument('--amount', type=float, help='Transaction amount')
    single_group.add_argument('--city-pop', type=int, help='City population')
    single_group.add_argument('--age', type=int, help='Customer age')
    single_group.add_argument('--hour', type=int, help='Transaction hour (0-23)')
    single_group.add_argument('--day', type=int, help='Transaction day (0=Monday, ..., 6=Sunday)')
    single_group.add_argument('--lat', type=float, help='Customer latitude')
    single_group.add_argument('--long', type=float, help='Customer longitude')
    single_group.add_argument('--merch-lat', type=float, help='Merchant latitude')
    single_group.add_argument('--merch-long', type=float, help='Merchant longitude')

    # Batch processing arguments
    batch_group = parser.add_argument_group('Batch processing')
    batch_group.add_argument('--input-file', type=str, help='Path to CSV file containing transactions')
    batch_group.add_argument('--output-file', type=str, help='Path to save prediction results')

    # Required arguments
    parser.add_argument('--model-path', required=True, help='Path to trained ML model')
    
    args = parser.parse_args()
    
    try:
        # Initialize FraudPredictionCLI
        predictor = FraudPredictionCLI(args.model_path)
        
        # Make predictions
        predictions = predictor.predict(args)
        
        # Handle output
        if isinstance(predictions, list):
            if args.output_file:
                pd.DataFrame({'fraud_probability': predictions}).to_csv(args.output_file, index=False)
                print(f"Saved batch predictions to {args.output_file}")
            else:
                print("Batch predictions:", predictions)
        else:
            print(f"Fraud Probability: {predictions * 100:.2f}%")

    except ValueError as e:
        print(f"Input Error: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"Prediction Failed: {str(e)}")
        exit(1)

