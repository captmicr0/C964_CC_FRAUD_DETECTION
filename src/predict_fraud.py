import logging
from datetime import datetime
from tqdm import tqdm

import os
import argparse
import pandas as pd
from sqlalchemy import create_engine
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# Log file directory from ENV
log_dir = os.environ.get("LOG_DIR", os.path.join(os.getcwd(), "../logs"))

# Ensure the log directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Get the current date and time when the program starts
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Configure logging with a dynamic filename
log_filename = f"FraudPredictionCLI_{start_time}.log"

# Construct the full log file path
log_path = os.path.join(log_dir, log_filename)

# Create a logger
logger = logging.getLogger("FraudPredictionCLI")
logger.setLevel(logging.DEBUG)  # Set the level to DEBUG to capture all messages

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s", datefmt="%H:%M:%S")

# Create a file handler
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)  # Log all levels to the file
file_handler.setFormatter(formatter)

# Create a stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Log all levels to the console
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class FraudPredictionCLI:
    def __init__(self, model_path):
        # Load both model and preprocessing artifacts
        pbar = tqdm(total=4, desc="Loading Model Artifacts", unit="step")
        self.model = joblib.load(os.path.join(model_path, 'unbalanced_model.pkl'))
        pbar.update(1)
        self.model_smote = joblib.load(os.path.join(model_path, 'balanced_model.pkl'))
        pbar.update(1)
        self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
        pbar.update(1)
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        pbar.update(1)
        
        # Fit order
        self.feature_set = [
            'merchant', 'category', 'amt', 'gender', 'street', 'city', 'state',
            'zip', 'lat', 'long', 'city_pop', 'job', 'dob', 'merch_lat',
            'merch_long', 'cc_bin', 'hour', 'day', 'month'
        ]

    def feature_engineering(self, df_test):
        """Feature engineering input data to useable data"""
        # Initialize progress bar with a placeholder total
        pbar = tqdm(total=9, desc="Feature Engineering", unit="step")
        
        # Drop columns that are not important
        # keep: trans_date_trans_time, cc_num,merchant, category, amt, gender, street, city, state, zip, lat, long, city_pop, job, dob, merch_lat, merch_long, is_fraud
        columns_to_drop = ['idx', 'trans_num', 'unix_time', 'first', 'last']
        df_test.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
        pbar.update(1)

        # Convert cc_num to cc_bin (first 6 digits of cc_num) and drop the column
        if 'cc_num' in df_test.columns:
            df_test['cc_bin'] = df_test['cc_num'].astype(str).str[:6]  # Extract first 6 digits
            df_test.drop('cc_num', axis=1, inplace=True)  # Drop original cc_num column
        pbar.update(1)

        # Choose features and target for the training and test data
        X_test = df_test.drop('is_fraud', axis=1, errors='ignore')  # features
        if 'is_fraud' in X_test.columns:
            y_test = df_test['is_fraud']  # target, optional
        pbar.update(1)

        # Reorder the DataFrame
        X_test = X_test.reindex(columns=self.feature_set, fill_value=None)
        pbar.update(1)

        # Recognize numerical and categorical features in the training data
        numerical_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
        pbar.update(1)

        # Extract features from transaction date and time, then remove original column
        if 'trans_date_trans_time' in X_test.columns:
            X_test['trans_date_trans_time'] = pd.to_datetime(X_test['trans_date_trans_time'])
            X_test['hour'] = X_test['trans_date_trans_time'].dt.hour
            X_test['day'] = X_test['trans_date_trans_time'].dt.day
            X_test['month'] = X_test['trans_date_trans_time'].dt.month
            X_test.drop('trans_date_trans_time', axis=1, inplace=True, errors='ignore')
        pbar.update(1)

        # Update after processing
        categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
        pbar.update(1)

        # Dynamically update total once categorical_cols length is known
        pbar.total += len(categorical_cols)  # Add the number of categorical columns to the total
        pbar.refresh()  # Refresh the progress bar to reflect the new total

        # Convert categories into usable data
        for col in categorical_cols:
            self.label_encoder.fit(X_test[col])
            # transform data
            X_test[col] = self.label_encoder.transform(X_test[col])
            pbar.update(1)
        pbar.update(1)
        
        # Scale features
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])  # for test data don't use fit_transform
        pbar.update(1)
        pbar.close()

        logger.info("Feature set of testing data:")
        logger.info(X_test.columns)

        # Return processed data
        if 'is_fraud' in X_test.columns:
            return X_test, y_test

        return X_test, None
    
    def predict_single(self, args):
        """Make a prediction using the trained model."""
        # Convert a single transaction into a DataFrame
        single_df = pd.DataFrame([{
            'merchant': args.merchant,
            'category': args.category,
            'amt': args.amount,
            'gender': args.gender,
            'street': args.street,
            'city': args.city,
            'state': args.state,
            'zip': args.zip,
            'lat': args.lat,
            'long': args.long,
            'city_pop': args.city_pop,
            'job': args.job,
            'dob': args.dob,
            'merch_lat': args.merch_lat,
            'merch_long': args.merch_long,
            'cc_bin': args.cc_bin,
            'hour': args.hour,
            'day': args.day,
            'month': args.month
        }])

        # Feature engineering on input data
        X_test, y_test = self.feature_engineering(single_df)
        
        # Make a prediction using the unbalanced model
        logger.info("Prediction based on unbalanced model:")
        prediction = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        logger.info(f"\tPrediction: {'Fraudulent' if int(prediction[0]) == 1 else 'Non-Fraudulent'}")
        logger.info(f"\tProbability of Fraud: {probabilities[0]:.2f}")

        # Make a prediction using the balanced model
        logger.info("Prediction based on balanced model:")
        prediction = self.model_smote.predict(X_test)
        probabilities = self.model_smote.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        logger.info(f"\tPrediction: {'Fraudulent' if int(prediction[0]) == 1 else 'Non-Fraudulent'}")
        logger.info(f"\tProbability of Fraud: {probabilities[0]:.2f}")

        # Return results also
        return {'prediction': 'fraudulent' if int(prediction[0]) == 1 else 'non-fraudulent',
                'probability': probabilities[0]}
    
    @staticmethod
    def visualize_data(df, prediction_visuals_path):
        """Generate visualizations for fraud analysis."""
        
        # 1. Distribution of Transaction Amounts (Fraud vs Non-Fraud)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='amt', hue='is_fraud', kde=True, bins=50)
        plt.title('Distribution of Transaction Amounts (Fraud vs Non-Fraud)')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')
        plt.legend(title='Fraudulent Transaction', labels=['Non-Fraudulent', 'Fraudulent'])
        plt.savefig(os.path.join(prediction_visuals_path, 'amount_distribution.png'))
        plt.close()

        # 2. Heatmap of Correlation Between Features
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Features')
        plt.savefig(os.path.join(prediction_visuals_path, 'feature_correlation.png'))
        plt.close()

        # 3. Geographic Distribution of Fraudulent Transactions
        plt.figure(figsize=(10, 6))
        fraud_df = df[df['is_fraud'] == 1]
        plt.scatter(fraud_df['long'], fraud_df['lat'], alpha=0.5, c='red')
        plt.title('Geographic Distribution of Fraudulent Transactions')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(os.path.join(prediction_visuals_path, 'geographic_distribution.png'))
        plt.close()

# HTTP Server for REST API
class FraudPredictionAPI(BaseHTTPRequestHandler):
    def __init__(self, *args, predictor=None, **kwargs):
        self.predictor = predictor
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                # Parse JSON input
                data = json.loads(post_data)
                
                # Validate required fields
                required_fields = [
                    "amount", "hour", "day", "month", "cc_bin", "street", "city", 
                    "state", "zip", "city_pop", "dob", "gender", "job", 
                    "lat", "long", "merchant", "merch_lat", "merch_long", 
                    "category"
                ]
                if not all(field in data for field in required_fields):
                    self.send_response(400)
                    self.end_headers()
                    response = {"error": f"Missing required fields: {required_fields}"}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                    return
                
                # Convert input data to argparse.Namespace for compatibility with predict_single
                input_args = argparse.Namespace(**data)
                
                # Make prediction
                result = self.predictor.predict_single(input_args)
                
                # Send response
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            
            except Exception as e:
                # Handle errors gracefully
                self.send_response(500)
                self.end_headers()
                response = {"error": str(e)}
                self.wfile.write(json.dumps(response).encode('utf-8'))

def run_api_server(predictor, host='0.0.0.0', port=8000):
    def handler(*args, **kwargs):
        FraudPredictionAPI(*args, predictor=predictor, **kwargs)

    server = HTTPServer((host, port), handler)
    print(f"Starting API server at http://{host}:{port}")
    server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Prediction System')
    
    # Updated arguments matching new features
    # idx, trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud
    # 0, 2019-01-01 00:00:18, 2703186189652095, "fraud_Rippin,  Kub and Mann", misc_net, 4.97, Jennifer, Banks, F, 561 Perry Cove, Moravian Falls, NC, 28654, 36.0788, -81.1781, 3495, "Psychologist,  counselling", 1988-03-09, 0b242abb623afc578575680df30655b9, 1325376018, 36.011293, -82.048315, 0
    
    parser.add_argument('--api-server', action='store_true',
                        help="Start REST API server instead of CLI mode")
    parser.add_argument('--host', default='0.0.0.0',
                        help="API server host (default: 0.0.0.0)")
    parser.add_argument('--port', type=int, default=8000,
                        help="API server port (default: 8000)")
    
    # Model artifacts path
    parser.add_argument(
        '--model-path', 
        default=os.getenv('MODEL_PATH', '../data/model/'),  # Fallback to ENV var MODEL_PATH or default value
        help='Directory where datasets are stored (default: ../data/model/ or ENV var MODEL_PATH)'
    )

    # Visualizations save path
    parser.add_argument(
        '--prediction-visuals-path',
        default=os.getenv('PREDICTION_VISUALS_PATH', '../data/prediction_visuals/'),  # Fallback to ENV var PREDICTION_VISUALS_PATH or default value
        help='Directory where prediction visuals are stored (default: ../data/prediction_visuals/ or ENV var PREDICTION_VISUALS_PATH)'
    )

    cli_args_group = parser.add_argument_group("CLI Arguments")
    cli_args_group.add_argument('--amount', type=float, help="Transaction Amount (float)")
    cli_args_group.add_argument('--hour', type=int, help="Transction Time - Hour (HH)")
    cli_args_group.add_argument('--day', type=int, help="Transction Time - Day (DD)")
    cli_args_group.add_argument('--month', type=int, help="Transction Time - Month (MM)")
    cli_args_group.add_argument('--cc-bin', help="Credit Card BIN (first 6 digits)")
    cli_args_group.add_argument('--street', help="Street Address")
    cli_args_group.add_argument('--city', help="City")
    cli_args_group.add_argument('--state', help="State")
    cli_args_group.add_argument('--zip', type=int, help="ZipCode")
    cli_args_group.add_argument('--city-pop', type=int, help="City Population (int)")
    cli_args_group.add_argument('--dob', help='Account Holder - Date of Birth (YYYY-MM-DD)')
    cli_args_group.add_argument('--gender', choices=["M","F"], help="Account Holder - Gender")
    cli_args_group.add_argument('--job', help="Account Holder - Occupation")
    cli_args_group.add_argument('--lat', type=float, help="Account Holder - Latitude (float)")
    cli_args_group.add_argument('--long', type=float, help="Account Holder - Longitude (float)")
    cli_args_group.add_argument('--merchant', help="Merchant Name")
    cli_args_group.add_argument('--merch-lat', type=float, help="Merchant - Latitude (float)")
    cli_args_group.add_argument('--merch-long', type=float, help="Merchant - Longitude (float)")
    cli_args_group.add_argument('--category',
                        choices=["entertainment","food_dining","gas_transport","grocery_net","grocery_pos",
                                 "health_fitness","home","kids_pets","misc_net","misc_pos","personal_care",
                                 "shopping_net","shopping_pos","travel"],
                        help="Merchant - Category")
    

    # Add DB/model path arguments from original script
    args = parser.parse_args()
    
    predictor = FraudPredictionCLI(args.model_path)
    
    if args.api_server:
        run_api_server(predictor, host=args.host, port=args.port)
    else:
        # Check that all CLI arguments are provided when not in API mode
        missing_arguments = [arg for arg in vars(args) if getattr(args, arg) is None and arg != "api_server"]
        
        if missing_arguments:
            parser.error(f"The following arguments are required in CLI mode: {missing_arguments}")
        
        result = predictor.predict_single(args)

# Examples:
# python predict_fraud.py --amount 1077.69 --hour 22 --day 21 --month 6 --cc-bin 400567 --street "458 Phillips Island Apt. 768" --city "Denham Springs" --state LA --zip 70726 --city-pop 71335 --dob 1994-05-31 --gender M --job "Herbalist" --lat 30.459 --long -90.9027 --merchant "Heathcote, Yost and Kertzmann" --merch-lat 31.204974 --merch-long -90.261595 --category shopping_net
# python predict_fraud.py --amount 41.28 --hour 12 --day 21 --month 6 --cc-bin 359821 --street "9333 Valentine Point" --city "Bellmore" --state NY --zip 11710 --city-pop 34496 --dob 1970-10-21 --gender F --job "Librarian, public" --lat 40.6729 --long -73.5365 --merchant "Swaniawski, Nitzsche and Welch" --merch-lat 40.49581 --merch-long -74.196111 --category health_fitness
# python predict_fraud.py --amount 843.91 --hour 23 --day 2 --month 1 --cc-bin 461331 --street "542 Steve Curve Suite 011" --city "Collettsville" --state NC --zip 28611 --city-pop 885 --dob 1988-09-15 --gender M --job "Soil scientist" --lat 35.9946 --long -81.7266 --merchant "Ruecker Group" --merch-lat 35.985612 --merch-long -81.383306 --category misc_net