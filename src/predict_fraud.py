import argparse
import os
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FraudPredictionCLI:
    def __init__(self, model_path):
        # Load both model and preprocessing artifacts
        pbar = tqdm(total=4, desc="Loading Model Artifacts", unit="step")
        self.model = joblib.load(os.path.join(model_path, 'unbalanced_model.pkl'))
        pbar.update(1)
        self.model_smote = joblib.load(os.path.join(model_path, 'balanced_model.pkl'))
        pbar.update(1)
        self.label_encoders = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
        pbar.update(1)
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        pbar.update(1)
        
        # Required features
        self.required_features = [
            'amt', 'city_pop', 'hour', 'day', 'month',
            'lat', 'long', 'merch_lat', 'merch_long',
            'category', 'gender', 'state', 'dob'
        ]

    def feature_engineering(self, df_test):
        """Feature engineering input data to useable data"""
        # Initialize progress bar with a placeholder total
        pbar = tqdm(total=7, desc="Feature Engineering", unit="step")
        
        # Drop columns that are not important
        # keep: trans_date_trans_time, cc_num,merchant, category, amt, gender, street, city, state, zip, lat, long, city_pop, job, dob, merch_lat, merch_long, is_fraud
        columns_to_drop = ['trans_num', 'unix_time', 'first', 'last']
        df_test.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
        pbar.update(1)

        # Choose features and target for the training and test data
        X_test = df_test.drop('is_fraud', axis=1)  # features
        if 'is_fraud' in X_test.columns:
            y_test = df_test['is_fraud']  # target, optional
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

        print("Feature set of testing data:")
        print(X_test.columns)

        # Return processed data
        if 'is_fraud' in X_test.columns:
            return X_test, y_test

        return X_test, None
    
    def single_to_df(self, args):
        """Convert a single transaction into a DataFrame."""
        return pd.DataFrame([{
            'amt': args.amount,
            'city_pop': args.city_pop,
            'hour': args.hour,
            'day': args.day,
            'month': args.month,
            'lat': args.lat,
            'long': args.long,
            'merch_lat': args.merch_lat,
            'merch_long': args.merch_long,
            'category': args.category,
            'gender': args.gender,
            'state': args.state,
            'dob': args.dob
        }])

    def predict_single(self, args):
        """Make a prediction using the trained model."""
        # Convert a single transaction into a DataFrame
        single_df = pd.DataFrame([{
            'amt': args.amount,
            'city_pop': args.city_pop,
            'hour': args.hour,
            'day': args.day,
            'month': args.month,
            'lat': args.lat,
            'long': args.long,
            'merch_lat': args.merch_lat,
            'merch_long': args.merch_long,
            'category': args.category,
            'gender': args.gender,
            'state': args.state,
            'dob': args.dob
        }])

        # Feature engineering on input data
        X_test, y_test = self.feature_engineering(single_df)
        
        # Make a prediction using the model
        prediction = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        print(f"Prediction: {'Fraudulent' if result['prediction'] == 1 else 'Non-Fraudulent'}")
        print(f"Probability of Fraud: {result['probability_of_fraud']:.2f}")

        return {
            "prediction": int(prediction[0]),  # 0: Non-fraudulent, 1: Fraudulent
            "probability_of_fraud": probabilities[0]
        }
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Prediction System')
    
    # Updated arguments matching new features
    # idx, trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud
    # 0, 2019-01-01 00:00:18, 2703186189652095, "fraud_Rippin,  Kub and Mann", misc_net, 4.97, Jennifer, Banks, F, 561 Perry Cove, Moravian Falls, NC, 28654, 36.0788, -81.1781, 3495, "Psychologist,  counselling", 1988-03-09, 0b242abb623afc578575680df30655b9, 1325376018, 36.011293, -82.048315, 0

    parser.add_argument('--amount', type=float, required=True, help="Transaction Amount (float)")
    parser.add_argument('--hour', type=int, required=True, help="Transction Time - Hour (HH)")
    parser.add_argument('--day', type=int, required=True, help="Transction Time - Day (DD)")
    parser.add_argument('--month', type=int, required=True, help="Transction Time - Month (MM)")
    parser.add_argument('--state', required=True, help="State (int)")
    parser.add_argument('--city-pop', type=int, required=True, help="City Population (int)")
    parser.add_argument('--dob', required=True, help='Account Holder - Date of Birth (YYYY-MM-DD)')
    parser.add_argument('--gender', choices=["M","F"], required=True, help="Account Holder - Gender")
    parser.add_argument('--lat', type=float, required=True, help="Account Holder - Latitude (float)")
    parser.add_argument('--long', type=float, required=True, help="Account Holder - Longitude (float)")
    parser.add_argument('--merch-lat', type=float, required=True, help="Merchant - Latitude (float)")
    parser.add_argument('--merch-long', type=float, required=True, help="Merchant - Longitude (float)")
    parser.add_argument('--category',
                        choices=["entertainment","food_dining","gas_transport","grocery_net","grocery_pos",
                                 "health_fitness","home","kids_pets","misc_net","misc_pos","personal_care",
                                 "shopping_net","shopping_pos","travel"],
                        required=True, help="Merchant - Category")
    
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

    # Add DB/model path arguments from original script
    args = parser.parse_args()
    
    # Initialize the predictor and make predictions
    predictor = FraudPredictionCLI(args.model_path)
    result = predictor.predict_single(args)
    