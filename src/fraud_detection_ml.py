import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, 
                            roc_curve, 
                            roc_auc_score,
                            confusion_matrix)
from imblearn.over_sampling import SMOTE
import joblib

class FraudDetector:
    def __init__(self, db_user, db_pass, db_host, db_name):
        self.engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"
        )
        self.model = None
        self.features = None
        
    def load_data(self, table_name):
        """Load and preprocess data from PostgreSQL"""
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.engine)
        
        # Feature engineering
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
        df['age'] = (pd.to_datetime('today') - df['dob']).dt.days // 365
        
        # Select features
        self.features = [
            'amt', 'city_pop', 'age', 'trans_hour', 'trans_day',
            'lat', 'long', 'merch_lat', 'merch_long'
        ]
        
        # Encode categoricals
        df = pd.get_dummies(df, columns=['category', 'gender', 'state'], drop_first=True)
        
        return df[self.features + [c for c in df.columns if c.startswith('category_')]], df['is_fraud']

    def train_model(self, test_size=0.2, table_name='fraud_train'):
        """Train and evaluate model with class balancing"""
        X, y = self.load_data(table_name)
        
        # Handle imbalance
        smote = SMOTE(sampling_strategy=0.1, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=test_size, random_state=42
        )
        
        # Model training
        self.model = XGBClassifier(
            scale_pos_weight=100,
            learning_rate=0.01,
            max_depth=5
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred):.2f}")
        
        # Generate visualizations
        self._plot_roc_curve(y_test, self.model.predict_proba(X_test)[:,1])
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_feature_importance()

    def save_model(self, path='fraud_model.pkl'):
        """Save trained model"""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection System')

    # Add arguments with fallback to environment variables
    parser.add_argument(
        '--db-host', 
        default=os.getenv('DB_HOST'), 
        required=os.getenv('DB_HOST') is None, 
        help='PostgreSQL host (can fallback to ENV var DB_HOST)'
    )
    parser.add_argument(
        '--db-name', 
        default=os.getenv('DB_NAME'), 
        required=os.getenv('DB_NAME') is None, 
        help='Database name (can fallback to ENV var DB_NAME)'
    )
    parser.add_argument(
        '--db-user', 
        default=os.getenv('DB_USER'), 
        required=os.getenv('DB_USER') is None, 
        help='Database user (can fallback to ENV var DB_USER)'
    )
    parser.add_argument(
        '--db-pass', 
        default=os.getenv('DB_PASS'), 
        required=os.getenv('DB_PASS') is None, 
        help='Database password (can fallback to ENV var DB_PASS)'
    )
    parser.add_argument(
        '--save-model', 
        default=os.getenv('SAVE_MODEL', 'fraud_model.pk1'),  # Fallback to ENV var SAVE_MODEL or default value
        help='Directory where datasets are stored (default: fraud_model.pkl or ENV var SAVE_MODEL)'
    )
    
    args = parser.parse_args()
    
    # Train and evaluate model
    detector = FraudDetector(args.db_user, args.db_pass, args.db_host, args.db_name)
    detector.train_model()
    detector.save_model(args.save_model)
