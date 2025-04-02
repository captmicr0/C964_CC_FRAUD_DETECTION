import logging
from datetime import datetime
#from tqdm import tqdm
from tqdm_loggable.auto import tqdm

import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, 
                            roc_curve, 
                            roc_auc_score,
                            confusion_matrix,
                            accuracy_score)
from imblearn.over_sampling import SMOTE
import joblib

# Log file directory from ENV
log_dir = os.environ.get("LOG_DIR", os.getcwd())

# Ensure the log directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Get the current date and time when the program starts
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Configure logging with a dynamic filename
log_filename = f"FraudDetector_{start_time}.log"

# Construct the full log file path
log_path = os.path.join(log_dir, log_filename)

# Create a logger
logger = logging.getLogger("FraudDetector")
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

class FraudDetector:
    def __init__(self, db_user, db_pass, db_host, db_name, eda_visuals_path, model_visuals_path):
        self.engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"
        )

        self.label_encoder = None
        self.scaler = None

        self.model = None
        self.model_smote = None

        self.eda_visuals_path = eda_visuals_path
        self.model_visuals_path = model_visuals_path

        # Ensure the save paths exists
        os.makedirs(self.eda_visuals_path, exist_ok=True)
        os.makedirs(self.model_visuals_path, exist_ok=True)
        
    def load_data(self, train_table, test_table):
        """Load and preprocess data from PostgreSQL"""
        def get_row_count(table_name):
            """Get the row count of a table."""
            query = text(f"SELECT COUNT(*) FROM {table_name}")
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return result.scalar()  # Fetch the scalar value (row count)
        
        # Get row counts for training and testing tables
        train_row_count = get_row_count(train_table)
        test_row_count = get_row_count(test_table)

        # Load training data with progress bar
        query = text(f"SELECT * FROM {train_table}")
        chunks = []
        # Visualize missing values and target variable distribution for training data
        with tqdm(total=train_row_count, desc="Loading Train Data", unit="rows") as pbar:
            for chunk in pd.read_sql(query, self.engine, chunksize=1000):  # Load in chunks of 1000 rows
                chunks.append(chunk)
                pbar.update(len(chunk))
        df_train = pd.concat(chunks)

        # Visualize missing values and target variable distribution for training data
        self.visualize_missing_values(df_train)
        self.visualize_is_fraud(df_train['is_fraud'])

        # Load testing data with progress bar
        query = text(f"SELECT * FROM {test_table}")
        chunks = []
        
        with tqdm(total=test_row_count, desc="Loading Test Data", unit="rows") as pbar:
            for chunk in pd.read_sql(query, self.engine, chunksize=1000):  # Load in chunks of 1000 rows
                chunks.append(chunk)
                pbar.update(len(chunk))
        df_test = pd.concat(chunks)
        
        return df_train, df_test

    def visualize_missing_values(self, df, fn='missing_values_bar_graph.png'):
        """Generate visualizations for EDA."""
        missing_values = df.isnull().sum()
        plt.figure(figsize=(10, 6))
        missing_values.plot(kind='bar', color='skyblue')
        plt.title('Missing Values per Column')
        plt.xlabel('Column Name')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_visuals_path, fn))
        plt.close()

    def visualize_is_fraud(self, y, fn='is_fraud_distribution.png'):
        """Generate a bar graph showing the distribution of the target variable (is_fraud)."""
        plt.figure(figsize=(8, 6))
        y.value_counts().plot(kind='bar', color=['skyblue', 'orange'])
        plt.title('Distribution of Target Variable (is_fraud)')
        plt.xlabel('Fraud (1) vs Non-Fraud (0)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_visuals_path, fn))
        plt.close()

    def _init_model(self, model_type):
        if model_type == 'randomforest':
            return RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1, # Use all CPU cores
                random_state=42
            )
        
        # fallback is always xgboost
        return XGBClassifier(
            scale_pos_weight=100,
            learning_rate=0.01,
            max_depth=5
        )

    def train_model(self, test_size=0.2, model_type='randomforest', train_table='fraud_train', test_table='fraud_test'):
        """Train and evaluate model with class balancing"""
        # Load data and select features
        df_train, df_test = self.load_data(train_table, test_table)

        X_train, y_train, X_test, y_test = self.feature_engineering(df_train, df_test)
        
        # Model training
        logger.info("Training unbalanced model...")
        self.model = self._init_model(model_type)
        self.model.fit(X_train, y_train)

        # Predict test data
        logger.info("Predicting on test data...")
        prediction = self.model.predict(X_test)

        # Evaluate and create visualizations
        logger.info("Model Evaluation:")
        logger.info(classification_report(y_test, prediction), end="")
        logger.info(f"AUC-ROC Score: {roc_auc_score(y_test, prediction):.2f}")
        logger.info(f"Accuracy: {accuracy_score(y_test, prediction):.2f}")

        self._plot_roc_curve(y_test, self.model.predict_proba(X_test)[:,1], fn='unbalanced.roc_curve.png')
        self._plot_confusion_matrix(y_test, prediction, fn='unbalanced.confusion_matrix.png')
        self._plot_feature_importance(X_train, fn='unbalanced.feature_importance.png')

        # Handle imbalance with SMOTE
        logger.info("Balancing training data...")
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # Model training
        logger.info("Training balanced model...")
        self.model_smote = self._init_model(model_type)
        self.model_smote.fit(X_res, y_res)

        # Predict test data
        logger.info("Predicting on test data...")
        prediction_smote = self.model_smote.predict(X_test)

        # Evaluate and create visualizations
        logger.info("Balanced Model Evaluation (SMOTE):")
        logger.info(classification_report(y_test, prediction_smote), end="")
        logger.info(f"AUC-ROC Score: {roc_auc_score(y_test, prediction_smote):.2f}")
        logger.info(f"Accuracy: {accuracy_score(y_test, prediction_smote):.2f}")

        self._plot_roc_curve(y_test, self.model_smote.predict_proba(X_test)[:,1], fn='balanced.roc_curve.png')
        self._plot_confusion_matrix(y_test, prediction_smote, fn='balanced.confusion_matrix.png')
        self._plot_feature_importance(X_res, fn='balanced.feature_importance.png')
    
    def feature_engineering(self, df_train, df_test):
        """Feature engineering input data to useable data"""
        # Initialize progress bar with a placeholder total
        pbar = tqdm(total=9, desc="Feature Engineering", unit="step")
        
        # Drop columns that are not important
        # keep: trans_date_trans_time, cc_num,merchant, category, amt, gender, street, city, state, zip, lat, long, city_pop, job, dob, merch_lat, merch_long, is_fraud
        columns_to_drop = ['idx', 'trans_num', 'unix_time', 'first', 'last']
        df_train.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
        df_test.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
        pbar.update(1)

        # Convert cc_num to cc_bin (first 6 digits of cc_num) and drop the column
        for df in [df_train, df_test]:
            df['cc_bin'] = df['cc_num'].astype(str).str[:6]  # Extract first 6 digits
            df.drop('cc_num', axis=1, inplace=True)  # Drop original cc_num column
        pbar.update(1)
        
        # Choose features and target for the training and test data
        X_train = df_train.drop('is_fraud', axis=1)  # features
        y_train = df_train['is_fraud']  # target variable
        X_test = df_test.drop('is_fraud', axis=1)  # features
        y_test = df_test['is_fraud']  # target
        pbar.update(1)

        # Recognize numerical and categorical features in the training data
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        pbar.update(1)

        # Extract features from transaction date and time, then remove original column
        X_train['trans_date_trans_time'] = pd.to_datetime(X_train['trans_date_trans_time'])
        X_test['trans_date_trans_time'] = pd.to_datetime(X_test['trans_date_trans_time'])
        
        X_train['hour'] = X_train['trans_date_trans_time'].dt.hour
        X_train['day'] = X_train['trans_date_trans_time'].dt.day
        X_train['month'] = X_train['trans_date_trans_time'].dt.month

        X_test['hour'] = X_test['trans_date_trans_time'].dt.hour
        X_test['day'] = X_test['trans_date_trans_time'].dt.day
        X_test['month'] = X_test['trans_date_trans_time'].dt.month
        
        X_train.drop('trans_date_trans_time', axis=1, inplace=True, errors='ignore')
        X_test.drop('trans_date_trans_time', axis=1, inplace=True, errors='ignore')
        pbar.update(1)

        # Update after processing
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        pbar.update(1)

        # Dynamically update total once categorical_cols length is known
        pbar.total += len(categorical_cols)  # Add the number of categorical columns to the total
        pbar.refresh()  # Refresh the progress bar to reflect the new total

        # Convert categories into usable data
        self.label_encoder = LabelEncoder()
        for col in categorical_cols:
            # avoid unseen labels
            combined = pd.concat([X_train[col], X_test[col]])
            self.label_encoder.fit(combined)

            # transform data
            X_train[col] = self.label_encoder.transform(X_train[col])
            X_test[col] = self.label_encoder.transform(X_test[col])

            pbar.update(1)
        pbar.update(1)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])  # for test data don't use fit_transform
        pbar.update(1)

        # Make sure both sets of data have matching columns
        X_train, X_test = X_train.align(X_test, join='inner', axis=1)
        pbar.update(1)
        pbar.close()

        logger.info("Feature set of training and testing data:")
        logger.info(X_train.columns)

        # Return processed data
        return X_train, y_train, X_test, y_test

    def _plot_roc_curve(self, y_true, y_probs, fn='roc_curve.png'):
        """Generate ROC curve visualization"""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure()
        plt.plot(fpr, tpr, label='XGBoost (AUC = %0.2f)' % roc_auc_score(y_true, y_probs))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.model_visuals_path, fn))
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, fn='confusion_matrix.png'):
        """Generate confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.model_visuals_path, fn))
        plt.close()

    def _plot_feature_importance(self, X, fn='feature_importance.png'):
        """Generate feature importance visualization."""
        # Get feature importances from the model
        importances = self.model.feature_importances_
        
        # Combine feature names and importances
        feature_names = X.columns.tolist()  # Ensure this includes all features used in training
        if len(importances) != len(feature_names):
            raise ValueError("Mismatch between model features and provided feature names.")
        
        # Create a DataFrame for sorting and plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Plot top 10 important features
        top_features = importance_df.head(10)
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 10 Important Features')
        plt.gca().invert_yaxis()  # Invert y-axis to show top features at the top
        plt.tight_layout()
        
        # Save and show plot
        plt.savefig(os.path.join(self.model_visuals_path, fn))
        plt.close()

    def save_artifacts(self, path):
        """Save model artifacts"""
        # Initialize progress bar with a placeholder total
        pbar = tqdm(total=4, desc="Saving Model Artifacts", unit="step")

        # Ensure the save path exists
        os.makedirs(path, exist_ok=True)
        
        # Save label encoder
        label_encoder_path = os.path.join(path, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, label_encoder_path)
        pbar.update(1)

        # Save scaler
        scaler_path = os.path.join(path, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        pbar.update(1)

        # Save unbalanced model
        unbalanced_model_path = os.path.join(path, 'unbalanced_model.pkl')
        joblib.dump(self.model, unbalanced_model_path)
        pbar.update(1)

        # Save balanced model
        balanced_model_path = os.path.join(path, 'balanced_model.pkl')
        joblib.dump(self.model_smote, balanced_model_path)
        pbar.update(1)
        pbar.close()

        logger.info(f"Label Encoder saved to {label_encoder_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Unbalanced Model saved to {unbalanced_model_path}")
        logger.info(f"Balanced Model saved to {balanced_model_path}")
    
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
        '--model-type',
        choices=['xgboost', 'randomforest'],
        default=os.getenv('MODEL_TYPE', 'randomforest'),  # Fallback to ENV var MODEL_PATH or default value
        help='Model type (default: randomforest or ENV var MODEL_TYPE)'
    )
    parser.add_argument(
        '--model-path', 
        default=os.getenv('MODEL_PATH', '../data/model/'),  # Fallback to ENV var MODEL_PATH or default value
        help='Directory where datasets are stored (default: ../data/model/ or ENV var MODEL_PATH)'
    )
    
    # Visualizations save path
    parser.add_argument(
        '--eda-visuals-path',
        default=os.getenv('EDA_VISUALS_PATH', '../data/eda_visuals/'),  # Fallback to ENV var EDA_VISUALS_PATH or default value
        help='Directory where EDA visuals are stored (default: ../data/eda_visuals/ or ENV var EDA_VISUALS_PATH)'
    )

    # Visualizations save path
    parser.add_argument(
        '--model-visuals-path',
        default=os.getenv('MODEL_VISUALS_PATH', '../data/model_visuals/'),  # Fallback to ENV var MODEL_VISUALS_PATH or default value
        help='Directory where model visuals are stored (default: ../data/model_visuals/ or ENV var MODEL_VISUALS_PATH)'
    )
    
    args = parser.parse_args()
    
    # Train and evaluate model
    detector = FraudDetector(args.db_user, args.db_pass, args.db_host, args.db_name,
                             args.eda_visuals_path, args.model_visuals_path)
    detector.train_model(model_type=args.model_type)
    detector.save_artifacts(args.model_path)
