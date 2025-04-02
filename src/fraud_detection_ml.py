import os
import argparse
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, 
                            roc_curve, 
                            roc_auc_score,
                            confusion_matrix)
from imblearn.over_sampling import SMOTE
import joblib

class FraudDetector:
    def __init__(self, db_user, db_pass, db_host, db_name, eda_visuals_path, model_visuals_path):
        self.engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"
        )

        self.model = None
        self.features = None

        self.eda_visuals_path = eda_visuals_path
        self.model_visuals_path = model_visuals_path

        # Ensure the save paths exists
        os.makedirs(self.eda_visuals_path, exist_ok=True)
        os.makedirs(self.model_visuals_path, exist_ok=True)
        
    def load_data(self, train_table, test_table):
        """Load and preprocess data from PostgreSQL"""
        query = f"SELECT * FROM {train_table}"
        df_train = pd.read_sql(query, self.engine)
        self.visualize_missing_values(df_train)
        self.visualize_is_fraud(df_train['is_fraud'])

        query = f"SELECT * FROM {test_table}"
        df_test = pd.read_sql(query, self.engine)
        
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

    def train_model(self, test_size=0.2, train_table='fraud_train', test_table='fraud_test'):
        """Train and evaluate model with class balancing"""
        # Load data and select features
        df_train, df_test = self.load_data(train_table, test_table)
        X_train, y_train, X_test, y_test = self.feature_engineering(df_train, df_test)

        # Handle imbalance
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        # Model training
        self.model = XGBClassifier(
            scale_pos_weight=100,
            learning_rate=0.01,
            max_depth=5
        )
        self.model.fit(X_train, y_train)

        # Predict test data
        prediction = self.model.predict(X_test)

        # Evaluate and create visualizations
        print("\nModel Evaluation:")
        print(classification_report(y_test, prediction))
        print(f"AUC-ROC Score: {roc_auc_score(y_test, prediction):.2f}")

        self._plot_roc_curve(y_test, self.model.predict_proba(X_test)[:,1])
        self._plot_confusion_matrix(y_test, prediction)

        self._plot_feature_importance(X_train)
    
    def feature_engineering(self, df_train, df_test):
        # Drop columns that are not important
        # keep: trans_date_trans_time, cc_num,merchant, category, amt, gender, street, city, state, zip, lat, long, city_pop, job, dob, merch_lat, merch_long, is_fraud
        columns_to_drop = ['trans_num', 'unix_time', 'first', 'last']
        df_train.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
        df_test.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

        # Choose features and target for the training and test data
        X_train = df_train.drop('is_fraud', axis=1)  # features
        y_train = df_train['is_fraud']  # target variable

        X_test = df_test.drop('is_fraud', axis=1)  # features
        y_test = df_test['is_fraud']  # target

        # Recognize numerical and categorical features in the training data
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
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

        # Update after processing
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Convert categories into usable data
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            # avoid unseen labels
            combined = pd.concat([X_train[col], X_test[col]])
            label_encoder.fit(combined)

            # transform data
            X_train[col] = label_encoder.transform(X_train[col])
            X_test[col] = label_encoder.transform(X_test[col])
        
        # Scale features
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])  # for test data don't use fit_transform

        # Make sure both sets of data have matching columns
        X_train, X_test = X_train.align(X_test, join='inner', axis=1)

        print("Feature set of training and testing data:")
        print(X_train.columns)

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

    def save_model(self, path):
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
        default=os.getenv('SAVE_MODEL', '../data/fraud_model.pkl'),  # Fallback to ENV var SAVE_MODEL or default value
        help='Directory where datasets are stored (default: ../data/fraud_model.pkl or ENV var SAVE_MODEL)'
    )
    
    # Visualizations save path
    parser.add_argument(
        '--eda-visuals-path',
        default=os.getenv('EDA_VISUALS_PATH', '../data/eda_visuals/'),  # Fallback to ENV var EDA_VISUALS_PATH or default value
        help='Directory where datasets are stored (default: ../data/eda_visuals/ or ENV var EDA_VISUALS_PATH)'
    )

    # Visualizations save path
    parser.add_argument(
        '--model-visuals-path',
        default=os.getenv('MODEL_VISUALS_PATH', '../data/model_visuals/'),  # Fallback to ENV var MODEL_VISUALS_PATH or default value
        help='Directory where datasets are stored (default: ../data/model_visuals/ or ENV var MODEL_VISUALS_PATH)'
    )
    
    args = parser.parse_args()
    
    # Train and evaluate model
    detector = FraudDetector(args.db_user, args.db_pass, args.db_host, args.db_name,
                             args.eda_visuals_path, args.model_visuals_path)
    detector.train_model()
    detector.save_model(args.save_model)
