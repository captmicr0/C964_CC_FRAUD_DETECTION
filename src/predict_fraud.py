import os
import argparse
import pandas as pd
import joblib
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

class FraudPredictionCLI:
    def __init__(self, model_path, db_engine=None):
        self.model = joblib.load(model_path)
        self.engine = db_engine
        self.required_features = [
            'amt', 'city_pop', 'age', 'trans_hour', 'trans_day',
            'lat', 'long', 'merch_lat', 'merch_long'
        ]

    def predict(self, args):
        input_source = self._get_input_source(args)
        features, identifiers, categories = self._preprocess_input(input_source, args)
        self._validate_input(features)
        
        probabilities = self.model.predict_proba(features)[:, 1]
        return self._format_results(features, identifiers, probabilities, categories)

    def _validate_input(self, input_data):
        missing = set(self.required_features) - set(input_data.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
    
    def _get_input_source(self, args):
        if args.input_file:
            return ('file', args.input_file)
        if args.input_table:
            return ('table', args.input_table)
        return ('single', None)

    def _preprocess_input(self, input_source, args):
        source_type, source_value = input_source
        
        if source_type == 'file':
            return self._preprocess_csv(source_value)
        if source_type == 'table':
            return self._preprocess_db_table(source_value)
        return self._preprocess_single(args), None

    def _preprocess_single(self, args):
        # One-hot encode the category input for single transaction processing
        category_columns = [
            'category_entertainment', 'category_grocery_pos',
            'category_misc_net', 'category_travel'
        ]

        data = {
            'amt': args.amount,
            'city_pop': args.city_pop,
            'age': args.age,
            'trans_hour': args.hour,
            'trans_day': args.day,
            'lat': args.lat,
            'long': args.long,
            'merch_lat': args.merch_lat,
            'merch_long': args.merch_long
        }
        
        # Add one-hot encoded category columns
        for col in category_columns:
            data[col] = 1 if col == f"category_{args.category}" else 0
        
        return pd.DataFrame([data])

    def _preprocess_csv(self, file_path):
        df = pd.read_csv(file_path, parse_dates=['trans_date_trans_time', 'dob'])

        # One-hot encode the category column for batch processing
        df = pd.get_dummies(df, columns=['category'], prefix='category')

        return self._process_common_features(df)

    def _preprocess_db_table(self, table_name):
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.engine)

        # One-hot encode the category column for batch processing
        df = pd.get_dummies(df, columns=['category'], prefix='category')

        return self._process_common_features(df)

    def _process_common_features(self, df):
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
        df['age'] = (pd.to_datetime('today') - df['dob']).dt.days // 365
        
        identifiers = df[['trans_num']] if 'trans_num' in df.columns else None
        
        # Include one-hot encoded category columns dynamically
        category_cols = [col for col in df.columns if col.startswith('category_')]
        if category_cols:
            df['category'] = df[category_cols].idxmax(axis=1).str.replace('category_', '')
        
        features = df[self.required_features + category_cols]
        
        return features, identifiers, df.get('category', None)

    def _format_results(self, features, identifiers, probabilities, categories):
        results = pd.DataFrame({
            'fraud_probability': probabilities,
            'amt': features['amt'],  # Add amt from processed features
            'city_pop': features['city_pop'],  # Add city_pop
            'category': categories  # Add original category
        })
        if identifiers is not None:
            results = pd.concat([identifiers.reset_index(drop=True), results], axis=1)
        return results
    
    @staticmethod
    def visualize_results(results, save_path):
        """Generate visualizations for batch results."""
        
        # Ensure the save path exists
        os.makedirs(save_path, exist_ok=True)

        # Pie Chart: Fraud Risk Distribution
        bins = [0, 30, 70, 100]
        labels = ['Low Risk', 'Medium Risk', 'High Risk']
        results['Risk Level'] = pd.cut(results['fraud_probability'] * 100, bins=bins, labels=labels)
        
        risk_counts = results['Risk Level'].value_counts()
        
        plt.figure(figsize=(8, 8))
        plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['green', 'orange', 'red'])
        plt.title('Fraud Risk Distribution')
        plt.savefig(os.path.join(save_path, 'fraud_risk_pie_chart.png'))
        
        # Histogram: Fraud Probability Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results['fraud_probability'] * 100, bins=10, color='blue', alpha=0.7)
        plt.xlabel('Fraud Probability (%)')
        plt.ylabel('Number of Transactions')
        plt.title('Distribution of Fraud Probabilities')
        plt.savefig(os.path.join(save_path, 'fraud_probability_histogram.png'))
        
        # Heatmap: Correlation Matrix
        correlation_data = results[['amt', 'city_pop', 'fraud_probability']]
        
        corr_matrix = correlation_data.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(save_path, 'correlation_heatmap.png'))
        
        # Scatter Plot: Transaction Amount vs Fraud Probability
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='amt',
            y='fraud_probability',
            data=results,
            hue='Risk Level',
            palette={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        )
        plt.xlabel('Transaction Amount ($)')
        plt.ylabel('Fraud Probability (%)')
        plt.title('Transaction Amount vs Fraud Probability')
        plt.legend(title='Risk Level')
        plt.savefig(os.path.join(save_path, 'scatter_plot_amt_vs_fraud.png'))

        print("Visualizations saved to {path}:")
        print("- fraud_risk_pie_chart.png")
        print("- fraud_probability_histogram.png")
        print("- correlation_heatmap.png")
        print("- scatter_plot_amt_vs_fraud.png")

        # Pie Chart: Fraud Probability by Category
        plt.figure(figsize=(10, 8))
        category_counts = results.groupby('category')['fraud_probability'].mean()
        plt.pie(category_counts, 
                labels=category_counts.index,
                autopct=lambda pct: f"{pct:.1f}%\n({category_counts.mean():.2f})",
                startangle=90)
        plt.title('Average Fraud Probability by Transaction Category')
        plt.savefig(os.path.join(save_path, 'category_fraud_pie.png'))
        

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fraud Prediction System')
    
    # Input sources
    input_group = parser.add_argument_group('Input options')
    input_mutex = input_group.add_mutually_exclusive_group()
    input_mutex.add_argument('--input-file', help='Path to input CSV file')
    input_mutex.add_argument('--input-table', default='fraud_test', help='Database table name for input')
    
    # Output destinations
    output_group = parser.add_argument_group('Output options')
    output_mutex = output_group.add_mutually_exclusive_group()
    output_mutex.add_argument('--output-file', help='Path to output CSV file')
    output_mutex.add_argument('--output-table', default='prediction_results', help='Database table name for output')
    
    # Database connection
    db_group = parser.add_argument_group('Database connection')
    parser.add_argument(
        '--db-host', 
        default=os.getenv('DB_HOST'),
        help='PostgreSQL host (can fallback to ENV var DB_HOST)'
    )
    parser.add_argument(
        '--db-name', 
        default=os.getenv('DB_NAME'),
        help='Database name (can fallback to ENV var DB_NAME)'
    )
    parser.add_argument(
        '--db-user', 
        default=os.getenv('DB_USER'),
        help='Database user (can fallback to ENV var DB_USER)'
    )
    parser.add_argument(
        '--db-pass', 
        default=os.getenv('DB_PASS'),
        help='Database password (can fallback to ENV var DB_PASS)'
    )
    
    # Single transaction args
    single_group = parser.add_argument_group('Single transaction')
    single_group.add_argument('--amount', type=float, help='Transaction amount')
    single_group.add_argument('--city-pop', type=int, help='City population')
    single_group.add_argument('--age', type=int, help='Customer age')
    single_group.add_argument('--hour', type=int, help='Transaction hour (0-23)')
    single_group.add_argument('--day', type=int, help='Transaction day (0-6)')
    single_group.add_argument('--lat', type=float, help='Customer latitude')
    single_group.add_argument('--long', type=float, help='Customer longitude')
    single_group.add_argument('--merch-lat', type=float, help='Merchant latitude')
    single_group.add_argument('--merch-long', type=float, help='Merchant longitude')
    
    # Category argument for single transactions
    single_group.add_argument('--category', choices=['entertainment', 'grocery_pos', 'misc_net', 'travel'],
                               help="Transaction category (e.g., entertainment, grocery_pos)")

    # Prediction visualizations save path
    parser.add_argument(
        '--prediction-visuals-path',
        default=os.getenv('PREDICTION_VISUALS_PATH', '../data/prediction_visuals/'),  # Fallback to ENV var PREDICTION_VISUALS_PATH or default value
        help='Directory where datasets are stored (default: ../data/prediction_visuals/ or ENV var PREDICTION_VISUALS_PATH)'
    )

    # Required args
    parser.add_argument(
        '--model-path',
        default=os.getenv('SAVE_MODEL', '../data/fraud_model.pkl'),  # Fallback to ENV var SAVE_MODEL or default value
        help='Directory where datasets are stored (default: ../data/fraud_model.pkl or ENV var SAVE_MODEL)'
    )

    args = parser.parse_args()

    # Validate input/output requirements
    if (args.input_table or args.output_table) and not all([args.db_host, args.db_name, args.db_user, args.db_pass]):
        parser.error("Database credentials required when using database input/output")
    
    # Initialize database engine if needed
    engine = None
    if args.db_host:
        engine = create_engine(f"postgresql+psycopg2://{args.db_user}:{args.db_pass}@{args.db_host}/{args.db_name}")

    # Run prediction
    predictor = FraudPredictionCLI(args.model_path, engine)
    results = predictor.predict(args)
    
    if args.output_file:
        results.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    elif args.output_table:
        results.to_sql(args.output_table, engine, if_exists='replace', index=False)
        print(f"Predictions saved to database table {args.output_table}")
    else:
        print("Fraud Probability Results:")
        print(results.to_string(index=False))
    
    if args.output_file or args.output_table:
        predictor.visualize_results(results, args.prediction_visuals_path)
