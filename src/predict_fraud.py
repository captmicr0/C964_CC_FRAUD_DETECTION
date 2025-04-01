import os
import argparse
import pandas as pd
import joblib
from sqlalchemy import create_engine

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
        features, identifiers = self._preprocess_input(input_source, args)
        self._validate_input(features)
        
        probabilities = self.model.predict_proba(features)[:, 1]
        return self._format_results(identifiers, probabilities)

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
        return pd.DataFrame([data])

    def _preprocess_csv(self, file_path):
        df = pd.read_csv(file_path, parse_dates=['trans_date_trans_time', 'dob'])
        return self._process_common_features(df)

    def _preprocess_db_table(self, table_name):
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.engine)
        return self._process_common_features(df)

    def _process_common_features(self, df):
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
        df['age'] = (pd.to_datetime('today') - df['dob']).dt.days // 365
        
        identifiers = df[['trans_num']] if 'trans_num' in df.columns else None
        features = df[self.required_features]
        
        return features, identifiers

    def _format_results(self, identifiers, probabilities):
        results = pd.DataFrame({'fraud_probability': probabilities})
        if identifiers is not None:
            results = pd.concat([identifiers.reset_index(drop=True), results], axis=1)
        return results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fraud Prediction System')
    
    # Input sources
    input_group = parser.add_argument_group('Input options')
    input_mutex = input_group.add_mutually_exclusive_group()
    input_mutex.add_argument('--input-file', help='Path to input CSV file')
    input_mutex.add_argument('--input-table', help='Database table name for input')
    
    # Output destinations
    output_group = parser.add_argument_group('Output options')
    output_mutex = output_group.add_mutually_exclusive_group()
    output_mutex.add_argument('--output-file', help='Path to output CSV file')
    output_mutex.add_argument('--output-table', help='Database table name for output')
    
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
    
    # Required args
    parser.add_argument('--model-path', required=True, help='Path to trained ML model')

    args = parser.parse_args()

    # Validate input/output requirements
    if (args.input_table or args.output_table) and not all([args.db_host, args.db_name, args.db_user, args.db_pass]):
        parser.error("Database credentials required when using database input/output")
    
    # Initialize database engine if needed
    engine = None
    if args.db_host:
        engine = create_engine(f"postgresql+psycopg2://{args.db_user}:{args.db_pass}@{args.db_host}/{args.db_name}")
    
    try:
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
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
