import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

class DataFrameImporter:
    def __init__(self, db_user, db_pass, db_host, db_name):
        self.engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}"
        )
        self.api = KaggleApi()
        self.api.authenticate()

    def preprocess_data(self, df, csv_file):
        """Override this method for dataset-specific preprocessing"""
        # Common preprocessing for all datasets
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        if 'is_fraud' in df.columns:
            df['is_fraud'] = df['is_fraud'].astype(bool)
        if 'merchant' in df.column:
            df['merchant'] = df['merchant'].str.replace('fraud_', '', regex=False)
        return df
    
    def get_table_name(self, csv_file):
        """Override this to set table names based on CSV filename"""
        return os.path.splitext(os.path.basename(csv_file))[0].lower()
    
    def create_table(self, table_name):
        """Override this for dataset-specific table schemas"""
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    idx SERIAL PRIMARY KEY,
                    trans_date_trans_time TIMESTAMP,
                    cc_num BIGINT,
                    merchant VARCHAR(255),
                    category VARCHAR(50),
                    amt FLOAT,
                    first VARCHAR(50),
                    last VARCHAR(50),
                    gender VARCHAR(1),
                    street VARCHAR(255),
                    city VARCHAR(100),
                    state VARCHAR(2),
                    zip INTEGER,
                    lat FLOAT,
                    long FLOAT,
                    city_pop INTEGER,
                    job VARCHAR(255),
                    dob DATE,
                    trans_num VARCHAR(255) UNIQUE,
                    unix_time INTEGER,
                    merch_lat FLOAT,
                    merch_long FLOAT,
                    is_fraud BOOLEAN
                )
            """))
            conn.commit()
    
    def import_csv(self, csv_path):
        """Main import workflow"""
        table_name = self.get_table_name(csv_path)
        
        print(f"Processing {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=['trans_date_trans_time', 'dob'])
        df = self.preprocess_data(df, csv_path)
        
        print(f"Creating table {table_name}...")
        self.create_table(table_name)
        
        print(f"Deleting old data from {table_name}...")
        with self.engine.connect() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY;"))
            conn.commit()
        
        print("Inserting data...")
        self._chunked_insert(df, table_name)
        
        print(f"Successfully imported {len(df)} records to {table_name}")

    def _chunked_insert(self, df, table_name, chunksize=1000):
        """Insert data with progress tracking"""
        total_rows = len(df)
        with tqdm(total=total_rows, desc=f"Writing to {table_name}") as pbar:
            for i in range(0, total_rows, chunksize):
                chunk = df.iloc[i:i + chunksize]
                chunk.to_sql(
                    name=table_name,
                    con=self.engine,
                    if_exists='append',
                    index=False,
                )
                pbar.update(len(chunk))

class FraudTrainImporter(DataFrameImporter):
    def get_table_name(self, csv_file):
        return "fraud_train"

class FraudTestImporter(DataFrameImporter):
    def get_table_name(self, csv_file):
        return "fraud_test"

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Import Kaggle dataset into PostgreSQL')
    parser.add_argument('--db-host', required=True, help='PostgreSQL host')
    parser.add_argument('--db-name', required=True, help='Database name')
    parser.add_argument('--db-user', required=True, help='Database user')
    parser.add_argument('--db-pass', required=True, help='Database password')
    parser.add_argument('--data-dir', default='../data', help='Directory where datasets are stored (default: ../data)')
    args = parser.parse_args()

    # Download dataset from Kaggle
    print("Downloading dataset from Kaggle...")

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    api.dataset_download_files(
        'kartik2112/fraud-detection',
        path=args.data_dir,
        unzip=True,
        quiet=False
    )
    
    # Initialize importers
    train_importer = FraudTrainImporter(args.db_user, args.db_pass, args.db_host, args.db_name)
    test_importer = FraudTestImporter(args.db_user, args.db_pass, args.db_host, args.db_name)

    # Import data into database
    train_importer.import_csv(os.path.join(args.data_dir, 'fraudTrain.csv'))
    test_importer.import_csv(os.path.join(args.data_dir, 'fraudTest.csv'))
