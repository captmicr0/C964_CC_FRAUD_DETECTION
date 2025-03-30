import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime
from tqdm import tqdm

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Import Kaggle dataset into PostgreSQL')
    parser.add_argument('--db-host', required=True, help='PostgreSQL host')
    parser.add_argument('--db-name', required=True, help='Database name')
    parser.add_argument('--db-user', required=True, help='Database user')
    parser.add_argument('--db-pass', required=True, help='Database password')
    args = parser.parse_args()

    try:
        # Download dataset from Kaggle
        print("Downloading dataset from Kaggle...")

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        api.dataset_download_files(
            'kartik2112/fraud-detection',
            path='../data',
            unzip=True,
            quiet=False
        )

        csv_file = os.path.join('../data', 'fraudTrain.csv')
        table_name = "fraud_train"

        # Load data with proper typing
        print("Loading data into pandas...")
        df = pd.read_csv(
            csv_file,
            parse_dates=['trans_date_trans_time', 'dob']
        )

        df.drop(columns=['Unnamed: 0'], inplace=True)  # Explicitly drop the column
        df['is_fraud'] = df['is_fraud'].astype(bool) # Convert is_fraud column to bool

        # Remove 'fraud_' prefix from merchant names
        print("Cleaning merchant names...")
        df['merchant'] = df['merchant'].str.replace('fraud_', '', regex=False)

        # Create PostgreSQL connection
        print("Connecting to database...")
        engine = create_engine(
            f"postgresql+psycopg2://{args.db_user}:{args.db_pass}@{args.db_host}/{args.db_name}"
        )

        # Create table with optimized schema
        print("Creating table schema if needed...")
        with engine.connect() as conn:
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

        # Delete all existing data from the table
        print(f"Deleting all existing data from '{table_name}'...")
        with engine.connect() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY;"))
            conn.commit()

        # Insert new data into the table with progress bar
        print(f"Inserting data into '{table_name}'... This may take a while...")

        chunksize = 1000  # Process in chunks of 1000 rows
        total_rows = len(df)

        with tqdm(total=total_rows, desc="Writing to database") as pbar:
            for i in range(0, total_rows, chunksize):
                chunk = df.iloc[i:i + chunksize]
                chunk.to_sql(
                    name=table_name,
                    con=engine,
                    if_exists='append',
                    index=False,
                )
                pbar.update(len(chunk))

        print(f"Successfully imported {len(df)} records into '{table_name}'.")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
