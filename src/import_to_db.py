import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
import kagglehub
from datetime import datetime

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
        dataset_path = kagglehub.dataset_download(
            'kartik2112/fraud-detection',
            path='../data'
        )
        csv_file = os.path.join(dataset_path, 'fraudTrain.csv')
        table_name = "fraudTrain"

        # Load data with proper typing
        print("Loading data into pandas...")
        df = pd.read_csv(csv_file, parse_dates=['trans_date_trans_time', 'dob'])

        # Create PostgreSQL connection
        print("Connecting to database...")
        engine = create_engine(
            f"postgresql+psycopg2://{args.db_user}:{args.db_password}@{args.db_host}/{args.db_name}"
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

        # Insert data
        # Insert data only if it doesn't exist
        print("Inserting data...")
        df['is_fraud'] = df['is_fraud'].astype(bool)

        # Use raw SQL for upsert logic
        with engine.connect() as conn:
            for _, row in df.iterrows():
                conn.execute(text(f"""
                    INSERT INTO {args.table_name} (
                        trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender,
                        street, city, state, zip, lat, long, city_pop, job, dob, trans_num,
                        unix_time, merch_lat, merch_long, is_fraud
                    )
                    VALUES (
                        :trans_date_trans_time, :cc_num, :merchant, :category, :amt, :first, :last,
                        :gender, :street, :city, :state, :zip, :lat, :long, :city_pop,
                        :job, :dob, :trans_num, :unix_time, :merch_lat, :merch_long, :is_fraud
                    )
                    ON CONFLICT (trans_num) DO NOTHING
                """), {
                    "trans_date_trans_time": row['trans_date_trans_time'],
                    "cc_num": row['cc_num'],
                    "merchant": row['merchant'],
                    "category": row['category'],
                    "amt": row['amt'],
                    "first": row['first'],
                    "last": row['last'],
                    "gender": row['gender'],
                    "street": row['street'],
                    "city": row['city'],
                    "state": row['state'],
                    "zip": row['zip'],
                    "lat": row['lat'],
                    "long": row['long'],
                    "city_pop": row['city_pop'],
                    "job": row['job'],
                    "dob": row['dob'],
                    "trans_num": row['trans_num'],
                    "unix_time": row['unix_time'],
                    "merch_lat": row['merch_lat'],
                    "merch_long": row['merch_long'],
                    "is_fraud": row['is_fraud']
                })

        print(f"Successfully imported {len(df)} records to {table_name}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
