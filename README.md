# Credit Card Fraud Detection System

## Overview
This project focuses on building a machine learning-based system to detect fraudulent credit card transactions. The system uses advanced classification techniques to analyze transaction data and identify potential fraud while minimizing false positives and false negatives.

## Project Goals
- Develop an accurate machine learning model for fraud detection.
- Handle class imbalance in the dataset effectively.
- Provide a simple and efficient command-line interface for predictions.
- Demonstrate the application of machine learning techniques in solving real-world problems.

## Features
- Data preprocessing and feature engineering
- Implementation of machine learning models
- Handling class imbalance using oversampling techniques
- Performance evaluation using appropriate metrics for imbalanced datasets
- Command-line interface for analyzing transaction data
- Dockerized deployment for portability and scalability
- PostgreSQL database integration for data storage and retrieval

## Technologies Used
- **Programming Language**: Python
- **Libraries**: TBD (potentially: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Imbalanced-learn, SQLAlchemy, Psycopg2)
- **Database**: PostgreSQL
- **Tools**: VSCode, pgAdmin, Docker, Git

## Dataset
The project uses the publicly available [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). This dataset contains anonymized transaction data, including features derived using Principal Component Analysis (PCA).

## Project Structure
```
├── data/                # Contains dataset files (not included in the repo)
├── src/                 # Source code for preprocessing, modeling, and evaluation
├── docker/              # Docker-related files (e.g., Dockerfile, docker-compose.yml)
├── results/             # Outputs such as plots, metrics, and reports
├── README.md            # Project documentation (this file)
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8+
- PostgreSQL
- Docker


### Steps to Set Up PostgreSQL Database
1. Start a PostgreSQL instance locally or via Docker:
```
docker run --name fraud-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=fraud_detection -p 5432:5432 -d postgres
```

2. Import the dataset into PostgreSQL:
- Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/` directory.
- Run the provided script to create a table and import data:
  ```
  python src/import_to_db.py --db-host localhost --db-name fraud_detection --db-user postgres --db-password password --csv-file data/creditcard.csv
  ```

This script will:
- Create a table in PostgreSQL to store transaction data.
- Load the contents of the CSV file into the database.

### Steps to Run Locally
1. Clone this repository:
```
git clone https://github.com/captmicr0/C964_CC_FRAUD_DETECTION.git
cd C964_CC_FRAUD_DETECTION
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/` directory.

4. Run the main script for training or predictions:
- For Training:
  ```
  python src/main.py --mode train --db-host localhost --db-name fraud_detection --db-user postgres --db-password password
  ```
- For Predictions:
  ```
  python src/main.py --mode predict --db-host localhost --db-name fraud_detection --db-user postgres --db-password password --input-file data/test_transactions.csv
  ```

### Steps to Run via Docker
1. Build the Docker image:
```
docker build -t fraud-detection .
```

2. Run the container:
- For Training:
  ```
  docker run --rm
  -e DB_HOST=host.docker.internal
  -e DB_NAME=fraud_detection
  -e DB_USER=postgres
  -e DB_PASSWORD=password
  fraud-detection --mode train
  ```
- For Predictions:
  ```
  docker run --rm
  -e DB_HOST=host.docker.internal
  -e DB_NAME=fraud_detection
  -e DB_USER=postgres
  -e DB_PASSWORD=password
  fraud-detection --mode predict --input-file /app/data/test_transactions.csv
  ```

## Future Enhancements
- Explore additional algorithms for improved performance.
- Add real-time prediction capabilities.
- Integrate advanced database queries for feature engineering.

## License
This project is licensed under the LICENSE License. See [LICENSE](LICENSE) for more details.
