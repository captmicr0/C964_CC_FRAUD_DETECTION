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
- Automated dataset download using Kaggle API

## Technologies Used
- **Programming Language**: Python
- **Libraries**: TBD (potentially: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Imbalanced-learn, SQLAlchemy, Psycopg2, Kagglehub)
- **Database**: PostgreSQL
- **Tools**: VSCode, pgAdmin, Docker, Docker-Compose, Git

## Dataset
The project uses the publicly available [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection). The dataset will be downloaded automatically using the Kaggle API via the `kagglehub` Python module.

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
- Docker-Compose

### Steps to Set Up Kaggle API Authentication
1. Create a Kaggle account (if you don’t already have one) at [Kaggle](https://www.kaggle.com).
2. Go to your Kaggle account settings and click on "Create New Token" under the "API" section. This will download a `kaggle.json` file containing your credentials.
3. Place the `kaggle.json` file in one of these locations:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `%HOMEPATH%/.kaggle/kaggle.json`
   - Docker: do one of the following
     - set the environtment variables `KAGGLE_USER` and `KAGGLE_KEY` with the information from `kaggle.json`
     - bind-mount `kaggle.json` to `~/.kaggle/kaggle.json` using `-v kaggle.json:~/.kaggle/kaggle.json`

### Steps to Run via Docker Compose (easiest option)
1. Build and start all services (application and PostgreSQL) using `docker-compose`:
```
docker-compose up --build
```

2. Once all services are running:
- The application container will automatically download the dataset using Kaggle API.
- It will import the dataset into the PostgreSQL database.

3. To train the model or make predictions:
- Open a terminal session in the application container:
  ```
  docker exec -it fraud-detection-app bash
  ```
- Run training:
  ```
  python src/main.py --db-host fraud-detection-db --db-name fraud_detection --db-user postgres --db-pass password --mode train
  ```
- Run predictions:
  - Database table as input
    ```
    python src/main.py --db-host localhost --db-name fraud_detection --db-user postgres --db-pass password --mode predict --input-table fraud_test
    ```
  - CSV file as input
    ```
    python src/main.py --db-host localhost --db-name fraud_detection --db-user postgres --db-pass password --mode predict --input-file ../data/fraudTest.csv
    ```

4. Stop all services when done:
```
docker-compose down
```

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

3. Start a PostgreSQL instance locally or via Docker:
```
docker run --name fraud-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=fraud_detection -p 5432:5432 -d postgres
```

4. Run the main script for training or predictions:
- For Importing Training Data:
  ```
  python src/import_to_db.py --db-host localhost --db-name fraud_detection --db-user postgres --db-pass password
  ```
- For Training:
  ```
  python src/main.py --db-host localhost --db-name fraud_detection --db-user postgres --db-password password --mode train
  ```
- For Predictions:
  - Database table as input
    ```
    python src/main.py --db-host localhost --db-name fraud_detection --db-user postgres --db-password password --mode predict --input-table fraud_test
    ```
  - CSV file as input
    ```
    python src/main.py --db-host localhost --db-name fraud_detection --db-user postgres --db-password password --mode predict --input-file ../data/fraudTest.csv
    ```

## Future Enhancements
- Explore additional algorithms for improved performance.
- Add real-time prediction capabilities.
- Integrate advanced database queries for feature engineering.

## License
This project is licensed under the LICENSE License. See [LICENSE](LICENSE) for more details.
