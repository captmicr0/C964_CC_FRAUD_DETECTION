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
- **Libraries**: pandas, sqlalchemy, psycopg2-binary, kaggle, tqdm, numpy, matplotlib, seaborn, scikit-learn, xgboost, imblearn, joblib
- **Database**: PostgreSQL
- **Tools**: VSCode, pgAdmin, Docker, Docker Compose, Git

## Dataset
The project uses the publicly available [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection). The dataset will be downloaded automatically using the Kaggle API via the `kaggle` Python module.

## Project Structure
```
├── data/                # Stores application data (dataset will be downloaded here)
│   ├── eda_visuals/     # Contains EDA visuals
│   ├── model_visuals/   # Contains ML model and evaluation visuals
│   ├── model/           # Contains ML model artifacts
│   └── kaggle.json      # Kaggle API key (retreived using steps below)
├── src/                 # Source code for preprocessing, modeling, and evaluation
├── docker/              # Docker-related files (e.g., Dockerfile, docker-compose.yml)
├── README.md            # Project documentation (this file)
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.13+
- PostgreSQL
- Docker
- Docker-Compose

### Steps to Set Up Kaggle API Authentication
1. Create a Kaggle account (if you don’t already have one) at [Kaggle](https://www.kaggle.com).
2. Go to your Kaggle account settings and click on "Create New Token" under the "API" section. This will download a `kaggle.json` file containing your credentials.
3. Place the `kaggle.json` file in one of these locations:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `%HOMEPATH%/.kaggle/kaggle.json`
   - Docker: place `kaggle.json` in the `data` subdirectory

### Steps to Run via Docker Compose (easiest option)
1. Clone this repository:
  ```
  git clone https://github.com/captmicr0/C964_CC_FRAUD_DETECTION.git
  cd C964_CC_FRAUD_DETECTION
  cd docker
  ```

2. Build and start all services (application and PostgreSQL) using `docker compose`:
  ```
  docker compose build
  docker compose up -d
  ```

3. Attach to the fraud-detection-app service to get an interactive terminal:
  ```
  docker compose exec fraud-detection-app bash
  ```

4. Download and import the dataset into the PostgreSQL Database:
  ```
  python src/import_to_db.py --data-dir /app/data/
  ```
  - The application will automatically download the dataset using Kaggle API.
  - It will import the dataset into the PostgreSQL database.

5. Train and evaluate the ML model:
  ```
  python src/fraud_detection_ml.py --model-type randomforest --model-path /app/data/model --eda-visuals-path /app/data/eda_visuals --model-visuals-path /app/data/model_visuals
  ```
  - It will being training the model(s) and evaluating them.
  - It will output EDA visualizations and model visualizations to the data directory
  - It will save the model artifacts in the data directory for later use (see below)

6. Run some predictions:
  ```
  python src/predict_fraud.py --model-path /app/data/model --amount 1077.69 --hour 22 --day 21 --month 6 --cc-bin 400567 --street "458 Phillips Island Apt. 768" --city "Denham Springs" --state LA --zip 70726 --city-pop 71335 --dob 1994-05-31 --gender M --job "Herbalist" --lat 30.459 --long -90.9027 --merchant "Heathcote, Yost and Kertzmann" --merch-lat 31.204974 --merch-long -90.261595 --category shopping_net
  ```
  ```
  python src/predict_fraud.py --model-path /app/data/model --amount 41.28 --hour 12 --day 21 --month 6 --cc-bin 359821 --street "9333 Valentine Point" --city "Bellmore" --state NY --zip 11710 --city-pop 34496 --dob 1970-10-21 --gender F --job "Librarian, public" --lat 40.6729 --long -73.5365 --merchant "Swaniawski, Nitzsche and Welch" --merch-lat 40.49581 --merch-long -74.196111 --category health_fitness
  ```
  ```
  python src/predict_fraud.py --model-path /app/data/model --amount 843.91 --hour 23 --day 2 --month 1 --cc-bin 461331 --street "542 Steve Curve Suite 011" --city "Collettsville" --state NC --zip 28611 --city-pop 885 --dob 1988-09-15 --gender M --job "Soil scientist" --lat 35.9946 --long -81.7266 --merchant "Ruecker Group" --merch-lat 35.985612 --merch-long -81.383306 --category misc_net
  ```

7. Optional: Start the REST API server and test it:
  ```
  python predict_fraud.py --api-server
  ```
  ```
  apt update && apt install curl # Ensure sure CURL is installed
  curl -X POST -H "Content-Type: application/json" \
  -d '{
    "amount": 843.91,
    "hour": 23,
    "day": 2,
    "month": 1,
    "cc_bin": "461331",
    "street": "542 Steve Curve Suite 011",
    "city": "Collettsville",
    "state": "NC",
    "zip": 28611,
    "city_pop": 885,
    "dob": "1988-09-15",
    "gender": "M",
    "job": "Soil scientist",
    "lat": 35.9946,
    "long": -81.7266,
    "merchant": "Ruecker Group",
    "merch_lat": 35.985612,
    "merch_long": -81.383306,
    "category": "misc_net"
  }' \
  http://10.10.0.200:8000/predict
  ```

8. Stop all services and remove the containers when done:
  ```
  docker compose down
  docker compose rm
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

4. Go to the src subdirectory in the terminal
  ```
  cd src
  ```

5. Download and import the dataset into the PostgreSQL Database:
  ```
  python import_to_db.py --db-host localhost --db-name fraud_detection --db-user postgres --db-pass password
  ```

6. Train and evaluate the ML model:
  ```
  python fraud_detection_ml.py --db-host localhost --db-name fraud_detection --db-user postgres --db-pass password --model-type randomforest
  ```

7. Run some predictions:
  ```
  python predict_fraud.py --amount 1077.69 --hour 22 --day 21 --month 6 --cc-bin 400567 --street "458 Phillips Island Apt. 768" --city "Denham Springs" --state LA --zip 70726 --city-pop 71335 --dob 1994-05-31 --gender M --job "Herbalist" --lat 30.459 --long -90.9027 --merchant "Heathcote, Yost and Kertzmann" --merch-lat 31.204974 --merch-long -90.261595 --category shopping_net
  ```
  ```
  python predict_fraud.py --amount 41.28 --hour 12 --day 21 --month 6 --cc-bin 359821 --street "9333 Valentine Point" --city "Bellmore" --state NY --zip 11710 --city-pop 34496 --dob 1970-10-21 --gender F --job "Librarian, public" --lat 40.6729 --long -73.5365 --merchant "Swaniawski, Nitzsche and Welch" --merch-lat 40.49581 --merch-long -74.196111 --category health_fitness
  ```
  ```
  python predict_fraud.py --amount 843.91 --hour 23 --day 2 --month 1 --cc-bin 461331 --street "542 Steve Curve Suite 011" --city "Collettsville" --state NC --zip 28611 --city-pop 885 --dob 1988-09-15 --gender M --job "Soil scientist" --lat 35.9946 --long -81.7266 --merchant "Ruecker Group" --merch-lat 35.985612 --merch-long -81.383306 --category misc_net
  ```

8. Optional: Start the REST API server and test it (unable to test on Windows):
  ```
  python predict_fraud.py --api-server
  ```
  ```
  curl -X POST -H "Content-Type: application/json" \
  -d '{
    "amount": 843.91,
    "hour": 23,
    "day": 2,
    "month": 1,
    "cc_bin": "461331",
    "street": "542 Steve Curve Suite 011",
    "city": "Collettsville",
    "state": "NC",
    "zip": 28611,
    "city_pop": 885,
    "dob": "1988-09-15",
    "gender": "M",
    "job": "Soil scientist",
    "lat": 35.9946,
    "long": -81.7266,
    "merchant": "Ruecker Group",
    "merch_lat": 35.985612,
    "merch_long": -81.383306,
    "category": "misc_net"
  }' \
  http://10.10.0.200:8000/predict
  ```

## Future Enhancements
- Explore additional algorithms for improved performance.
- Add real-time prediction capabilities.
- Integrate advanced database queries for feature engineering.

## License
This project is licensed under the MIT License.

Copyright (c) [2025] [Ryan V.]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
