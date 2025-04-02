#!/bin/bash

cd src

# Perform initialization tasks
python import_to_db.py --data-dir /app/data/

# Train and verify the ML model
python fraud_detection_ml.py --model-type randomforest --model-path /app/data/model --eda-visuals-path /app/data/eda_visuals --model-visuals-path /app/data/model_visuals

# Keep the container running
exec sleep infinity
