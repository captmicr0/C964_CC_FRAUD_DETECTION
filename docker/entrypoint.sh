#!/bin/bash

# Perform initialization tasks
python src/import_to_db.py --db-host $DB_HOST --db-name $DB_NAME --db-user $DB_USER --db-password $DB_PASSWORD

# Keep the container running
exec sleep infinity
