# Base image with Python 3.13+
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code into the container
COPY src ./src

# Set the entrypoint script
CMD ["sleep", "infinity"]
