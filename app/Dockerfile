FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any are needed for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dashboard requirements and install them
COPY requirements.txt ./dashboard_reqs.txt
COPY NebulaX-Hackathon-ProblemStatement/PS3/doors/requirements.txt ./door_reqs.txt
COPY NebulaX-Hackathon-ProblemStatement/PS3/shm/requirements.txt ./shm_reqs.txt

# Install all requirements including xgboost and scikit-learn
RUN pip install --no-cache-dir -r dashboard_reqs.txt -r door_reqs.txt -r shm_reqs.txt xgboost scikit-learn

# Copy the entire workspace
COPY . /app/

# Set working directory to the dashboard so uvicorn finds app.py
WORKDIR /app

# Expose the port Cloud Run uses
EXPOSE 8080

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
