FROM python:3.11-slim

# Workdir inside container
WORKDIR /app

# Install system deps (optional but safe for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pandas \
    numpy \
    scikit-learn \
    joblib

# Copy only the app code (data & models will be mounted as volumes)
COPY app.py /app/app.py

# Create folders for data and models (will be overlaid by volumes)
RUN mkdir -p /app/data /app/models

EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
