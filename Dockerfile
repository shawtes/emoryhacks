# Multi-stage build for production
FROM python:3.11-slim as backend

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY emoryhacks/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY emoryhacks/ /app/emoryhacks/
COPY emoryhacks/api/ /app/emoryhacks/api/

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "emoryhacks.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


