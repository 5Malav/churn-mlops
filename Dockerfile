# Start from a small, official Python image
FROM python:3.11-slim

# Set the working directory inside the image
# WORKDIR /app
# Install pip-tools (we use pip-sync for reproducible installs)
# RUN pip install --no-cache-dir pip-tools

# Set the working directory inside the image
WORKDIR /app

# Install system libraries LightGBM needs at runtime (libgomp = OpenMP).
# The slim base image omits these, so we add them explicitly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip-tools (we use pip-sync for reproducible installs)
RUN pip install --no-cache-dir pip-tools

# Copy ONLY requirements first (so this layer caches)
# COPY requirements.txt .
# Copy ONLY the lean API requirements first (so this layer caches)
COPY requirements-api.txt .

# Install the locked dependencies
# RUN pip-sync requirements.txt
# Install the lean runtime dependencies only
RUN pip-sync requirements-api.txt

# Now copy the source code (changes more often than deps)
COPY src/ ./src/

# Copy the trained model
COPY models/churn_model.txt ./models/churn_model.txt

# Document which port the API listens on
EXPOSE 8000

# Start the API when the container runs
CMD ["uvicorn", "src.churn_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
