FROM python:3.11-slim

WORKDIR /app

# LightGBM needs the OpenMP runtime, which slim images do not ship
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install serving dependencies only (training extras live in requirements-dev.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the transactionguard package itself so imports work
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
