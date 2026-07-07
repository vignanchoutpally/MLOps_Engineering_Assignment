FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker build cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Run the batch job with default arguments.
# This will default to looking for config.yaml and data.csv in /app,
# write output to metrics.json and logs to run.log, and print metrics.json to stdout.
ENTRYPOINT ["python", "run.py"]
