# MLOps Batch Processing Job

A robust, production-grade Python batch processing job designed to ingest financial OHLCV market data, compute rolling metrics and signal thresholds, record runtime metrics, and log detailed execution checkpoints. The project is fully containerized and designed for MLOps pipeline deployments.

---

## Folder Structure

```text
project/
├── run.py             # Main pipeline orchestrator & processing script
├── config.yaml        # Pipeline configuration (seed, window, version)
├── requirements.txt   # Pinned library dependencies
├── Dockerfile         # python:3.9-slim multi-stage execution setup
├── README.md          # Comprehensive documentation (this file)
├── .gitignore         # Prevents source control pollution
├── data.csv           # Input OHLCV market data (auto-detects quote-wrapped rows)
├── metrics.json       # Generated pipeline run metrics (success/error JSON)
└── run.log            # Execution log file with tracebacks
```

---

## Requirements

- Python 3.9 or higher
- `pandas>=1.3.0`
- `numpy>=1.20.0`
- `pyyaml>=5.4`

---

## Virtual Environment Setup & Installation

To run this application locally, set up a Python virtual environment:

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (cmd):
# venv\Scripts\activate.bat
# On Windows (PowerShell):
# venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Running Locally

Execute the pipeline using the default files:

```bash
python run.py
```

Alternatively, specify custom file paths via command line arguments:

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

---

## Running with Docker

This application can be built and run as a standalone Docker container.

### Build the Image

Run the following command from the `project/` directory:

```bash
docker build -t mlops-batch-job .
```

### Run the Container

Running the container will execute the batch processing job with the default arguments. The execution prints the generated `metrics.json` schema to `stdout` and writes both `metrics.json` and `run.log` inside the container's `/app` workspace:

```bash
docker run --rm mlops-batch-job
```

### Run with Custom Data (Volume Mounting)

To run the container against local custom inputs and output the results back to your host machine:

```bash
docker run --rm \
  -v $(pwd)/my_data.csv:/app/data.csv \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  -v $(pwd)/output:/app \
  mlops-batch-job
```

---

## Example Outputs

### Success Case

**`metrics.json`**
```json
{
    "version": "v1",
    "rows_processed": 10000,
    "metric": "signal_rate",
    "value": 0.4989,
    "latency_ms": 36,
    "seed": 42,
    "status": "success"
}
```

**`run.log`**
```text
2026-07-07 10:41:36,481 - INFO - Job started
2026-07-07 10:41:36,483 - INFO - Configuration loaded
2026-07-07 10:41:36,483 - INFO - Validation passed
2026-07-07 10:41:36,500 - WARNING - Detected quote-wrapped CSV columns. Strip outer quotes and re-parse.
2026-07-07 10:41:36,515 - INFO - Rows loaded: 10000
2026-07-07 10:41:36,516 - INFO - Rolling mean started
2026-07-07 10:41:36,516 - INFO - Signal generation started
2026-07-07 10:41:36,517 - INFO - Output written to metrics.json
2026-07-07 10:41:36,517 - INFO - Metrics summary: {'version': 'v1', 'rows_processed': 10000, 'metric': 'signal_rate', 'value': 0.4989, 'latency_ms': 36, 'seed': 42, 'status': 'success'}
2026-07-07 10:41:36,517 - INFO - Elapsed runtime: 0.0370 seconds
2026-07-07 10:41:36,517 - INFO - Job completed
```

### Error Case

If input data or configurations are corrupt or invalid, the pipeline writes an error descriptor to `metrics.json` and exits with code `1`:

**`metrics.json`**
```json
{
    "version": "v1",
    "status": "error",
    "error_message": "Missing 'close' column in dataset"
}
```

---

## Design Decisions

### 1. Robust and Custom Data Parsing
* **Quote-wrapped CSV Detection**: In typical spreadsheet exports, a CSV might be output with outer quotes enclosing the entire row, making standard libraries read the row as a single string. `run.py` detects this format (exactly one column containing commas) and performs automatic regex-free line sanitization to extract columns correctly.
* **Typing & Validation Separations**: Configurations are validated against strict data types and constraints (e.g. `seed >= 0`, `window > 0`) before any execution takes place.
* **Strict Numeric Conversions**: The CSV `close` column is explicitly converted to numeric types. Any non-numeric value or NaN in this column immediately raises a processing validation error.

### 2. High-Precision Latency Measurement
* Latencies are computed using Python's high-precision `time.perf_counter()` to obtain sub-millisecond accuracy in performance tracking.

### 3. Graceful Error Fallbacks
* If configuration validation fails before the version string is parsed, the script automatically falls back to version `"v1"` for compiling the output error JSON, ensuring valid schema generation under all failure conditions.

### 4. Deterministic Execution
* The random number generator seed is configured globally via `numpy.random.seed(seed)` using the validated value from `config.yaml`, ensuring that downstream stochastic operations remain entirely reproducible.
