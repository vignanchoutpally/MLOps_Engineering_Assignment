#!/usr/bin/env python3
"""
MLOps Batch Processing Job.

This script loads configurations, validates parameters and datasets,
computes a rolling mean on financial close prices, generates trading signals,
and outputs performance metrics and logs.
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

# Central Constants
DEFAULT_INPUT_FILE = "data.csv"
DEFAULT_CONFIG_FILE = "config.yaml"
DEFAULT_OUTPUT_FILE = "metrics.json"
DEFAULT_LOG_FILE = "run.log"


def configure_logging(log_file: Optional[Path]) -> None:
    """
    Configure system-wide logging to write to both stdout and a log file.

    Args:
        log_file: Optional path to the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Define log formatting with timestamp, level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console logging (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File logging (if log_file path is provided)
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback output in case of permission issues or other file errors
            print(f"Warning: Could not configure log file {log_file} due to: {e}", file=sys.stderr)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict representing the configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If config file is empty or contains invalid YAML.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError("Configuration file is empty")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in configuration file: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that the configuration contains correct fields, types, and values.

    Args:
        config: Configuration dictionary.

    Raises:
        ValueError: If fields are missing or values are out of bounds.
        TypeError: If types of fields are incorrect.
    """
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")
        
    required_fields = ["seed", "window", "version"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: '{field}'")
            
    # Validate 'seed' type and boundaries
    seed = config["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"Invalid type for 'seed': expected int, got {type(seed).__name__}")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Invalid value for 'seed': {seed}. Must be between 0 and 2^32 - 1.")
        
    # Validate 'window' type and boundaries
    window = config["window"]
    if not isinstance(window, int) or isinstance(window, bool):
        raise TypeError(f"Invalid type for 'window': expected int, got {type(window).__name__}")
    if window <= 0:
        raise ValueError(f"Invalid value for 'window': {window}. Must be a positive integer.")
        
    # Validate 'version' type and content
    version = config["version"]
    if not isinstance(version, (str, int, float)):
        raise TypeError(f"Invalid type for 'version': expected string/number, got {type(version).__name__}")
    if isinstance(version, str) and not version.strip():
        raise ValueError("Invalid value for 'version': version string cannot be empty or whitespace.")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Detects and handles quote-wrapped columns where pandas would read the entire row
    as one single column name (e.g. '"col1,col2,col3"').

    Args:
        file_path: Path to the input CSV file.

    Returns:
        pd.DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If file is empty, corrupted, or parsing fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found at: {file_path}")
    
    # Handle empty files gracefully
    if file_path.stat().st_size == 0:
        raise ValueError("Input file is empty")
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Invalid CSV format or corrupted file: {e}")
        
    if df.empty:
        raise ValueError("Input CSV has no data rows")
        
    # Detect quote-wrapped row structure:
    # If the df has exactly 1 column and that column contains commas
    if len(df.columns) == 1 and ',' in df.columns[0]:
        logging.warning("Detected quote-wrapped CSV columns. Strip outer quotes and re-parse.")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned_lines = []
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                elif line.startswith("'") and line.endswith("'"):
                    line = line[1:-1]
                cleaned_lines.append(line)
            
            cleaned_csv = "\n".join(cleaned_lines)
            df = pd.read_csv(io.StringIO(cleaned_csv))
        except Exception as e:
            raise ValueError(f"Failed to parse quote-wrapped CSV: {e}")
            
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate that the loaded dataset contains a valid close price series.

    Args:
        df: Loaded DataFrame.

    Raises:
        ValueError: If dataset is missing the close column, or contains non-numeric
                    or NaN close values.
    """
    if "close" not in df.columns:
        raise ValueError("Missing 'close' column in dataset")
        
    # Check for NaN values in the close column
    if df["close"].isna().any():
        raise ValueError("Dataset contains NaN values in 'close' column")
        
    # Check if close column is numeric
    try:
        pd.to_numeric(df["close"], errors='raise')
    except (ValueError, TypeError) as e:
        raise ValueError(f"Non-numeric values found in 'close' column: {e}")


def compute_rolling_mean(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute rolling mean on the 'close' column with the given window size.

    First window - 1 rows will have NaN rolling mean.

    Args:
        df: Input DataFrame.
        window: Window size for moving average.

    Returns:
        pd.Series containing the rolling mean.
    """
    logging.info("Rolling mean started")
    close_series = df["close"].astype(float)
    return close_series.rolling(window=window).mean()


def generate_signal(df: pd.DataFrame, rolling_mean: pd.Series) -> pd.Series:
    """
    Generate signal = 1 if close > rolling_mean else 0.

    NaN rolling means produce signal = 0.

    Args:
        df: Input DataFrame.
        rolling_mean: Computed rolling mean series.

    Returns:
        pd.Series of the generated signals (0 or 1).
    """
    logging.info("Signal generation started")
    close_series = df["close"].astype(float)
    # Perform strict boolean element-wise comparison and cast to int.
    # Where rolling_mean is NaN, condition evaluates to False, resulting in 0.
    signal = np.where((close_series > rolling_mean) & rolling_mean.notna(), 1, 0)
    return pd.Series(signal, index=df.index)


def calculate_metrics(
    df: pd.DataFrame,
    signal: pd.Series,
    latency_ms: int,
    seed: int,
    version: str
) -> Dict[str, Any]:
    """
    Calculate summary metrics of the batch run.

    Args:
        df: Input DataFrame.
        signal: Generated signal series.
        latency_ms: Execution latency in milliseconds.
        seed: Random seed used.
        version: Version string.

    Returns:
        Dict matching the required metrics schema.
    """
    rows_processed = len(df)
    signal_rate = float(signal.mean()) if rows_processed > 0 else 0.0
    
    return {
        "version": str(version),
        "rows_processed": int(rows_processed),
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": int(latency_ms),
        "seed": int(seed),
        "status": "success"
    }


def write_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """
    Write metrics dictionary to the output JSON file.

    Args:
        metrics: Metrics to write.
        output_path: Path to the output JSON file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Output written to {output_path}")
    except Exception as e:
        logging.error(f"Failed to write metrics to {output_path}: {e}")


def main() -> None:
    """
    Main driver function for the MLOps Batch Processing pipeline.
    """
    start_time = time.perf_counter()
    
    parser = argparse.ArgumentParser(description="MLOps Batch processing job.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE, help="Path to input CSV data file")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, help="Path to YAML configuration file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Path to output JSON metrics file")
    parser.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE, help="Path to log file")
    args = parser.parse_args()
    
    # Initialize logging to file and stdout
    log_path = Path(args.log_file)
    configure_logging(log_path)
    
    logging.info("Job started")
    
    input_path = Path(args.input)
    config_path = Path(args.config)
    output_path = Path(args.output)
    
    # Fallback default variables for metric generation in case config fails
    version = "v1"
    seed = 42
    
    try:
        # Load configuration
        config = load_config(config_path)
        logging.info("Configuration loaded")
        
        # Validate configuration parameters
        validate_config(config)
        logging.info("Validation passed")
        
        version = config["version"]
        seed = config["seed"]
        window = config["window"]
        
        # Apply deterministic random seed
        np.random.seed(seed)
        
        # Load dataset and handle potential quote-wrapped format
        df = load_dataset(input_path)
        logging.info(f"Rows loaded: {len(df)}")
        
        # Validate dataset sanity
        validate_dataset(df)
        
        # Compute metrics & processing
        rolling_mean = compute_rolling_mean(df, window)
        signal = generate_signal(df, rolling_mean)
        
        # Compute performance latency
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Compile success metrics
        metrics = calculate_metrics(df, signal, latency_ms, seed, version)
        
        # Save metrics to output file
        write_metrics(metrics, output_path)
        
        # Print metrics schema directly to stdout
        print(json.dumps(metrics, indent=4))
        
        elapsed_runtime = end_time - start_time
        logging.info(f"Metrics summary: {metrics}")
        logging.info(f"Elapsed runtime: {elapsed_runtime:.4f} seconds")
        logging.info("Job completed")
        
        sys.exit(0)
        
    except Exception as e:
        # Log all exceptions with full tracebacks
        logging.exception("An error occurred during execution:")
        
        # Generate standard error metrics output
        error_metrics = {
            "version": str(version),
            "status": "error",
            "error_message": str(e)
        }
        
        # Attempt to save metrics to output file
        write_metrics(error_metrics, output_path)
        
        # Print error metrics schema directly to stdout
        print(json.dumps(error_metrics, indent=4))
        
        sys.exit(1)


if __name__ == "__main__":
    main()
