"""
Utility functions for the crop disease detection system.
"""

import os
import json
import pickle
import joblib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Set up logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_directory_structure(base_path: str) -> Dict[str, str]:
    """
    Create the standard directory structure for the project.
    
    Args:
        base_path: Base directory path for the project
    
    Returns:
        Dictionary mapping directory names to their paths
    """
    directories = {
        'data': 'data',
        'models': 'models',
        'results': 'results',
        'plots': 'results/plots',
        'logs': 'logs',
        'reports': 'results/reports'
    }
    
    created_dirs = {}
    for name, rel_path in directories.items():
        full_path = os.path.join(base_path, rel_path)
        os.makedirs(full_path, exist_ok=True)
        created_dirs[name] = full_path
    
    return created_dirs


def save_model_artifacts(model: Any, preprocessor: Any, metadata: Dict[str, Any], 
                        model_path: str) -> None:
    """
    Save model, preprocessor, and metadata to files.
    
    Args:
        model: Trained model object
        preprocessor: Fitted preprocessor object
        metadata: Model metadata and configuration
        model_path: Base path for saving model artifacts
    """
    # Create model directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, f"{model_path}_model.joblib")
    
    # Save preprocessor
    joblib.dump(preprocessor, f"{model_path}_preprocessor.joblib")
    
    # Save metadata
    with open(f"{model_path}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_model_artifacts(model_path: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load model, preprocessor, and metadata from files.
    
    Args:
        model_path: Base path for loading model artifacts
    
    Returns:
        Tuple of (model, preprocessor, metadata)
    """
    # Load model
    model = joblib.load(f"{model_path}_model.joblib")
    
    # Load preprocessor
    preprocessor = joblib.load(f"{model_path}_preprocessor.joblib")
    
    # Load metadata
    with open(f"{model_path}_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return model, preprocessor, metadata


def get_spectral_wavelengths(df: pd.DataFrame) -> List[float]:
    """
    Extract wavelength values from column names.
    
    Args:
        df: DataFrame with spectral columns (X437, X447, etc.)
    
    Returns:
        List of wavelength values in nanometers
    """
    spectral_columns = [col for col in df.columns if col.startswith('X') and col[1:].isdigit()]
    wavelengths = [float(col[1:]) for col in spectral_columns]
    return sorted(wavelengths)


def validate_spectral_data(spectra: np.ndarray, wavelengths: List[float]) -> Dict[str, Any]:
    """
    Validate spectral data quality and characteristics.
    
    Args:
        spectra: Spectral data array (samples x wavelengths)
        wavelengths: List of wavelength values
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    # Check for negative values (invalid reflectance)
    negative_count = np.sum(spectra < 0)
    if negative_count > 0:
        validation_results['issues'].append(f"Found {negative_count} negative reflectance values")
        validation_results['valid'] = False
    
    # Check for extremely high values (>100% reflectance is suspicious)
    high_count = np.sum(spectra > 100)
    if high_count > 0:
        validation_results['issues'].append(f"Found {high_count} reflectance values >100%")
    
    # Check for missing values
    missing_count = np.sum(np.isnan(spectra))
    if missing_count > 0:
        validation_results['issues'].append(f"Found {missing_count} missing values")
        validation_results['valid'] = False
    
    # Calculate basic statistics
    validation_results['statistics'] = {
        'mean_reflectance': float(np.nanmean(spectra)),
        'std_reflectance': float(np.nanstd(spectra)),
        'min_reflectance': float(np.nanmin(spectra)),
        'max_reflectance': float(np.nanmax(spectra)),
        'missing_percentage': float(missing_count / spectra.size * 100)
    }
    
    return validation_results


def calculate_memory_usage(data_size: int, dtype: str = 'float64') -> Dict[str, float]:
    """
    Calculate estimated memory usage for data processing.
    
    Args:
        data_size: Number of elements in the data
        dtype: Data type string
    
    Returns:
        Dictionary with memory usage estimates in MB and GB
    """
    # Bytes per element for different data types
    dtype_sizes = {
        'float64': 8,
        'float32': 4,
        'int64': 8,
        'int32': 4
    }
    
    bytes_per_element = dtype_sizes.get(dtype, 8)
    total_bytes = data_size * bytes_per_element
    
    return {
        'bytes': total_bytes,
        'mb': total_bytes / (1024 * 1024),
        'gb': total_bytes / (1024 * 1024 * 1024)
    }


def generate_timestamp() -> str:
    """Generate a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    # Note: sklearn models will use their own random_state parameters