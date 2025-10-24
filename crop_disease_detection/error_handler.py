"""
Error handling and recovery mechanisms for hyperspectral crop disease detection.

This module implements robust error handling throughout the pipeline with
fallback strategies for common failure scenarios and recovery mechanisms.
"""

import os
import sys
import traceback
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

from .logger import ProcessLogger


class PipelineError(Exception):
    """Base exception for pipeline-specific errors."""
    pass


class DataLoadError(PipelineError):
    """Exception raised for data loading errors."""
    pass


class PreprocessingError(PipelineError):
    """Exception raised for preprocessing errors."""
    pass


class ModelTrainingError(PipelineError):
    """Exception raised for model training errors."""
    pass


class ModelEvaluationError(PipelineError):
    """Exception raised for model evaluation errors."""
    pass


class DeploymentError(PipelineError):
    """Exception raised for deployment errors."""
    pass


class ErrorHandler:
    """
    Comprehensive error handling and recovery system.
    
    Provides robust error handling throughout the pipeline with fallback
    strategies, recovery mechanisms, and detailed error reporting.
    """
    
    def __init__(self, logger: Optional[ProcessLogger] = None,
                 enable_recovery: bool = True,
                 max_retries: int = 3,
                 recovery_strategies: Optional[Dict[str, str]] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: ProcessLogger instance for error logging
            enable_recovery: Enable automatic recovery mechanisms
            max_retries: Maximum number of retry attempts
            recovery_strategies: Custom recovery strategies
        """
        self.logger = logger
        self.enable_recovery = enable_recovery
        self.max_retries = max_retries
        
        # Default recovery strategies
        self.recovery_strategies = {
            'data_loading': 'fallback_dataset',
            'preprocessing': 'simplified_preprocessing',
            'model_training': 'fallback_model',
            'model_evaluation': 'basic_evaluation',
            'feature_analysis': 'skip_optional',
            'deployment': 'basic_export'
        }
        
        if recovery_strategies:
            self.recovery_strategies.update(recovery_strategies)
        
        # Error tracking
        self.error_history = []
        self.recovery_attempts = {}
        self.fallback_used = {}
        
        # Fallback configurations
        self.fallback_configs = self._initialize_fallback_configs()
    
    def _initialize_fallback_configs(self) -> Dict[str, Any]:
        """Initialize fallback configurations for different components."""
        return {
            'data_loading': {
                'sample_size': 1000,  # Use smaller sample if full dataset fails
                'required_columns': ['stage'],  # Minimum required columns
                'fallback_file': None  # Path to fallback dataset
            },
            'preprocessing': {
                'simple_scaling': True,  # Use simple min-max scaling
                'skip_outlier_detection': True,  # Skip complex outlier detection
                'handle_missing_simple': True  # Use simple mean imputation
            },
            'model_training': {
                'algorithms': ['random_forest'],  # Use only reliable algorithms
                'reduced_complexity': True,  # Reduce model complexity
                'skip_hyperparameter_tuning': True  # Skip time-consuming tuning
            },
            'model_evaluation': {
                'basic_metrics_only': True,  # Calculate only basic metrics
                'skip_visualizations': True,  # Skip complex visualizations
                'reduced_validation': True  # Use smaller validation set
            },
            'feature_analysis': {
                'skip_permutation_importance': True,  # Skip expensive calculations
                'reduced_feature_count': 10,  # Analyze fewer features
                'skip_biological_validation': False  # Keep biological validation
            },
            'deployment': {
                'basic_export_only': True,  # Export model without extras
                'skip_deployment_package': True,  # Skip deployment package creation
                'minimal_metadata': True  # Include minimal metadata only
            }
        }
    
    def handle_error(self, error: Exception, context: str, 
                    recovery_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            recovery_data: Additional data for recovery
            
        Returns:
            Dictionary with error handling results
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': time.time(),
            'traceback': traceback.format_exc(),
            'recovery_attempted': False,
            'recovery_successful': False,
            'fallback_used': None
        }
        
        # Log the error
        if self.logger:
            self.logger.logger.error(f"Error in {context}: {str(error)}")
            self.logger.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Add to error history
        self.error_history.append(error_info)
        
        # Attempt recovery if enabled
        if self.enable_recovery and context in self.recovery_strategies:
            try:
                recovery_result = self._attempt_recovery(error, context, recovery_data)
                error_info.update(recovery_result)
                
            except Exception as recovery_error:
                if self.logger:
                    self.logger.logger.error(f"Recovery failed for {context}: {str(recovery_error)}")
                
                error_info['recovery_error'] = str(recovery_error)
        
        return error_info
    
    def _attempt_recovery(self, error: Exception, context: str,
                         recovery_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Attempt recovery based on context and error type."""
        recovery_strategy = self.recovery_strategies.get(context)
        
        if not recovery_strategy:
            return {'recovery_attempted': False, 'reason': 'No recovery strategy defined'}
        
        # Track recovery attempts
        if context not in self.recovery_attempts:
            self.recovery_attempts[context] = 0
        
        self.recovery_attempts[context] += 1
        
        if self.recovery_attempts[context] > self.max_retries:
            return {
                'recovery_attempted': False,
                'reason': f'Max retries ({self.max_retries}) exceeded'
            }
        
        if self.logger:
            self.logger.logger.info(f"Attempting recovery for {context} using strategy: {recovery_strategy}")
        
        # Execute recovery strategy
        recovery_methods = {
            'fallback_dataset': self._recover_data_loading,
            'simplified_preprocessing': self._recover_preprocessing,
            'fallback_model': self._recover_model_training,
            'basic_evaluation': self._recover_model_evaluation,
            'skip_optional': self._recover_feature_analysis,
            'basic_export': self._recover_deployment
        }
        
        recovery_method = recovery_methods.get(recovery_strategy)
        if recovery_method:
            try:
                result = recovery_method(error, recovery_data or {})
                
                self.fallback_used[context] = recovery_strategy
                
                if self.logger:
                    self.logger.logger.info(f"Recovery successful for {context}")
                
                return {
                    'recovery_attempted': True,
                    'recovery_successful': True,
                    'fallback_used': recovery_strategy,
                    'recovery_result': result
                }
                
            except Exception as recovery_error:
                if self.logger:
                    self.logger.logger.error(f"Recovery method failed: {str(recovery_error)}")
                
                return {
                    'recovery_attempted': True,
                    'recovery_successful': False,
                    'recovery_error': str(recovery_error)
                }
        
        return {
            'recovery_attempted': False,
            'reason': f'No recovery method for strategy: {recovery_strategy}'
        }
    
    def _recover_data_loading(self, error: Exception, recovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from data loading errors."""
        config = self.fallback_configs['data_loading']
        
        # Try fallback dataset if available
        if config.get('fallback_file') and os.path.exists(config['fallback_file']):
            return {'fallback_dataset': config['fallback_file']}
        
        # Try to create synthetic data as last resort
        if 'create_synthetic' in recovery_data and recovery_data['create_synthetic']:
            synthetic_data = self._create_synthetic_dataset(config['sample_size'])
            return {'synthetic_data': synthetic_data}
        
        # Suggest manual intervention
        return {
            'suggestion': 'Check dataset file path and format',
            'required_action': 'Provide valid dataset file'
        }
    
    def _recover_preprocessing(self, error: Exception, recovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from preprocessing errors."""
        config = self.fallback_configs['preprocessing']
        
        return {
            'simplified_preprocessing': True,
            'use_simple_scaling': config['simple_scaling'],
            'skip_outlier_detection': config['skip_outlier_detection'],
            'handle_missing_simple': config['handle_missing_simple']
        }
    
    def _recover_model_training(self, error: Exception, recovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from model training errors."""
        config = self.fallback_configs['model_training']
        
        return {
            'fallback_algorithms': config['algorithms'],
            'reduced_complexity': config['reduced_complexity'],
            'skip_hyperparameter_tuning': config['skip_hyperparameter_tuning']
        }
    
    def _recover_model_evaluation(self, error: Exception, recovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from model evaluation errors."""
        config = self.fallback_configs['model_evaluation']
        
        return {
            'basic_metrics_only': config['basic_metrics_only'],
            'skip_visualizations': config['skip_visualizations'],
            'reduced_validation': config['reduced_validation']
        }
    
    def _recover_feature_analysis(self, error: Exception, recovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from feature analysis errors."""
        config = self.fallback_configs['feature_analysis']
        
        return {
            'skip_permutation_importance': config['skip_permutation_importance'],
            'reduced_feature_count': config['reduced_feature_count'],
            'skip_biological_validation': config['skip_biological_validation']
        }
    
    def _recover_deployment(self, error: Exception, recovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from deployment errors."""
        config = self.fallback_configs['deployment']
        
        return {
            'basic_export_only': config['basic_export_only'],
            'skip_deployment_package': config['skip_deployment_package'],
            'minimal_metadata': config['minimal_metadata']
        }
    
    def _create_synthetic_dataset(self, n_samples: int) -> pd.DataFrame:
        """Create synthetic dataset as fallback."""
        np.random.seed(42)
        
        # Create synthetic spectral data
        n_wavelengths = 131
        wavelengths = np.linspace(437, 2345, n_wavelengths)
        
        # Generate realistic spectral curves
        data = {}
        
        # Add spectral columns
        for i, wl in enumerate(wavelengths):
            # Simulate realistic reflectance values
            base_reflectance = 0.3 + 0.4 * np.random.random(n_samples)
            noise = 0.05 * np.random.randn(n_samples)
            data[f'X{wl:.0f}'] = np.clip(base_reflectance + noise, 0, 1)
        
        # Add metadata columns
        data['unique_id'] = range(n_samples)
        data['country'] = np.random.choice(['USA', 'Canada', 'Mexico'], n_samples)
        data['crop'] = np.random.choice(['corn', 'soybean', 'wheat'], n_samples)
        data['stage'] = np.random.choice(['Early_Mid', 'Late', 'Critical', 'Mature_Senesc'], n_samples)
        data['lat'] = 40 + 10 * np.random.random(n_samples)
        data['long'] = -100 + 20 * np.random.random(n_samples)
        data['year'] = np.random.choice([2007, 2008, 2009], n_samples)
        data['month'] = np.random.choice(range(4, 11), n_samples)  # Growing season
        
        df = pd.DataFrame(data)
        
        if self.logger:
            self.logger.logger.info(f"Created synthetic dataset with {n_samples} samples")
        
        return df
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors and recovery attempts."""
        summary = {
            'total_errors': len(self.error_history),
            'errors_by_context': {},
            'recovery_attempts': self.recovery_attempts.copy(),
            'fallbacks_used': self.fallback_used.copy(),
            'recent_errors': self.error_history[-5:] if self.error_history else []
        }
        
        # Count errors by context
        for error in self.error_history:
            context = error['context']
            if context not in summary['errors_by_context']:
                summary['errors_by_context'][context] = 0
            summary['errors_by_context'][context] += 1
        
        return summary
    
    def save_error_log(self, filepath: str):
        """Save error log to file."""
        error_log = {
            'error_summary': self.get_error_summary(),
            'detailed_errors': self.error_history,
            'fallback_configs': self.fallback_configs,
            'recovery_strategies': self.recovery_strategies
        }
        
        with open(filepath, 'w') as f:
            json.dump(error_log, f, indent=2, default=str)
        
        if self.logger:
            self.logger.log_file_operation("save_error_log", filepath, True)


def with_error_handling(context: str, error_handler: Optional[ErrorHandler] = None,
                       recovery_data: Optional[Dict[str, Any]] = None):
    """
    Decorator for adding error handling to functions.
    
    Args:
        context: Context identifier for error handling
        error_handler: ErrorHandler instance
        recovery_data: Additional data for recovery
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    error_info = error_handler.handle_error(e, context, recovery_data)
                    
                    # If recovery was successful, return recovery result
                    if error_info.get('recovery_successful'):
                        return error_info.get('recovery_result')
                
                # Re-raise the original exception if no recovery or recovery failed
                raise
        
        return wrapper
    return decorator


def validate_pipeline_requirements() -> Dict[str, Any]:
    """
    Validate system requirements for pipeline execution.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'python_version': True,
        'required_packages': True,
        'memory_available': True,
        'disk_space': True,
        'warnings': [],
        'errors': []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        validation_results['python_version'] = False
        validation_results['errors'].append(f"Python 3.8+ required, found {sys.version}")
    
    # Check required packages
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        validation_results['required_packages'] = False
        validation_results['errors'].append(f"Missing packages: {', '.join(missing_packages)}")
    
    # Check available memory (approximate)
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb < 2:
            validation_results['memory_available'] = False
            validation_results['errors'].append(f"Insufficient memory: {available_memory_gb:.1f}GB available, 2GB+ recommended")
        elif available_memory_gb < 4:
            validation_results['warnings'].append(f"Limited memory: {available_memory_gb:.1f}GB available, 4GB+ recommended for optimal performance")
    
    except ImportError:
        validation_results['warnings'].append("Cannot check memory usage (psutil not available)")
    
    # Check disk space
    try:
        disk_usage = os.statvfs('.')
        available_space_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
        
        if available_space_gb < 1:
            validation_results['disk_space'] = False
            validation_results['errors'].append(f"Insufficient disk space: {available_space_gb:.1f}GB available, 1GB+ required")
    
    except (AttributeError, OSError):
        # Windows or other systems without statvfs
        validation_results['warnings'].append("Cannot check disk space on this system")
    
    return validation_results


def create_recovery_config(output_path: str = "recovery_config.json"):
    """Create a recovery configuration file with default settings."""
    recovery_config = {
        "error_handling": {
            "enable_recovery": True,
            "max_retries": 3,
            "log_errors": True
        },
        "fallback_strategies": {
            "data_loading": "fallback_dataset",
            "preprocessing": "simplified_preprocessing",
            "model_training": "fallback_model",
            "model_evaluation": "basic_evaluation",
            "feature_analysis": "skip_optional",
            "deployment": "basic_export"
        },
        "fallback_configs": {
            "data_loading": {
                "sample_size": 1000,
                "create_synthetic_if_needed": True
            },
            "preprocessing": {
                "use_simple_methods": True,
                "skip_complex_operations": True
            },
            "model_training": {
                "use_reliable_algorithms_only": True,
                "reduce_complexity": True
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(recovery_config, f, indent=2)
    
    print(f"Recovery configuration created: {output_path}")
    return output_path