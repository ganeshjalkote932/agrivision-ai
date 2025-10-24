"""
Enhanced logging system for hyperspectral crop disease detection.

This module implements comprehensive logging with timestamps, structured output,
and detailed process tracking for all pipeline components.
"""

import logging
import os
import json
import datetime
from typing import Dict, List, Any, Optional, Union
import sys
from pathlib import Path
import traceback
from contextlib import contextmanager

from .config import DataConfig


class ProcessLogger:
    """
    Enhanced logger for tracking all processing steps with detailed information.
    
    Provides structured logging with timestamps, process tracking, and
    comprehensive error reporting for the entire pipeline.
    """
    
    def __init__(self, name: str = "crop_disease_detection", 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True):
        """
        Initialize the process logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_json: Enable structured JSON logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Process tracking
        self.process_start_time = datetime.datetime.now()
        self.process_steps = []
        self.current_step = None
        self.step_counter = 0
        
        # Setup handlers
        if enable_console:
            self._setup_console_handler()
        
        if enable_file:
            self._setup_file_handler()
        
        if enable_json:
            self._setup_json_handler()
        
        # Log session start
        self.log_process_start()
    
    def _setup_console_handler(self):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup file logging handler."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file_path = log_file
    
    def _setup_json_handler(self):
        """Setup JSON structured logging handler."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.log_dir / f"{self.name}_{timestamp}.json"
        
        self.json_handler = JsonFileHandler(json_file)
        self.logger.addHandler(self.json_handler)
        
        self.json_log_path = json_file
    
    def log_process_start(self):
        """Log the start of the processing session."""
        session_info = {
            'session_id': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': self.process_start_time.isoformat(),
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'log_directory': str(self.log_dir)
        }
        
        self.logger.info("="*60)
        self.logger.info("HYPERSPECTRAL CROP DISEASE DETECTION - PROCESSING SESSION STARTED")
        self.logger.info("="*60)
        
        for key, value in session_info.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("="*60)
    
    @contextmanager
    def log_step(self, step_name: str, description: str = "", **kwargs):
        """
        Context manager for logging processing steps with timing.
        
        Args:
            step_name: Name of the processing step
            description: Detailed description of the step
            **kwargs: Additional metadata for the step
        """
        self.step_counter += 1
        step_id = f"STEP_{self.step_counter:02d}"
        
        step_info = {
            'step_id': step_id,
            'step_name': step_name,
            'description': description,
            'start_time': datetime.datetime.now().isoformat(),
            'metadata': kwargs
        }
        
        self.current_step = step_info
        
        self.logger.info(f"\n{'-'*50}")
        self.logger.info(f"{step_id}: {step_name}")
        if description:
            self.logger.info(f"Description: {description}")
        
        for key, value in kwargs.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info(f"Started at: {step_info['start_time']}")
        self.logger.info(f"{'-'*50}")
        
        start_time = datetime.datetime.now()
        
        try:
            yield self
            
            # Step completed successfully
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            step_info.update({
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'status': 'SUCCESS'
            })
            
            self.logger.info(f"{step_id} COMPLETED SUCCESSFULLY")
            self.logger.info(f"Duration: {duration:.2f} seconds")
            
        except Exception as e:
            # Step failed
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            step_info.update({
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            self.logger.error(f"{step_id} FAILED")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"Duration: {duration:.2f} seconds")
            
            raise
        
        finally:
            self.process_steps.append(step_info)
            self.current_step = None
    
    def log_data_info(self, data_name: str, data: Any, **metadata):
        """
        Log information about data objects (DataFrames, arrays, etc.).
        
        Args:
            data_name: Name/description of the data
            data: The data object
            **metadata: Additional metadata
        """
        info = {'data_name': data_name}
        
        # Extract data information based on type
        if hasattr(data, 'shape'):
            info['shape'] = data.shape
            info['dtype'] = str(data.dtype) if hasattr(data, 'dtype') else 'unknown'
            
        if hasattr(data, '__len__'):
            info['length'] = len(data)
            
        if hasattr(data, 'columns'):  # pandas DataFrame
            info['columns'] = list(data.columns)
            info['memory_usage'] = data.memory_usage(deep=True).sum()
            
        if hasattr(data, 'describe'):  # pandas DataFrame/Series
            try:
                desc = data.describe()
                info['statistics'] = desc.to_dict() if hasattr(desc, 'to_dict') else str(desc)
            except:
                pass
        
        # Add custom metadata
        info.update(metadata)
        
        self.logger.info(f"Data Info - {data_name}:")
        for key, value in info.items():
            if key != 'data_name':
                self.logger.info(f"  {key}: {value}")
    
    def log_model_info(self, model_name: str, model: Any, performance_metrics: Optional[Dict] = None):
        """
        Log information about trained models.
        
        Args:
            model_name: Name of the model
            model: The model object
            performance_metrics: Performance metrics dictionary
        """
        info = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'model_module': type(model).__module__
        }
        
        # Extract model-specific information
        if hasattr(model, 'get_params'):
            info['parameters'] = model.get_params()
            
        if hasattr(model, 'feature_importances_'):
            info['has_feature_importance'] = True
            info['n_features'] = len(model.feature_importances_)
            
        if hasattr(model, 'n_features_in_'):
            info['n_features_in'] = model.n_features_in_
            
        if performance_metrics:
            info['performance_metrics'] = performance_metrics
        
        self.logger.info(f"Model Info - {model_name}:")
        for key, value in info.items():
            if key != 'model_name':
                if isinstance(value, dict) and len(str(value)) > 200:
                    self.logger.info(f"  {key}: <large_dict_with_{len(value)}_items>")
                else:
                    self.logger.info(f"  {key}: {value}")
    
    def log_performance_metrics(self, metrics: Dict[str, float], context: str = ""):
        """
        Log performance metrics in a structured format.
        
        Args:
            metrics: Dictionary of metric_name -> value
            context: Context description for the metrics
        """
        context_str = f" - {context}" if context else ""
        self.logger.info(f"Performance Metrics{context_str}:")
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric_name}: {value:.6f}")
            else:
                self.logger.info(f"  {metric_name}: {value}")
    
    def log_file_operation(self, operation: str, file_path: str, success: bool = True, **metadata):
        """
        Log file operations (save, load, etc.).
        
        Args:
            operation: Type of operation (save, load, delete, etc.)
            file_path: Path to the file
            success: Whether the operation was successful
            **metadata: Additional metadata
        """
        status = "SUCCESS" if success else "FAILED"
        file_size = "unknown"
        
        if success and os.path.exists(file_path):
            try:
                file_size = f"{os.path.getsize(file_path)} bytes"
            except:
                pass
        
        self.logger.info(f"File Operation - {operation.upper()}: {status}")
        self.logger.info(f"  File: {file_path}")
        self.logger.info(f"  Size: {file_size}")
        
        for key, value in metadata.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_process_summary(self):
        """Log a summary of the entire processing session."""
        end_time = datetime.datetime.now()
        total_duration = (end_time - self.process_start_time).total_seconds()
        
        successful_steps = sum(1 for step in self.process_steps if step.get('status') == 'SUCCESS')
        failed_steps = sum(1 for step in self.process_steps if step.get('status') == 'FAILED')
        
        self.logger.info("\n" + "="*60)
        self.logger.info("PROCESSING SESSION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Session Duration: {total_duration:.2f} seconds")
        self.logger.info(f"Total Steps: {len(self.process_steps)}")
        self.logger.info(f"Successful Steps: {successful_steps}")
        self.logger.info(f"Failed Steps: {failed_steps}")
        
        if self.process_steps:
            self.logger.info("\nStep Summary:")
            for step in self.process_steps:
                status_icon = "✓" if step.get('status') == 'SUCCESS' else "✗"
                duration = step.get('duration_seconds', 0)
                self.logger.info(f"  {status_icon} {step['step_id']}: {step['step_name']} ({duration:.2f}s)")
        
        self.logger.info("="*60)
        
        # Save process summary to JSON
        summary = {
            'session_summary': {
                'start_time': self.process_start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'total_steps': len(self.process_steps),
                'successful_steps': successful_steps,
                'failed_steps': failed_steps
            },
            'steps': self.process_steps
        }
        
        summary_file = self.log_dir / f"session_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"Session summary saved to: {summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save session summary: {e}")
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger


class JsonFileHandler(logging.Handler):
    """Custom logging handler for structured JSON output."""
    
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        
    def emit(self, record):
        """Emit a log record as JSON."""
        try:
            log_entry = {
                'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage()
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.format(record)
            
            # Write to file
            with open(self.filename, 'a') as f:
                json.dump(log_entry, f, default=str)
                f.write('\n')
                
        except Exception:
            self.handleError(record)