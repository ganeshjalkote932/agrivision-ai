"""
Disease detection inference pipeline for hyperspectral crop data.

This module implements the DiseaseDetector class for real-time predictions
with preprocessing consistency and deployment-ready inference capabilities.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import warnings

from .base import BaseProcessor
from .model_exporter import ModelExporter


class DiseaseDetector(BaseProcessor):
    """
    Real-time disease detection system for hyperspectral crop data.
    
    Provides consistent preprocessing and inference pipeline for deployment,
    ensuring reproducible predictions on new hyperspectral samples.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 preprocessor_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 model_name: Optional[str] = None,
                 model_exporter: Optional[ModelExporter] = None):
        """
        Initialize the disease detector.
        
        Args:
            model_path: Path to trained model file
            preprocessor_path: Path to preprocessor file
            metadata_path: Path to model metadata file
            model_name: Name of model to load from exporter
            model_exporter: ModelExporter instance for loading models
        """
        super().__init__()
        
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.wavelengths = None
        self.feature_names = None
        self.is_ready = False
        
        # Load model components
        if model_name and model_exporter:
            self._load_from_exporter(model_name, model_exporter)
        elif model_path:
            self._load_from_paths(model_path, preprocessor_path, metadata_path)
        else:
            self.logger.warning("No model specified. Use load_model() to initialize.")
    
    def _load_from_exporter(self, model_name: str, model_exporter: ModelExporter):
        """Load model components from ModelExporter."""
        try:
            model_data = model_exporter.load_model_with_preprocessing(model_name)
            
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.metadata = model_data['metadata']
            
            # Extract feature information
            feature_info = self.metadata.get('feature_info', {})
            self.wavelengths = feature_info.get('wavelengths')
            self.feature_names = feature_info.get('feature_names')
            
            self.is_ready = True
            self.logger.info(f"Model '{model_name}' loaded successfully from exporter")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from exporter: {e}")
            raise 
   
    def _load_from_paths(self, model_path: str, preprocessor_path: Optional[str] = None,
                        metadata_path: Optional[str] = None):
        """Load model components from file paths."""
        try:
            # Load model
            self.model = joblib.load(model_path)
            self.logger.info(f"Model loaded from: {model_path}")
            
            # Load preprocessor if available
            if preprocessor_path and os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                self.logger.info(f"Preprocessor loaded from: {preprocessor_path}")
            
            # Load metadata if available
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Extract feature information
                feature_info = self.metadata.get('feature_info', {})
                self.wavelengths = feature_info.get('wavelengths')
                self.feature_names = feature_info.get('feature_names')
            
            self.is_ready = True
            self.logger.info("Model components loaded successfully from paths")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from paths: {e}")
            raise
    
    def load_model(self, model_path: str, preprocessor_path: Optional[str] = None,
                  metadata_path: Optional[str] = None):
        """
        Load model components from file paths.
        
        Args:
            model_path: Path to trained model file
            preprocessor_path: Path to preprocessor file
            metadata_path: Path to model metadata file
        """
        self._load_from_paths(model_path, preprocessor_path, metadata_path)
    
    def validate_input_data(self, spectral_data: np.ndarray) -> Tuple[bool, str]:
        """
        Validate input spectral data format and dimensions.
        
        Args:
            spectral_data: Input spectral data array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(spectral_data, np.ndarray):
            return False, "Input must be a numpy array"
        
        if spectral_data.ndim not in [1, 2]:
            return False, "Input must be 1D or 2D array"
        
        # Check feature dimensions
        expected_features = None
        if self.wavelengths:
            expected_features = len(self.wavelengths)
        elif hasattr(self.model, 'n_features_in_'):
            expected_features = self.model.n_features_in_
        
        if expected_features:
            actual_features = spectral_data.shape[-1]
            if actual_features != expected_features:
                return False, f"Expected {expected_features} features, got {actual_features}"
        
        # Check for valid spectral values (allow for preprocessed data)
        # Raw reflectance should be 0-1, but preprocessed data can have different ranges
        if self.preprocessor is None:
            # Only validate raw reflectance range if no preprocessing
            if np.any(spectral_data < 0) or np.any(spectral_data > 1.5):
                return False, "Raw spectral values should be in range [0, 1.5] for reflectance data"
        else:
            # For preprocessed data, just check for extreme outliers
            if np.any(np.abs(spectral_data) > 100):
                return False, "Spectral values contain extreme outliers"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(spectral_data)) or np.any(np.isinf(spectral_data)):
            return False, "Input contains NaN or infinite values"
        
        return True, "Input validation passed"
    
    def preprocess_input(self, spectral_data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to input spectral data.
        
        Args:
            spectral_data: Raw spectral data
            
        Returns:
            Preprocessed spectral data
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure correct shape
        if spectral_data.ndim == 1:
            spectral_data = spectral_data.reshape(1, -1)
        
        # Apply preprocessing if available
        if self.preprocessor:
            try:
                processed_data = self.preprocessor.transform(spectral_data)
                self.logger.debug("Preprocessing applied successfully")
                return processed_data
            except Exception as e:
                self.logger.error(f"Preprocessing failed: {e}")
                raise
        else:
            self.logger.debug("No preprocessing applied")
            return spectral_data
    
    def predict_single_sample(self, spectral_data: np.ndarray, 
                            include_probabilities: bool = True,
                            include_metadata: bool = False) -> Dict[str, Any]:
        """
        Predict disease status for a single hyperspectral sample.
        
        Args:
            spectral_data: Array of spectral values (shape: n_wavelengths,)
            include_probabilities: Whether to include prediction probabilities
            include_metadata: Whether to include prediction metadata
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        is_valid, error_msg = self.validate_input_data(spectral_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {error_msg}")
        
        # Preprocess data
        processed_data = self.preprocess_input(spectral_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'disease_detected': bool(prediction),
            'prediction_label': 'Diseased' if prediction == 1 else 'Healthy',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add probabilities if requested and available
        if include_probabilities and hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(processed_data)[0]
                result['probabilities'] = {
                    'healthy': float(probabilities[0]),
                    'diseased': float(probabilities[1])
                }
                result['confidence'] = float(max(probabilities))
            except Exception as e:
                self.logger.warning(f"Failed to get probabilities: {e}")
        
        # Add metadata if requested
        if include_metadata:
            result['metadata'] = {
                'model_type': type(self.model).__name__,
                'preprocessing_applied': self.preprocessor is not None,
                'input_shape': spectral_data.shape,
                'processed_shape': processed_data.shape,
                'model_metadata': self.metadata.get('model_name', 'unknown')
            }
        
        return result 
   
    def predict_batch(self, spectral_data: np.ndarray,
                     include_probabilities: bool = True,
                     batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Predict disease status for multiple samples.
        
        Args:
            spectral_data: Array of spectral values (shape: n_samples, n_wavelengths)
            include_probabilities: Whether to include prediction probabilities
            batch_size: Optional batch size for processing large datasets
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        is_valid, error_msg = self.validate_input_data(spectral_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {error_msg}")
        
        # Ensure 2D array
        if spectral_data.ndim == 1:
            spectral_data = spectral_data.reshape(1, -1)
        
        n_samples = spectral_data.shape[0]
        results = []
        
        # Process in batches if batch_size specified
        if batch_size and n_samples > batch_size:
            self.logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_data = spectral_data[i:end_idx]
                batch_results = self._process_batch(batch_data, include_probabilities, i)
                results.extend(batch_results)
        else:
            results = self._process_batch(spectral_data, include_probabilities, 0)
        
        return results
    
    def _process_batch(self, batch_data: np.ndarray, include_probabilities: bool,
                      start_index: int) -> List[Dict[str, Any]]:
        """Process a batch of samples."""
        # Preprocess batch
        processed_data = self.preprocess_input(batch_data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        # Get probabilities if available and requested
        probabilities = None
        if include_probabilities and hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(processed_data)
            except Exception as e:
                self.logger.warning(f"Failed to get batch probabilities: {e}")
        
        # Format results
        batch_results = []
        timestamp = datetime.now().isoformat()
        
        for i, pred in enumerate(predictions):
            result = {
                'sample_index': start_index + i,
                'prediction': int(pred),
                'disease_detected': bool(pred),
                'prediction_label': 'Diseased' if pred == 1 else 'Healthy',
                'timestamp': timestamp
            }
            
            if probabilities is not None:
                result['probabilities'] = {
                    'healthy': float(probabilities[i][0]),
                    'diseased': float(probabilities[i][1])
                }
                result['confidence'] = float(max(probabilities[i]))
            
            batch_results.append(result)
        
        return batch_results
    
    def predict_from_csv(self, csv_path: str, spectral_columns: Optional[List[str]] = None,
                        output_path: Optional[str] = None,
                        include_probabilities: bool = True) -> pd.DataFrame:
        """
        Predict disease status from CSV file with hyperspectral data.
        
        Args:
            csv_path: Path to CSV file with spectral data
            spectral_columns: List of spectral column names (if None, auto-detect X* columns)
            output_path: Optional path to save results CSV
            include_probabilities: Whether to include prediction probabilities
            
        Returns:
            DataFrame with original data and predictions
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded CSV with {len(df)} samples from: {csv_path}")
        
        # Auto-detect spectral columns if not provided
        if spectral_columns is None:
            spectral_columns = [col for col in df.columns if col.startswith('X')]
            self.logger.info(f"Auto-detected {len(spectral_columns)} spectral columns")
        
        if not spectral_columns:
            raise ValueError("No spectral columns found. Specify spectral_columns parameter.")
        
        # Extract spectral data
        spectral_data = df[spectral_columns].values
        
        # Make predictions
        predictions = self.predict_batch(spectral_data, include_probabilities)
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result['prediction'] = [p['prediction'] for p in predictions]
        df_result['disease_detected'] = [p['disease_detected'] for p in predictions]
        df_result['prediction_label'] = [p['prediction_label'] for p in predictions]
        
        if include_probabilities and 'probabilities' in predictions[0]:
            df_result['prob_healthy'] = [p['probabilities']['healthy'] for p in predictions]
            df_result['prob_diseased'] = [p['probabilities']['diseased'] for p in predictions]
            df_result['confidence'] = [p['confidence'] for p in predictions]
        
        # Save results if output path provided
        if output_path:
            df_result.to_csv(output_path, index=False)
            self.logger.info(f"Results saved to: {output_path}")
        
        return df_result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_ready:
            return {'status': 'Model not loaded'}
        
        info = {
            'status': 'Ready',
            'model_type': type(self.model).__name__,
            'has_preprocessor': self.preprocessor is not None,
            'has_probabilities': hasattr(self.model, 'predict_proba'),
            'metadata': self.metadata
        }
        
        if hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
        
        if self.wavelengths:
            info['wavelength_range'] = [min(self.wavelengths), max(self.wavelengths)]
            info['n_wavelengths'] = len(self.wavelengths)
        
        return info
    
    def process(self, data: Any) -> Any:
        """
        Process method required by BaseProcessor.
        
        Args:
            data: Input spectral data or configuration
            
        Returns:
            Prediction results
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return self.predict_single_sample(data)
            else:
                return self.predict_batch(data)
        elif isinstance(data, dict):
            if 'spectral_data' in data:
                spectral_data = data['spectral_data']
                if spectral_data.ndim == 1:
                    return self.predict_single_sample(spectral_data, **data.get('options', {}))
                else:
                    return self.predict_batch(spectral_data, **data.get('options', {}))
        
        return data