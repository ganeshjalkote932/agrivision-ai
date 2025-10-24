"""
Model export and serialization module for hyperspectral crop disease detection.

This module implements comprehensive model serialization with preprocessing parameters,
metadata tracking, and version control for deployment-ready model artifacts.
"""

import os
import json
import joblib
import pickle
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import hashlib
import platform

from .base import BaseProcessor
from .config import ModelConfig


class ModelExporter(BaseProcessor):
    """
    Comprehensive model export and serialization system.
    
    Handles saving trained models with preprocessing parameters, metadata tracking,
    version control, and deployment preparation for hyperspectral disease detection.
    """
    
    def __init__(self, output_dir: str = "models", model_config: Optional[ModelConfig] = None):
        """
        Initialize the model exporter.
        
        Args:
            output_dir: Directory to save model artifacts
            model_config: Optional ModelConfig instance
        """
        super().__init__()
        self.output_dir = output_dir
        self.model_config = model_config or ModelConfig()
        os.makedirs(output_dir, exist_ok=True)
        
        # Model registry for tracking exported models
        self.model_registry = {}
        self.registry_file = os.path.join(output_dir, "model_registry.json")
        self._load_registry()
    
    def _load_registry(self):
        """Load existing model registry if it exists."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    self.model_registry = json.load(f)
                self.logger.info(f"Loaded model registry with {len(self.model_registry)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to load model registry: {e}")
                self.model_registry = {}
    
    def _save_registry(self):
        """Save model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
            self.logger.info("Model registry saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
    
    def _generate_model_hash(self, model: BaseEstimator, preprocessor: Any = None) -> str:
        """
        Generate a unique hash for the model and preprocessor combination.
        
        Args:
            model: Trained model
            preprocessor: Preprocessing pipeline
            
        Returns:
            SHA256 hash string
        """
        # Create a string representation of model parameters
        model_str = str(model.get_params()) if hasattr(model, 'get_params') else str(model)
        
        # Add preprocessor parameters if available
        if preprocessor is not None:
            if hasattr(preprocessor, 'get_params'):
                model_str += str(preprocessor.get_params())
            else:
                model_str += str(preprocessor)
        
        # Generate hash
        return hashlib.sha256(model_str.encode()).hexdigest()[:16]
    
    def _create_model_metadata(self, model: BaseEstimator, model_name: str,
                             performance_metrics: Optional[Dict[str, float]] = None,
                             training_info: Optional[Dict[str, Any]] = None,
                             feature_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create comprehensive metadata for the model.
        
        Args:
            model: Trained model
            model_name: Name/identifier for the model
            performance_metrics: Model performance metrics
            training_info: Training configuration and parameters
            feature_info: Feature and wavelength information
            
        Returns:
            Dictionary with model metadata
        """
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'export_timestamp': datetime.datetime.now().isoformat(),
            'export_version': '1.0.0',
            'framework_info': {
                'sklearn_version': getattr(model, '__module__', 'unknown'),
                'python_version': platform.python_version(),
                'platform': platform.platform()
            },
            'model_parameters': model.get_params() if hasattr(model, 'get_params') else {},
            'model_hash': self._generate_model_hash(model),
            'performance_metrics': performance_metrics or {},
            'training_info': training_info or {},
            'feature_info': feature_info or {},
            'deployment_ready': True
        }
        
        # Add model-specific information
        if hasattr(model, 'feature_importances_'):
            metadata['has_feature_importance'] = True
            metadata['n_features'] = len(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            metadata['has_coefficients'] = True
            metadata['n_features'] = len(model.coef_[0]) if model.coef_.ndim > 1 else len(model.coef_)
        
        return metadata
    
    def save_model_with_preprocessing(self, model: BaseEstimator, preprocessor: Any,
                                    model_name: str, wavelengths: Optional[List[float]] = None,
                                    performance_metrics: Optional[Dict[str, float]] = None,
                                    training_config: Optional[Dict[str, Any]] = None,
                                    feature_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Save model with preprocessing pipeline and comprehensive metadata.
        
        Args:
            model: Trained model to save
            preprocessor: Preprocessing pipeline (scaler, etc.)
            model_name: Unique name for the model
            wavelengths: List of wavelengths corresponding to features
            performance_metrics: Model performance metrics
            training_config: Training configuration used
            feature_names: Names of input features
            
        Returns:
            Dictionary with paths to saved files
        """
        self.logger.info(f"Saving model: {model_name}")
        
        # Create model-specific directory
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Prepare file paths
        model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
        preprocessor_path = os.path.join(model_dir, f"{model_name}_preprocessor.joblib")
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        config_path = os.path.join(model_dir, f"{model_name}_config.json")
        
        # Save model
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to: {model_path}")
        
        # Save preprocessor
        if preprocessor is not None:
            joblib.dump(preprocessor, preprocessor_path)
            self.logger.info(f"Preprocessor saved to: {preprocessor_path}")
        
        # Create and save metadata
        feature_info = {
            'wavelengths': wavelengths,
            'feature_names': feature_names,
            'n_features': len(wavelengths) if wavelengths else None,
            'wavelength_range': [min(wavelengths), max(wavelengths)] if wavelengths else None
        }
        
        metadata = self._create_model_metadata(
            model, model_name, performance_metrics, training_config, feature_info
        )
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Metadata saved to: {metadata_path}")
        
        # Save training configuration
        config_data = {
            'model_config': self.model_config.__dict__ if self.model_config else {},
            'training_config': training_config or {},
            'export_settings': {
                'compression': 'joblib_default',
                'format_version': '1.0'
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        # Update model registry
        registry_entry = {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path if preprocessor else None,
            'metadata_path': metadata_path,
            'config_path': config_path,
            'export_timestamp': metadata['export_timestamp'],
            'model_hash': metadata['model_hash'],
            'performance_metrics': performance_metrics or {}
        }
        
        self.model_registry[model_name] = registry_entry
        self._save_registry()
        
        # Return file paths
        file_paths = {
            'model': model_path,
            'preprocessor': preprocessor_path if preprocessor else None,
            'metadata': metadata_path,
            'config': config_path,
            'model_dir': model_dir
        }
        
        self.logger.info(f"Model export completed: {model_name}")
        return file_paths
    
    def load_model_with_preprocessing(self, model_name: str) -> Dict[str, Any]:
        """
        Load model with preprocessing pipeline and metadata.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Dictionary containing model, preprocessor, and metadata
        """
        self.logger.info(f"Loading model: {model_name}")
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        registry_entry = self.model_registry[model_name]
        
        # Load model
        model = joblib.load(registry_entry['model_path'])
        self.logger.info(f"Model loaded from: {registry_entry['model_path']}")
        
        # Load preprocessor if available
        preprocessor = None
        if registry_entry['preprocessor_path'] and os.path.exists(registry_entry['preprocessor_path']):
            preprocessor = joblib.load(registry_entry['preprocessor_path'])
            self.logger.info(f"Preprocessor loaded from: {registry_entry['preprocessor_path']}")
        
        # Load metadata
        metadata = {}
        if os.path.exists(registry_entry['metadata_path']):
            with open(registry_entry['metadata_path'], 'r') as f:
                metadata = json.load(f)
        
        # Load configuration
        config = {}
        if os.path.exists(registry_entry['config_path']):
            with open(registry_entry['config_path'], 'r') as f:
                config = json.load(f)
        
        return {
            'model': model,
            'preprocessor': preprocessor,
            'metadata': metadata,
            'config': config,
            'registry_entry': registry_entry
        }
    
    def export_model_for_deployment(self, model_name: str, 
                                  deployment_format: str = "joblib",
                                  include_examples: bool = True) -> Dict[str, str]:
        """
        Export model in deployment-ready format with examples.
        
        Args:
            model_name: Name of the model to export
            deployment_format: Format for deployment ("joblib", "pickle", "onnx")
            include_examples: Whether to include usage examples
            
        Returns:
            Dictionary with deployment package paths
        """
        self.logger.info(f"Exporting model for deployment: {model_name}")
        
        # Load model components
        model_data = self.load_model_with_preprocessing(model_name)
        
        # Create deployment directory
        deployment_dir = os.path.join(self.output_dir, f"{model_name}_deployment")
        os.makedirs(deployment_dir, exist_ok=True)
        
        deployment_paths = {}
        
        # Export in requested format
        if deployment_format == "joblib":
            model_file = os.path.join(deployment_dir, f"{model_name}.joblib")
            joblib.dump(model_data['model'], model_file)
            deployment_paths['model'] = model_file
            
        elif deployment_format == "pickle":
            model_file = os.path.join(deployment_dir, f"{model_name}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_data['model'], f)
            deployment_paths['model'] = model_file
            
        else:
            raise ValueError(f"Unsupported deployment format: {deployment_format}")
        
        # Export preprocessor
        if model_data['preprocessor']:
            preprocessor_file = os.path.join(deployment_dir, f"{model_name}_preprocessor.joblib")
            joblib.dump(model_data['preprocessor'], preprocessor_file)
            deployment_paths['preprocessor'] = preprocessor_file
        
        # Copy metadata and config
        metadata_file = os.path.join(deployment_dir, "model_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(model_data['metadata'], f, indent=2, default=str)
        deployment_paths['metadata'] = metadata_file
        
        # Create deployment info
        deployment_info = {
            'model_name': model_name,
            'deployment_format': deployment_format,
            'deployment_timestamp': datetime.datetime.now().isoformat(),
            'files': deployment_paths,
            'usage_instructions': self._generate_usage_instructions(model_name, model_data)
        }
        
        deployment_info_file = os.path.join(deployment_dir, "deployment_info.json")
        with open(deployment_info_file, 'w') as f:
            json.dump(deployment_info, f, indent=2, default=str)
        deployment_paths['deployment_info'] = deployment_info_file
        
        # Generate example code if requested
        if include_examples:
            example_file = self._create_usage_examples(deployment_dir, model_name, model_data)
            deployment_paths['examples'] = example_file
        
        self.logger.info(f"Deployment package created: {deployment_dir}")
        return deployment_paths
    
    def _generate_usage_instructions(self, model_name: str, model_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate usage instructions for the deployed model."""
        instructions = {
            'loading': f"model = joblib.load('{model_name}.joblib')",
            'preprocessing': "preprocessor = joblib.load('{}_preprocessor.joblib')".format(model_name) if model_data['preprocessor'] else "No preprocessing required",
            'prediction': "prediction = model.predict(preprocessed_data)",
            'probability': "probabilities = model.predict_proba(preprocessed_data)" if hasattr(model_data['model'], 'predict_proba') else "Probabilities not available"
        }
        return instructions
    
    def _create_usage_examples(self, deployment_dir: str, model_name: str, 
                             model_data: Dict[str, Any]) -> str:
        """Create example usage code file."""
        example_code = f'''#!/usr/bin/env python3
"""
Usage example for {model_name} hyperspectral crop disease detection model.

This example demonstrates how to load and use the trained model for inference
on new hyperspectral data samples.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class {model_name.replace('-', '_').title()}Predictor:
    """
    Predictor class for {model_name} model.
    """
    
    def __init__(self, model_path: str, preprocessor_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the preprocessor file (optional)
        """
        self.model = joblib.load(model_path)
        self.preprocessor = None
        
        if preprocessor_path:
            self.preprocessor = joblib.load(preprocessor_path)
        
        print(f"Model loaded: {{type(self.model).__name__}}")
        if self.preprocessor:
            print(f"Preprocessor loaded: {{type(self.preprocessor).__name__}}")
    
    def predict_single_sample(self, spectral_data: np.ndarray) -> Dict[str, Any]:
        """
        Predict disease status for a single hyperspectral sample.
        
        Args:
            spectral_data: Array of spectral values (shape: n_wavelengths,)
            
        Returns:
            Dictionary with prediction results
        """
        # Ensure correct shape
        if spectral_data.ndim == 1:
            spectral_data = spectral_data.reshape(1, -1)
        
        # Apply preprocessing if available
        if self.preprocessor:
            spectral_data = self.preprocessor.transform(spectral_data)
        
        # Make prediction
        prediction = self.model.predict(spectral_data)[0]
        
        result = {{
            'prediction': int(prediction),
            'disease_detected': bool(prediction),
            'prediction_label': 'Diseased' if prediction == 1 else 'Healthy'
        }}
        
        # Add probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(spectral_data)[0]
            result['probabilities'] = {{
                'healthy': float(probabilities[0]),
                'diseased': float(probabilities[1])
            }}
            result['confidence'] = float(max(probabilities))
        
        return result
    
    def predict_batch(self, spectral_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict disease status for multiple samples.
        
        Args:
            spectral_data: Array of spectral values (shape: n_samples, n_wavelengths)
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Apply preprocessing if available
        processed_data = spectral_data
        if self.preprocessor:
            processed_data = self.preprocessor.transform(spectral_data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_data)
        
        # Format results
        for i, pred in enumerate(predictions):
            result = {{
                'sample_index': i,
                'prediction': int(pred),
                'disease_detected': bool(pred),
                'prediction_label': 'Diseased' if pred == 1 else 'Healthy'
            }}
            
            if probabilities is not None:
                result['probabilities'] = {{
                    'healthy': float(probabilities[i][0]),
                    'diseased': float(probabilities[i][1])
                }}
                result['confidence'] = float(max(probabilities[i]))
            
            results.append(result)
        
        return results


def example_usage():
    """Example usage of the predictor."""
    
    # Initialize predictor
    predictor = {model_name.replace('-', '_').title()}Predictor(
        model_path='{model_name}.joblib',
        preprocessor_path='{model_name}_preprocessor.joblib'
    )
    
    # Example 1: Single sample prediction
    print("\\n=== Single Sample Prediction ===")
    
    # Create example spectral data (131 wavelengths from 437-2345 nm)
    np.random.seed(42)
    sample_spectra = np.random.random(131) * 0.5 + 0.2  # Simulated reflectance values
    
    result = predictor.predict_single_sample(sample_spectra)
    print(f"Prediction: {{result['prediction_label']}}")
    if 'confidence' in result:
        print(f"Confidence: {{result['confidence']:.3f}}")
    
    # Example 2: Batch prediction
    print("\\n=== Batch Prediction ===")
    
    # Create batch of samples
    batch_spectra = np.random.random((5, 131)) * 0.5 + 0.2
    
    batch_results = predictor.predict_batch(batch_spectra)
    for result in batch_results:
        print(f"Sample {{result['sample_index']}}: {{result['prediction_label']}}")
    
    # Example 3: Loading from CSV file
    print("\\n=== CSV File Processing ===")
    print("# To process a CSV file with hyperspectral data:")
    print("# df = pd.read_csv('your_data.csv')")
    print("# spectral_columns = [col for col in df.columns if col.startswith('X')]")
    print("# spectral_data = df[spectral_columns].values")
    print("# results = predictor.predict_batch(spectral_data)")


if __name__ == "__main__":
    example_usage()
'''
        
        example_file = os.path.join(deployment_dir, f"{model_name}_usage_example.py")
        with open(example_file, 'w') as f:
            f.write(example_code)
        
        self.logger.info(f"Usage example created: {example_file}")
        return example_file
    
    def list_exported_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all exported models with their metadata.
        
        Returns:
            Dictionary of model_name -> model_info
        """
        return self.model_registry.copy()
    
    def delete_model(self, model_name: str, confirm: bool = False) -> bool:
        """
        Delete a model and all its associated files.
        
        Args:
            model_name: Name of the model to delete
            confirm: Confirmation flag for deletion
            
        Returns:
            True if deletion successful, False otherwise
        """
        if not confirm:
            self.logger.warning(f"Deletion not confirmed for model: {model_name}")
            return False
        
        if model_name not in self.model_registry:
            self.logger.error(f"Model '{model_name}' not found in registry")
            return False
        
        try:
            registry_entry = self.model_registry[model_name]
            
            # Delete model files
            for file_path in [registry_entry['model_path'], registry_entry['preprocessor_path'], 
                            registry_entry['metadata_path'], registry_entry['config_path']]:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove model directory if empty
            model_dir = os.path.dirname(registry_entry['model_path'])
            if os.path.exists(model_dir) and not os.listdir(model_dir):
                os.rmdir(model_dir)
            
            # Remove from registry
            del self.model_registry[model_name]
            self._save_registry()
            
            self.logger.info(f"Model '{model_name}' deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model '{model_name}': {e}")
            return False
    
    def process(self, data: Any) -> Any:
        """
        Process method required by BaseProcessor.
        
        Args:
            data: Input data containing model and export parameters
            
        Returns:
            Export results
        """
        if isinstance(data, dict):
            model = data.get('model')
            preprocessor = data.get('preprocessor')
            model_name = data.get('model_name', 'exported_model')
            
            if model is not None:
                return self.save_model_with_preprocessing(
                    model, preprocessor, model_name,
                    wavelengths=data.get('wavelengths'),
                    performance_metrics=data.get('performance_metrics'),
                    training_config=data.get('training_config'),
                    feature_names=data.get('feature_names')
                )
        
        return data