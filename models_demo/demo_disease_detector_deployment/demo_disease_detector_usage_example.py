#!/usr/bin/env python3
"""
Usage example for demo_disease_detector hyperspectral crop disease detection model.

This example demonstrates how to load and use the trained model for inference
on new hyperspectral data samples.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class Demo_Disease_DetectorPredictor:
    """
    Predictor class for demo_disease_detector model.
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
        
        print(f"Model loaded: {type(self.model).__name__}")
        if self.preprocessor:
            print(f"Preprocessor loaded: {type(self.preprocessor).__name__}")
    
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
        
        result = {
            'prediction': int(prediction),
            'disease_detected': bool(prediction),
            'prediction_label': 'Diseased' if prediction == 1 else 'Healthy'
        }
        
        # Add probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(spectral_data)[0]
            result['probabilities'] = {
                'healthy': float(probabilities[0]),
                'diseased': float(probabilities[1])
            }
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
            result = {
                'sample_index': i,
                'prediction': int(pred),
                'disease_detected': bool(pred),
                'prediction_label': 'Diseased' if pred == 1 else 'Healthy'
            }
            
            if probabilities is not None:
                result['probabilities'] = {
                    'healthy': float(probabilities[i][0]),
                    'diseased': float(probabilities[i][1])
                }
                result['confidence'] = float(max(probabilities[i]))
            
            results.append(result)
        
        return results


def example_usage():
    """Example usage of the predictor."""
    
    # Initialize predictor
    predictor = Demo_Disease_DetectorPredictor(
        model_path='demo_disease_detector.joblib',
        preprocessor_path='demo_disease_detector_preprocessor.joblib'
    )
    
    # Example 1: Single sample prediction
    print("\n=== Single Sample Prediction ===")
    
    # Create example spectral data (131 wavelengths from 437-2345 nm)
    np.random.seed(42)
    sample_spectra = np.random.random(131) * 0.5 + 0.2  # Simulated reflectance values
    
    result = predictor.predict_single_sample(sample_spectra)
    print(f"Prediction: {result['prediction_label']}")
    if 'confidence' in result:
        print(f"Confidence: {result['confidence']:.3f}")
    
    # Example 2: Batch prediction
    print("\n=== Batch Prediction ===")
    
    # Create batch of samples
    batch_spectra = np.random.random((5, 131)) * 0.5 + 0.2
    
    batch_results = predictor.predict_batch(batch_spectra)
    for result in batch_results:
        print(f"Sample {result['sample_index']}: {result['prediction_label']}")
    
    # Example 3: Loading from CSV file
    print("\n=== CSV File Processing ===")
    print("# To process a CSV file with hyperspectral data:")
    print("# df = pd.read_csv('your_data.csv')")
    print("# spectral_columns = [col for col in df.columns if col.startswith('X')]")
    print("# spectral_data = df[spectral_columns].values")
    print("# results = predictor.predict_batch(spectral_data)")


if __name__ == "__main__":
    example_usage()
