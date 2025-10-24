# Hyperspectral Crop Disease Detection - Deployment Guide

This guide provides comprehensive instructions for deploying the trained hyperspectral crop disease detection models in production environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Model Export](#model-export)
3. [Deployment Options](#deployment-options)
4. [API Integration](#api-integration)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

```bash
# Required Python packages
pip install numpy pandas scikit-learn joblib matplotlib seaborn

# Optional for enhanced functionality
pip install flask fastapi uvicorn  # For API deployment
```

### Basic Usage

```python
from crop_disease_detection.disease_detector import DiseaseDetector
import numpy as np

# Initialize detector with trained model
detector = DiseaseDetector(
    model_path="models/best_model/best_model_model.joblib",
    preprocessor_path="models/best_model/best_model_preprocessor.joblib",
    metadata_path="models/best_model/best_model_metadata.json"
)

# Predict single sample
spectral_data = np.random.random(131)  # 131 wavelengths
result = detector.predict_single_sample(spectral_data)
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Model Export

### Exporting Trained Models

```python
from crop_disease_detection.model_exporter import ModelExporter
from crop_disease_detection.config import ModelConfig

# Initialize exporter
exporter = ModelExporter(output_dir="models")

# Export model with preprocessing pipeline
file_paths = exporter.save_model_with_preprocessing(
    model=trained_model,
    preprocessor=fitted_preprocessor,
    model_name="production_model_v1",
    wavelengths=wavelength_list,
    performance_metrics={
        'accuracy': 0.91,
        'precision': 0.89,
        'recall': 0.93,
        'f1_score': 0.91
    },
    training_config={
        'algorithm': 'RandomForest',
        'n_estimators': 100,
        'max_depth': 15
    }
)

print(f"Model exported to: {file_paths['model_dir']}")
```

### Creating Deployment Package

```python
# Create deployment-ready package
deployment_paths = exporter.export_model_for_deployment(
    model_name="production_model_v1",
    deployment_format="joblib",
    include_examples=True
)

print(f"Deployment package: {deployment_paths}")
```

## Deployment Options

### Option 1: Standalone Python Application

Create a standalone application for batch processing:

```python
#!/usr/bin/env python3
"""
Standalone crop disease detection application.
"""

import argparse
import pandas as pd
from crop_disease_detection.disease_detector import DiseaseDetector

def main():
    parser = argparse.ArgumentParser(description='Crop Disease Detection')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--model', required=True, help='Model directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DiseaseDetector(
        model_path=f"{args.model}/model.joblib",
        preprocessor_path=f"{args.model}/preprocessor.joblib"
    )
    
    # Process CSV file
    results = detector.predict_from_csv(args.input, output_path=args.output)
    print(f"Processed {len(results)} samples. Results saved to {args.output}")

if __name__ == "__main__":
    main()
```

### Option 2: REST API with Flask

```python
from flask import Flask, request, jsonify
import numpy as np
from crop_disease_detection.disease_detector import DiseaseDetector

app = Flask(__name__)

# Initialize detector globally
detector = DiseaseDetector(
    model_path="models/production_model/model.joblib",
    preprocessor_path="models/production_model/preprocessor.joblib"
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        spectral_data = np.array(data['spectral_data'])
        
        result = detector.predict_single_sample(spectral_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.json
        spectral_data = np.array(data['spectral_data'])
        
        results = detector.predict_batch(spectral_data)
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    model_info = detector.get_model_info()
    return jsonify(model_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Option 3: FastAPI (Recommended for Production)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from crop_disease_detection.disease_detector import DiseaseDetector

app = FastAPI(title="Crop Disease Detection API", version="1.0.0")

# Initialize detector
detector = DiseaseDetector(
    model_path="models/production_model/model.joblib",
    preprocessor_path="models/production_model/preprocessor.joblib"
)

class SpectralData(BaseModel):
    spectral_values: List[float]

class BatchSpectralData(BaseModel):
    samples: List[List[float]]

@app.post("/predict")
async def predict_single(data: SpectralData):
    try:
        spectral_array = np.array(data.spectral_values)
        result = detector.predict_single_sample(spectral_array)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(data: BatchSpectralData):
    try:
        spectral_array = np.array(data.samples)
        results = detector.predict_batch(spectral_array)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return detector.get_model_info()

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Integration

### Example Client Code

```python
import requests
import numpy as np

# API endpoint
API_URL = "http://localhost:8000"

# Single prediction
spectral_data = np.random.random(131).tolist()
response = requests.post(
    f"{API_URL}/predict",
    json={"spectral_values": spectral_data}
)
result = response.json()
print(f"Prediction: {result['prediction_label']}")

# Batch prediction
batch_data = [np.random.random(131).tolist() for _ in range(5)]
response = requests.post(
    f"{API_URL}/predict_batch",
    json={"samples": batch_data}
)
results = response.json()
print(f"Batch results: {len(results['predictions'])} predictions")
```

### JavaScript Integration

```javascript
// Single prediction
async function predictDisease(spectralData) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            spectral_values: spectralData
        })
    });
    
    const result = await response.json();
    return result;
}

// Usage
const spectralValues = new Array(131).fill(0).map(() => Math.random());
predictDisease(spectralValues).then(result => {
    console.log(`Disease detected: ${result.disease_detected}`);
    console.log(`Confidence: ${result.confidence}`);
});
```

## Performance Optimization

### Memory Optimization

```python
# For large batch processing
detector = DiseaseDetector(model_path="model.joblib")

# Process in smaller batches to manage memory
large_dataset = np.random.random((10000, 131))
batch_size = 100

all_results = []
for i in range(0, len(large_dataset), batch_size):
    batch = large_dataset[i:i+batch_size]
    batch_results = detector.predict_batch(batch)
    all_results.extend(batch_results)
```

### Caching for Repeated Predictions

```python
from functools import lru_cache
import hashlib

class CachedDiseaseDetector(DiseaseDetector):
    @lru_cache(maxsize=1000)
    def _cached_predict(self, data_hash):
        # This would need the actual spectral data, 
        # but demonstrates the caching concept
        pass
    
    def predict_with_cache(self, spectral_data):
        # Create hash of input data
        data_hash = hashlib.md5(spectral_data.tobytes()).hexdigest()
        
        # Check cache or compute prediction
        # Implementation would store and retrieve from cache
        return self.predict_single_sample(spectral_data)
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY crop_disease_detection/ ./crop_disease_detection/
COPY models/ ./models/
COPY app.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  disease-detector:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models/production_model
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - disease-detector
    restart: unless-stopped
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```python
   # Check if model files exist
   import os
   model_path = "models/my_model/model.joblib"
   if not os.path.exists(model_path):
       print(f"Model file not found: {model_path}")
   ```

2. **Input Validation Errors**
   ```python
   # Validate input dimensions
   detector = DiseaseDetector(model_path="model.joblib")
   spectral_data = np.random.random(131)  # Ensure correct number of features
   
   is_valid, error_msg = detector.validate_input_data(spectral_data)
   if not is_valid:
       print(f"Input validation failed: {error_msg}")
   ```

3. **Memory Issues**
   ```python
   # Monitor memory usage
   import psutil
   
   def check_memory():
       memory = psutil.virtual_memory()
       print(f"Memory usage: {memory.percent}%")
       return memory.percent < 90  # Return False if memory > 90%
   
   # Process in smaller batches if memory is high
   if not check_memory():
       batch_size = 50  # Reduce batch size
   ```

### Performance Monitoring

```python
import time
import logging

class MonitoredDiseaseDetector(DiseaseDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_count = 0
        self.total_time = 0
    
    def predict_single_sample(self, *args, **kwargs):
        start_time = time.time()
        result = super().predict_single_sample(*args, **kwargs)
        
        prediction_time = time.time() - start_time
        self.prediction_count += 1
        self.total_time += prediction_time
        
        avg_time = self.total_time / self.prediction_count
        logging.info(f"Prediction {self.prediction_count}: {prediction_time:.3f}s (avg: {avg_time:.3f}s)")
        
        return result
```

## Security Considerations

### Input Sanitization

```python
def sanitize_spectral_input(spectral_data):
    """Sanitize and validate spectral input data."""
    
    # Convert to numpy array
    if not isinstance(spectral_data, np.ndarray):
        spectral_data = np.array(spectral_data)
    
    # Check for reasonable spectral values
    if np.any(spectral_data < 0) or np.any(spectral_data > 2.0):
        raise ValueError("Spectral values out of reasonable range")
    
    # Check for suspicious patterns
    if np.all(spectral_data == spectral_data[0]):
        raise ValueError("All spectral values are identical (suspicious)")
    
    return spectral_data
```

### Rate Limiting (Flask example)

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Prediction logic here
    pass
```

This deployment guide provides comprehensive instructions for deploying the hyperspectral crop disease detection system in various environments, from standalone applications to production APIs with proper monitoring and security considerations.