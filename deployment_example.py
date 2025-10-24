#!/usr/bin/env python3
"""
Simple deployment example for hyperspectral crop disease detection.

This script demonstrates how to export a trained model and create a deployment-ready
inference pipeline for real-time disease detection.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crop_disease_detection.model_exporter import ModelExporter
from crop_disease_detection.disease_detector import DiseaseDetector
from crop_disease_detection.config import ModelConfig


def create_sample_model():
    """Create a sample trained model for demonstration."""
    print("Creating sample model for demonstration...")
    
    # Create synthetic hyperspectral data (realistic reflectance values)
    X, y = make_classification(
        n_samples=1000,
        n_features=131,  # 131 wavelengths
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Scale to realistic reflectance range (0-1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0.1, 0.8))
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit preprocessor
    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate performance metrics
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    performance_metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test)
    }
    
    # Create wavelengths array
    wavelengths = np.linspace(437, 2345, 131).tolist()
    
    print(f"Model trained - Test accuracy: {test_accuracy:.4f}")
    
    return model, preprocessor, performance_metrics, wavelengths, X_test_scaled, y_test


def demonstrate_model_export():
    """Demonstrate model export functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL EXPORT")
    print("="*60)
    
    # Create sample model
    model, preprocessor, performance_metrics, wavelengths, X_test, y_test = create_sample_model()
    
    # Initialize model exporter
    model_config = ModelConfig()
    exporter = ModelExporter(output_dir="models_demo", model_config=model_config)
    
    # Export model with preprocessing
    print("\n1. Exporting model with preprocessing pipeline...")
    file_paths = exporter.save_model_with_preprocessing(
        model=model,
        preprocessor=preprocessor,
        model_name="demo_disease_detector",
        wavelengths=wavelengths,
        performance_metrics=performance_metrics,
        training_config={
            'algorithm': 'RandomForest',
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        },
        feature_names=[f"X{w:.0f}" for w in wavelengths]
    )
    
    print(f"✓ Model exported to: {file_paths['model_dir']}")
    print(f"✓ Files created:")
    for key, path in file_paths.items():
        if path:
            print(f"  - {key}: {os.path.basename(path)}")
    
    # Create deployment package
    print("\n2. Creating deployment package...")
    deployment_paths = exporter.export_model_for_deployment(
        model_name="demo_disease_detector",
        deployment_format="joblib",
        include_examples=True
    )
    
    print(f"✓ Deployment package created:")
    for key, path in deployment_paths.items():
        print(f"  - {key}: {os.path.basename(path)}")
    
    return file_paths, X_test, y_test


def demonstrate_inference_pipeline(file_paths, X_test, y_test):
    """Demonstrate inference pipeline functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING INFERENCE PIPELINE")
    print("="*60)
    
    # Initialize disease detector
    print("\n1. Initializing disease detector...")
    detector = DiseaseDetector(
        model_path=file_paths['model'],
        preprocessor_path=file_paths['preprocessor'],
        metadata_path=file_paths['metadata']
    )
    
    print("✓ Disease detector initialized")
    
    # Get model info
    model_info = detector.get_model_info()
    print(f"✓ Model type: {model_info['model_type']}")
    print(f"✓ Has preprocessor: {model_info['has_preprocessor']}")
    print(f"✓ Has probabilities: {model_info['has_probabilities']}")
    
    # Test single sample prediction
    print("\n2. Testing single sample prediction...")
    sample_idx = 0
    single_sample = X_test[sample_idx]
    
    result = detector.predict_single_sample(single_sample, include_metadata=True)
    
    print(f"✓ Sample {sample_idx} prediction:")
    print(f"  - Prediction: {result['prediction_label']}")
    print(f"  - Confidence: {result.get('confidence', 'N/A'):.3f}")
    print(f"  - Actual label: {'Diseased' if y_test[sample_idx] == 1 else 'Healthy'}")
    
    # Test batch prediction
    print("\n3. Testing batch prediction...")
    batch_size = 10
    batch_samples = X_test[:batch_size]
    
    batch_results = detector.predict_batch(batch_samples)
    
    print(f"✓ Batch prediction results ({batch_size} samples):")
    correct_predictions = 0
    for i, result in enumerate(batch_results):
        actual = y_test[i]
        predicted = result['prediction']
        is_correct = actual == predicted
        if is_correct:
            correct_predictions += 1
        
        print(f"  Sample {i}: {result['prediction_label']} "
              f"(confidence: {result.get('confidence', 0):.3f}) "
              f"{'✓' if is_correct else '✗'}")
    
    batch_accuracy = correct_predictions / batch_size
    print(f"✓ Batch accuracy: {batch_accuracy:.3f}")
    
    # Test CSV processing
    print("\n4. Testing CSV file processing...")
    
    # Create sample CSV file
    csv_data = pd.DataFrame(X_test[:20])
    csv_data.columns = [f"X{i+437}" for i in range(131)]  # Wavelength column names
    csv_data['sample_id'] = range(len(csv_data))
    csv_data['actual_label'] = y_test[:20]
    
    csv_path = "demo_samples.csv"
    csv_data.to_csv(csv_path, index=False)
    
    # Process CSV
    results_df = detector.predict_from_csv(
        csv_path=csv_path,
        output_path="demo_results.csv"
    )
    
    print(f"✓ Processed CSV file: {len(results_df)} samples")
    print(f"✓ Results saved to: demo_results.csv")
    
    # Calculate accuracy on CSV results
    csv_accuracy = (results_df['prediction'] == results_df['actual_label']).mean()
    print(f"✓ CSV processing accuracy: {csv_accuracy:.3f}")
    
    # Clean up demo files
    os.remove(csv_path)
    os.remove("demo_results.csv")


def demonstrate_model_loading():
    """Demonstrate loading models from the exporter."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL LOADING")
    print("="*60)
    
    # Initialize exporter
    exporter = ModelExporter(output_dir="models_demo")
    
    # List exported models
    print("\n1. Listing exported models...")
    models = exporter.list_exported_models()
    
    for model_name, model_info in models.items():
        print(f"✓ Model: {model_name}")
        print(f"  - Export time: {model_info['export_timestamp']}")
        print(f"  - Performance: {model_info.get('performance_metrics', {})}")
    
    # Load model using exporter
    print("\n2. Loading model using exporter...")
    if models:
        model_name = list(models.keys())[0]
        loaded_data = exporter.load_model_with_preprocessing(model_name)
        
        print(f"✓ Loaded model: {model_name}")
        print(f"  - Model type: {type(loaded_data['model']).__name__}")
        print(f"  - Has preprocessor: {loaded_data['preprocessor'] is not None}")
        print(f"  - Metadata keys: {list(loaded_data['metadata'].keys())}")
    
    # Initialize detector using exporter
    print("\n3. Initializing detector using exporter...")
    if models:
        detector = DiseaseDetector(
            model_name=model_name,
            model_exporter=exporter
        )
        
        print(f"✓ Detector initialized from exporter")
        
        # Test prediction with realistic spectral values
        sample_data = np.random.uniform(0.1, 0.8, 131)  # Realistic reflectance range
        result = detector.predict_single_sample(sample_data)
        print(f"✓ Test prediction: {result['prediction_label']}")


def main():
    """Main demonstration function."""
    print("HYPERSPECTRAL CROP DISEASE DETECTION - DEPLOYMENT DEMO")
    print("="*60)
    
    try:
        # Demonstrate model export
        file_paths, X_test, y_test = demonstrate_model_export()
        
        # Demonstrate inference pipeline
        demonstrate_inference_pipeline(file_paths, X_test, y_test)
        
        # Demonstrate model loading
        demonstrate_model_loading()
        
        print("\n" + "="*60)
        print("✅ ALL DEPLOYMENT DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Check the 'models_demo' directory for exported models")
        print("2. Review the deployment package in 'models_demo/demo_disease_detector_deployment'")
        print("3. Use the generated example code for your own deployment")
        print("4. Refer to DEPLOYMENT_GUIDE.md for production deployment instructions")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)