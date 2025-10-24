# Design Document

## Overview

The Crop Disease Detection System is a machine learning pipeline designed to classify hyperspectral crop data as healthy or diseased. The system implements a modular architecture with separate components for data loading, preprocessing, model training, evaluation, and deployment. The design prioritizes memory efficiency for resource-constrained hardware while maintaining high accuracy through optimized feature engineering and model selection.

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loader   │───▶│  Preprocessor   │───▶│ Model Trainer   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Validator     │    │   Visualizer    │    │   Evaluator     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Model Exporter  │    │ Report Generator│    │ Feature Analyzer│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow Pipeline

1. **Input Stage**: CSV file containing hyperspectral data with metadata and spectral features
2. **Preprocessing Stage**: Feature extraction, normalization, and quality validation
3. **Training Stage**: Model training with hyperparameter optimization
4. **Evaluation Stage**: Performance assessment and validation
5. **Export Stage**: Model serialization and deployment preparation

## Components and Interfaces

### DataLoader Class

**Purpose**: Handle CSV loading, initial validation, and data structure analysis

**Key Methods**:
- `load_dataset(file_path: str) -> pd.DataFrame`
- `validate_data_structure() -> Dict[str, Any]`
- `generate_summary_statistics() -> Dict[str, Any]`
- `detect_data_quality_issues() -> List[str]`

**Inputs**: CSV file path
**Outputs**: Pandas DataFrame, validation report, summary statistics

### Preprocessor Class

**Purpose**: Transform raw spectral data into ML-ready features

**Key Methods**:
- `extract_spectral_features(df: pd.DataFrame) -> np.ndarray`
- `create_disease_labels(df: pd.DataFrame) -> np.ndarray`
- `normalize_features(features: np.ndarray, method: str) -> np.ndarray`
- `handle_missing_values(features: np.ndarray) -> np.ndarray`
- `detect_outliers(features: np.ndarray) -> np.ndarray`

**Design Decisions**:
- Use StandardScaler for normalization to handle varying spectral ranges
- Implement multiple outlier detection methods (IQR, Z-score, Isolation Forest)
- Create binary disease labels from crop stage and condition metadata
- Preserve preprocessing parameters for consistent inference

### ModelTrainer Class

**Purpose**: Train and optimize multiple classifier types

**Key Methods**:
- `train_mlp_classifier(X_train, y_train) -> MLPClassifier`
- `train_random_forest(X_train, y_train) -> RandomForestClassifier`
- `train_svm_classifier(X_train, y_train) -> SVC`
- `optimize_hyperparameters(model_type: str) -> Dict[str, Any]`
- `select_best_model(models: List) -> Any`

**Hardware Optimization**:
- Implement batch processing for MLP to manage memory usage
- Use n_jobs=-1 for Random Forest with CPU core detection
- Optimize SVM kernel selection for dataset size
- Implement early stopping for neural networks

### ModelEvaluator Class

**Purpose**: Comprehensive model performance assessment

**Key Methods**:
- `calculate_metrics(y_true, y_pred) -> Dict[str, float]`
- `generate_confusion_matrix(y_true, y_pred) -> np.ndarray`
- `plot_roc_curve(y_true, y_proba) -> matplotlib.Figure`
- `plot_learning_curves(model, X, y) -> matplotlib.Figure`
- `statistical_significance_test(results: List) -> Dict[str, float]`

### FeatureAnalyzer Class

**Purpose**: Interpret model decisions and identify important wavelengths

**Key Methods**:
- `extract_feature_importance(model) -> np.ndarray`
- `identify_top_wavelengths(importance_scores: np.ndarray, n_top: int) -> List[float]`
- `plot_spectral_importance(wavelengths: List, importance: np.ndarray) -> matplotlib.Figure`
- `validate_biological_relevance(top_wavelengths: List) -> Dict[str, str]`

## Data Models

### Input Data Schema

```python
class HyperspectralData:
    unique_id: int
    country: str
    crop: str  # corn, soybean, winter_wheat, cotton, rice
    stage: str  # Critical, Emerge_VEarly, Mature_Senesc, etc.
    spectral_features: Dict[str, float]  # X437 to X2345 wavelengths
    metadata: Dict[str, Any]  # lat, long, year, month, etc.
```

### Disease Label Creation Strategy

Since the original dataset doesn't contain explicit disease labels, the system will create binary labels based on:

1. **Crop Stage Analysis**: 
   - "Critical" stage → Diseased (1)
   - "Mature_Senesc" stage → Potentially diseased (1)
   - "Emerge_VEarly", "Early_Mid" → Healthy (0)
   - "Late", "Harvest" → Healthy (0)

2. **Spectral Anomaly Detection**:
   - Use statistical methods to identify spectral signatures indicating stress
   - Compare against known healthy spectral profiles
   - Flag samples with significant deviations as potentially diseased

### Processed Data Schema

```python
class ProcessedSample:
    sample_id: int
    normalized_spectra: np.ndarray  # Shape: (n_wavelengths,)
    disease_label: int  # 0=healthy, 1=diseased
    wavelengths: List[float]  # Corresponding wavelength values
    quality_score: float  # Data quality assessment
```

## Error Handling

### Data Quality Issues

1. **Missing Spectral Values**:
   - Strategy: Interpolation using neighboring wavelengths
   - Fallback: Remove samples with >10% missing values
   - Logging: Record all imputation actions

2. **Outlier Detection**:
   - Method: Combined IQR and Isolation Forest approach
   - Threshold: Samples beyond 3 standard deviations
   - Action: Flag for review, option to exclude or transform

3. **Invalid Spectral Ranges**:
   - Validation: Check for negative reflectance values
   - Correction: Apply physical constraints (0-100% reflectance)
   - Alert: Log all corrections for transparency

### Model Training Issues

1. **Memory Constraints**:
   - Batch processing for large datasets
   - Feature selection to reduce dimensionality
   - Model complexity reduction if needed

2. **Convergence Problems**:
   - Multiple random initializations
   - Adaptive learning rate schedules
   - Alternative optimization algorithms

3. **Class Imbalance**:
   - SMOTE oversampling for minority class
   - Class weight adjustment
   - Stratified sampling strategies

## Testing Strategy

### Unit Testing

1. **Data Loading Tests**:
   - Validate CSV parsing accuracy
   - Test error handling for malformed files
   - Verify data type conversions

2. **Preprocessing Tests**:
   - Test normalization consistency
   - Validate outlier detection accuracy
   - Check feature extraction correctness

3. **Model Training Tests**:
   - Verify hyperparameter optimization
   - Test model serialization/deserialization
   - Validate cross-validation implementation

### Integration Testing

1. **End-to-End Pipeline**:
   - Test complete workflow with sample data
   - Validate output format consistency
   - Check performance benchmarks

2. **Hardware Compatibility**:
   - Test memory usage under constraints
   - Validate CPU utilization efficiency
   - Check processing time benchmarks

### Performance Testing

1. **Accuracy Benchmarks**:
   - Target: >95% classification accuracy
   - Baseline: Compare against simple threshold methods
   - Validation: Cross-validation with multiple splits

2. **Resource Usage**:
   - Memory: <4GB peak usage for full dataset
   - CPU: Efficient multi-core utilization
   - Time: <30 minutes for complete training pipeline

## Model Selection Criteria

### Primary Models

1. **Multi-Layer Perceptron (MLP)**:
   - Architecture: 2-3 hidden layers with dropout
   - Activation: ReLU with batch normalization
   - Optimization: Adam optimizer with learning rate scheduling
   - Advantages: Can capture non-linear spectral relationships

2. **Random Forest**:
   - Trees: 100-500 estimators based on performance
   - Features: sqrt(n_features) per split
   - Depth: Limited to prevent overfitting
   - Advantages: Feature importance, robust to outliers

3. **Support Vector Machine (SVM)**:
   - Kernel: RBF with optimized gamma and C parameters
   - Scaling: Required due to spectral feature ranges
   - Implementation: sklearn.svm.SVC with probability estimates
   - Advantages: Effective in high-dimensional spaces

### Model Selection Process

1. **Cross-Validation**: 5-fold stratified CV for robust evaluation
2. **Hyperparameter Optimization**: Grid search with early stopping
3. **Ensemble Methods**: Voting classifier if individual models perform similarly
4. **Final Selection**: Based on validation accuracy, interpretability, and inference speed

## Deployment Considerations

### Model Artifacts

1. **Trained Model**: Serialized using joblib or pickle
2. **Preprocessing Pipeline**: StandardScaler parameters and transformations
3. **Feature Metadata**: Wavelength mappings and importance scores
4. **Configuration**: Hyperparameters and training settings

### Inference Pipeline

```python
class DiseaseDetector:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
    
    def predict(self, spectral_data: np.ndarray) -> Dict[str, Any]:
        # Preprocess input data
        processed_data = self.preprocessor.transform(spectral_data)
        
        # Generate prediction
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        return {
            'disease_detected': bool(prediction[0]),
            'confidence': float(probability[0].max()),
            'class_probabilities': probability[0].tolist()
        }
```

### Performance Monitoring

1. **Prediction Logging**: Track all predictions with timestamps
2. **Confidence Thresholds**: Alert for low-confidence predictions
3. **Model Drift Detection**: Monitor prediction distribution changes
4. **Retraining Triggers**: Automatic alerts when performance degrades