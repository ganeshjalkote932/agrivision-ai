"""
Configuration settings for the crop disease detection system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    csv_file_path: str = "GHISACONUS_2008_001_speclib.csv"
    test_size: float = 0.2
    random_state: int = 42
    min_wavelength: float = 400.0
    max_wavelength: float = 2500.0
    
    # Disease labeling strategy
    diseased_stages: List[str] = None
    healthy_stages: List[str] = None
    
    def __post_init__(self):
        if self.diseased_stages is None:
            self.diseased_stages = ['Critical', 'Mature_Senesc']
        if self.healthy_stages is None:
            self.healthy_stages = ['Emerge_VEarly', 'Early_Mid', 'Late', 'Harvest']


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    normalization_method: str = "standard"  # "standard", "minmax", "robust"
    outlier_detection_method: str = "isolation_forest"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 0.1  # Fraction of outliers to detect
    missing_value_strategy: str = "interpolate"  # "interpolate", "mean", "median", "drop"
    max_missing_percentage: float = 10.0  # Maximum percentage of missing values per sample


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # MLP Configuration
    mlp_hidden_layers: tuple = (256, 128, 64)
    mlp_activation: str = "relu"
    mlp_solver: str = "adam"
    mlp_learning_rate: str = "adaptive"
    mlp_max_iter: int = 1000
    mlp_early_stopping: bool = True
    mlp_validation_fraction: float = 0.1
    
    # Random Forest Configuration
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_max_features: str = "sqrt"
    rf_bootstrap: bool = True
    
    # SVM Configuration
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    svm_probability: bool = True
    
    # General training configuration
    cv_folds: int = 5
    n_jobs: int = -1  # Use all available cores
    random_state: int = 42


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    # MLP hyperparameter grid
    mlp_param_grid: Dict[str, List[Any]] = None
    
    # Random Forest hyperparameter grid
    rf_param_grid: Dict[str, List[Any]] = None
    
    # SVM hyperparameter grid
    svm_param_grid: Dict[str, List[Any]] = None
    
    # Grid search configuration
    scoring: str = "accuracy"
    cv_folds: int = 3  # Reduced for hyperparameter search
    n_jobs: int = -1
    verbose: int = 1
    
    def __post_init__(self):
        if self.mlp_param_grid is None:
            self.mlp_param_grid = {
                'hidden_layer_sizes': [(128, 64), (256, 128, 64), (512, 256, 128)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }
        
        if self.rf_param_grid is None:
            self.rf_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        if self.svm_param_grid is None:
            self.svm_param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    metrics: List[str] = None
    plot_roc_curve: bool = True
    plot_confusion_matrix: bool = True
    plot_learning_curves: bool = True
    plot_feature_importance: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    plots_dir: str = "results/plots"
    reports_dir: str = "results/reports"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Hardware constraints
    max_memory_gb: float = 4.0
    batch_size: int = 1000
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/crop_disease_detection.log"
    
    # Output directories
    results_dir: str = "results"
    models_dir: str = "models"
    plots_dir: str = "results/plots"
    reports_dir: str = "results/reports"
    
    # Reproducibility
    random_state: int = 42
    
    # Performance targets
    target_accuracy: float = 0.95
    max_training_time_minutes: int = 30


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self):
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.hyperparameter = HyperparameterConfig()
        self.evaluation = EvaluationConfig()
        self.system = SystemConfig()
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'model': self.model.__dict__,
            'hyperparameter': self.hyperparameter.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__
        }


# Global configuration instance
config = Config()