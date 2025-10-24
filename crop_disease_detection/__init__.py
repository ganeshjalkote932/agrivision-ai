"""
Crop Disease Detection System
A machine learning pipeline for detecting disease in crops using hyperspectral data.
"""

__version__ = "1.0.0"
__author__ = "Agricultural ML Team"

from .data_loader import DataLoader
# from .preprocessor import Preprocessor
# from .model_trainer import ModelTrainer
# from .model_evaluator import ModelEvaluator
# from .feature_analyzer import FeatureAnalyzer
# from .disease_detector import DiseaseDetector

__all__ = [
    'DataLoader',
    # 'Preprocessor', 
    # 'ModelTrainer',
    # 'ModelEvaluator',
    # 'FeatureAnalyzer',
    # 'DiseaseDetector'
]