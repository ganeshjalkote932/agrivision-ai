"""
Base classes and interfaces for the crop disease detection system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging


@dataclass
class HyperspectralData:
    """Data structure for hyperspectral crop samples."""
    unique_id: int
    country: str
    crop: str
    stage: str
    spectral_features: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class ProcessedSample:
    """Data structure for processed hyperspectral samples."""
    sample_id: int
    normalized_spectra: np.ndarray
    disease_label: int
    wavelengths: List[float]
    quality_score: float


class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data and return processed output."""
        pass
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data format and quality."""
        return True


class BaseModel(ABC):
    """Abstract base class for machine learning models."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_trained = False
        self.model = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on provided data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        pass


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pass


class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass


class DataQualityError(Exception):
    """Raised when data quality issues are detected."""
    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass