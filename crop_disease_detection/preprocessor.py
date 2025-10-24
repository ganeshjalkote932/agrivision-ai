"""
Data preprocessing module for hyperspectral crop disease detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib

from .base import BaseProcessor
from .config import PreprocessingConfig


class Preprocessor(BaseProcessor):
    """
    Comprehensive preprocessing pipeline for hyperspectral crop data.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize the preprocessor."""
        super().__init__()
        self.config = config
        self.scaler = None
        self.imputer = None
        self.is_fitted = False
        self.feature_names = None
        self.selected_wavelengths = None
        
    def extract_spectral_features(self, df: pd.DataFrame, spectral_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract spectral features from X437-X2345 columns.
        
        Args:
            df: DataFrame containing hyperspectral data
            spectral_columns: Optional list of spectral column names. If None, auto-detects X437-X2345 range
            
        Returns:
            numpy array of spectral features
        """
        self.logger.info("Extracting spectral features from X437-X2345 range...")
        
        # Auto-detect spectral columns if not provided
        if spectral_columns is None:
            spectral_columns = self._get_spectral_columns(df)
        
        # Validate spectral columns exist in dataframe
        missing_cols = [col for col in spectral_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing spectral columns: {missing_cols}")
        
        # Extract spectral data
        spectral_data = df[spectral_columns].copy()
        
        # Convert to numeric, handling any non-numeric values
        for col in spectral_columns:
            spectral_data[col] = pd.to_numeric(spectral_data[col], errors='coerce')
        
        # Store wavelength information
        wavelengths = [float(col[1:]) for col in spectral_columns]
        self.selected_wavelengths = wavelengths
        self.feature_names = spectral_columns
        
        features = spectral_data.values
        self.logger.info(f"Extracted {features.shape[1]} spectral features from {features.shape[0]} samples")
        self.logger.info(f"Wavelength range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
        
        return features
    
    def _get_spectral_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect spectral columns in X437-X2345 range.
        
        Args:
            df: DataFrame to search for spectral columns
            
        Returns:
            List of spectral column names in wavelength order
        """
        spectral_cols = []
        
        for col in df.columns:
            if col.startswith('X') and len(col) > 1:
                try:
                    wavelength = float(col[1:])
                    # Check if wavelength is in expected range (437-2345 nm)
                    if 437 <= wavelength <= 2345:
                        spectral_cols.append(col)
                except ValueError:
                    continue
        
        # Sort by wavelength
        spectral_cols.sort(key=lambda x: float(x[1:]))
        
        if not spectral_cols:
            raise ValueError("No spectral columns found in X437-X2345 range")
        
        self.logger.info(f"Auto-detected {len(spectral_cols)} spectral columns")
        return spectral_cols
    
    def separate_metadata_from_spectral(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate metadata columns from spectral data columns.
        
        Args:
            df: Original DataFrame with mixed metadata and spectral data
            
        Returns:
            Tuple of (metadata_df, spectral_df)
        """
        self.logger.info("Separating metadata from spectral data...")
        
        # Get spectral columns
        spectral_columns = self._get_spectral_columns(df)
        
        # Metadata columns are all non-spectral columns
        metadata_columns = [col for col in df.columns if col not in spectral_columns]
        
        # Create separate dataframes
        metadata_df = df[metadata_columns].copy()
        spectral_df = df[spectral_columns].copy()
        
        self.logger.info(f"Separated {len(metadata_columns)} metadata columns and {len(spectral_columns)} spectral columns")
        self.logger.info(f"Metadata columns: {metadata_columns[:5]}{'...' if len(metadata_columns) > 5 else ''}")
        
        return metadata_df, spectral_df
    
    def handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values in spectral data."""
        missing_count = np.sum(np.isnan(X))
        if missing_count == 0:
            self.logger.info("No missing values found")
            return X
            
        self.logger.info(f"Handling {missing_count} missing values...")
        
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='mean')
            X_processed = self.imputer.fit_transform(X)
        else:
            X_processed = self.imputer.transform(X)
            
        return X_processed
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize spectral features."""
        self.logger.info(f"Normalizing features using {self.config.normalization_method} scaling")
        
        if self.config.normalization_method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_normalized = self.scaler.fit_transform(X)
            else:
                X_normalized = self.scaler.transform(X)
        elif self.config.normalization_method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                X_normalized = self.scaler.fit_transform(X)
            else:
                X_normalized = self.scaler.transform(X)
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")
            
        return X_normalized
    
    def create_disease_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Create binary disease labels from crop metadata."""
        from .label_creator import DiseaseLabelCreator
        
        label_creator = DiseaseLabelCreator(self.config)
        spectral_columns = [col for col in df.columns if col.startswith('X') and col[1:].isdigit()]
        labels, _ = label_creator.process(df, spectral_columns, method="stage_based")
        return labels
    
    def fit_transform(self, df: pd.DataFrame, spectral_columns: Optional[List[str]] = None, 
                     create_labels: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: DataFrame containing hyperspectral data
            spectral_columns: Optional list of spectral column names. If None, auto-detects X437-X2345 range
            create_labels: Whether to create disease labels from metadata
            
        Returns:
            Tuple of (processed_features, labels)
        """
        self.logger.info("Fitting preprocessor and transforming data...")
        
        # Extract spectral features
        X = self.extract_spectral_features(df, spectral_columns)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Normalize features
        X = self.normalize_features(X)
        
        # Mark as fitted
        self.is_fitted = True
        
        # Create labels if requested
        y = None
        if create_labels:
            y = self.create_disease_labels(df)
            
            # Ensure labels match processed data length
            if len(y) != len(X):
                min_length = min(len(X), len(y))
                X = X[:min_length]
                y = y[:min_length]
        
        self.logger.info(f"Preprocessing completed: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def transform(self, df: pd.DataFrame, spectral_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: DataFrame containing hyperspectral data
            spectral_columns: Optional list of spectral column names. If None, uses same columns as fit
            
        Returns:
            Processed spectral features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Use fit_transform() first.")
        
        self.logger.info("Transforming new data...")
        
        # Use same spectral columns as during fitting if not specified
        if spectral_columns is None:
            spectral_columns = self.feature_names
        
        # Extract spectral features
        X = self.extract_spectral_features(df, spectral_columns)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Normalize features
        X = self.normalize_features(X)
        
        self.logger.info(f"Data transformation completed: {X.shape[0]} samples, {X.shape[1]} features")
        return X
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the fitted preprocessor to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        preprocessor_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'config': self.config,
            'selected_wavelengths': self.selected_wavelengths,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        self.logger.info(f"Preprocessor saved to: {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str) -> 'Preprocessor':
        """Load a fitted preprocessor from file."""
        preprocessor_data = joblib.load(filepath)
        
        # Create new instance
        preprocessor = cls(preprocessor_data['config'])
        
        # Restore fitted components
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.imputer = preprocessor_data['imputer']
        preprocessor.selected_wavelengths = preprocessor_data['selected_wavelengths']
        preprocessor.feature_names = preprocessor_data['feature_names']
        preprocessor.is_fitted = preprocessor_data['is_fitted']
        
        return preprocessor
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about processed features.
        
        Returns:
            Dictionary containing feature information
        """
        if not self.is_fitted:
            return {
                'status': 'not_fitted',
                'message': 'Preprocessor has not been fitted yet'
            }
        
        return {
            'status': 'fitted',
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'wavelength_range': f"{min(self.selected_wavelengths):.1f}-{max(self.selected_wavelengths):.1f} nm" if self.selected_wavelengths else None,
            'wavelength_count': len(self.selected_wavelengths) if self.selected_wavelengths else 0,
            'normalization_method': self.config.normalization_method,
            'has_imputer': self.imputer is not None,
            'has_scaler': self.scaler is not None
        }
    
    def get_spectral_range_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the spectral range being processed.
        
        Returns:
            Dictionary with spectral range details
        """
        if not self.selected_wavelengths:
            return {'error': 'No spectral data processed yet'}
        
        wavelengths = np.array(self.selected_wavelengths)
        
        return {
            'min_wavelength': float(np.min(wavelengths)),
            'max_wavelength': float(np.max(wavelengths)),
            'n_wavelengths': len(wavelengths),
            'wavelength_step_mean': float(np.mean(np.diff(wavelengths))) if len(wavelengths) > 1 else 0,
            'wavelength_step_std': float(np.std(np.diff(wavelengths))) if len(wavelengths) > 1 else 0,
            'spectral_resolution': 'variable' if len(wavelengths) > 1 and np.std(np.diff(wavelengths)) > 1 else 'uniform'
        }
    
    def process(self, df: pd.DataFrame, spectral_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main processing method for backward compatibility.
        
        Args:
            df: DataFrame containing hyperspectral data
            spectral_columns: Optional list of spectral column names
            
        Returns:
            Tuple of (processed_features, labels)
        """
        return self.fit_transform(df, spectral_columns)