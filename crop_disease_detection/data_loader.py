"""
Data loading and validation module for hyperspectral crop data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path

from .base import BaseProcessor, HyperspectralData, DataQualityError
from .utils import get_spectral_wavelengths, validate_spectral_data, calculate_memory_usage
from .config import DataConfig
from .data_quality import DataQualityAssessor
from .visualization import DataVisualizer


class DataLoader(BaseProcessor):
    """
    Handles loading and initial validation of hyperspectral CSV data.
    
    This class provides methods to load CSV files containing hyperspectral data,
    validate data structure and quality, and generate comprehensive statistics.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: DataConfig instance with loading parameters
        """
        super().__init__()
        self.config = config
        self.dataset = None
        self.spectral_columns = None
        self.metadata_columns = None
        self.wavelengths = None
        
        # Initialize quality assessor and visualizer
        self.quality_assessor = DataQualityAssessor()
        self.visualizer = DataVisualizer()
        
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load hyperspectral dataset from CSV file with comprehensive error handling.
        
        Args:
            file_path: Path to the CSV file containing hyperspectral data
            
        Returns:
            Loaded pandas DataFrame
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            DataQualityError: If the data structure is invalid
        """
        self.logger.info(f"Loading dataset from: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            # Load CSV with error handling for different encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise DataQualityError("Could not read CSV file with any supported encoding")
            
            # Store the dataset
            self.dataset = df
            
            # Identify column types
            self._identify_column_types()
            
            # Basic validation
            self._validate_basic_structure()
            
            self.logger.info(f"Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} columns")
            
            return df
            
        except pd.errors.EmptyDataError:
            raise DataQualityError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise DataQualityError(f"Error parsing CSV file: {str(e)}")
        except Exception as e:
            raise DataQualityError(f"Unexpected error loading dataset: {str(e)}")
    
    def _identify_column_types(self) -> None:
        """Identify spectral and metadata columns in the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Identify spectral columns (X followed by numbers)
        self.spectral_columns = [col for col in self.dataset.columns 
                               if col.startswith('X') and col[1:].replace('.', '').isdigit()]
        
        # Sort spectral columns by wavelength
        self.spectral_columns.sort(key=lambda x: float(x[1:]))
        
        # Extract wavelengths
        self.wavelengths = [float(col[1:]) for col in self.spectral_columns]
        
        # Identify metadata columns
        self.metadata_columns = [col for col in self.dataset.columns 
                               if col not in self.spectral_columns]
        
        self.logger.info(f"Identified {len(self.spectral_columns)} spectral columns")
        self.logger.info(f"Wavelength range: {min(self.wavelengths):.1f} - {max(self.wavelengths):.1f} nm")
        self.logger.info(f"Metadata columns: {len(self.metadata_columns)}")
    
    def _validate_basic_structure(self) -> None:
        """Validate basic data structure requirements."""
        if len(self.spectral_columns) == 0:
            raise DataQualityError("No spectral columns found (expected columns starting with 'X')")
        
        # Check for required metadata columns
        required_columns = ['Crop', 'Stage']
        missing_columns = [col for col in required_columns if col not in self.metadata_columns]
        if missing_columns:
            raise DataQualityError(f"Missing required columns: {missing_columns}")
        
        # Validate wavelength range
        if min(self.wavelengths) < self.config.min_wavelength:
            self.logger.warning(f"Minimum wavelength {min(self.wavelengths)} below expected range")
        if max(self.wavelengths) > self.config.max_wavelength:
            self.logger.warning(f"Maximum wavelength {max(self.wavelengths)} above expected range")
    
    def validate_data_structure(self) -> Dict[str, Any]:
        """
        Perform comprehensive validation of data structure and types.
        
        Returns:
            Dictionary containing validation results and recommendations
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'structure_info': {}
        }
        
        # Check data types
        spectral_data = self.dataset[self.spectral_columns]
        
        # Validate spectral data types
        non_numeric_spectral = []
        for col in self.spectral_columns:
            if not pd.api.types.is_numeric_dtype(self.dataset[col]):
                non_numeric_spectral.append(col)
        
        if non_numeric_spectral:
            validation_results['issues'].append(
                f"Non-numeric spectral columns found: {non_numeric_spectral[:5]}..."
            )
            validation_results['valid'] = False
        
        # Check for duplicate samples
        if 'UniqueID' in self.dataset.columns:
            duplicate_ids = self.dataset['UniqueID'].duplicated().sum()
            if duplicate_ids > 0:
                validation_results['warnings'].append(f"Found {duplicate_ids} duplicate UniqueIDs")
        
        # Validate spectral ranges
        spectral_validation = validate_spectral_data(spectral_data.values, self.wavelengths)
        if not spectral_validation['valid']:
            validation_results['issues'].extend(spectral_validation['issues'])
            validation_results['valid'] = False
        
        # Memory usage estimation
        memory_info = calculate_memory_usage(spectral_data.size, 'float64')
        validation_results['structure_info']['memory_usage_mb'] = memory_info['mb']
        
        if memory_info['gb'] > 3.0:  # Conservative limit for 4GB system
            validation_results['warnings'].append(
                f"High memory usage estimated: {memory_info['gb']:.2f} GB"
            )
            validation_results['recommendations'].append(
                "Consider using float32 precision or batch processing"
            )
        
        # Check crop and stage distributions
        crop_counts = self.dataset['Crop'].value_counts()
        stage_counts = self.dataset['Stage'].value_counts()
        
        validation_results['structure_info']['crop_distribution'] = crop_counts.to_dict()
        validation_results['structure_info']['stage_distribution'] = stage_counts.to_dict()
        
        # Check for class imbalance in stages
        min_stage_count = stage_counts.min()
        max_stage_count = stage_counts.max()
        imbalance_ratio = max_stage_count / min_stage_count if min_stage_count > 0 else float('inf')
        
        if imbalance_ratio > 10:
            validation_results['warnings'].append(
                f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
            )
            validation_results['recommendations'].append(
                "Consider using stratified sampling or class balancing techniques"
            )
        
        self.logger.info(f"Data structure validation completed. Valid: {validation_results['valid']}")
        if validation_results['issues']:
            self.logger.warning(f"Issues found: {len(validation_results['issues'])}")
        if validation_results['warnings']:
            self.logger.warning(f"Warnings: {len(validation_results['warnings'])}")
        
        return validation_results
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for the dataset.
        
        Returns:
            Dictionary containing detailed dataset statistics
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        summary_stats = {
            'dataset_info': {},
            'spectral_statistics': {},
            'metadata_statistics': {},
            'quality_metrics': {}
        }
        
        # Basic dataset information
        summary_stats['dataset_info'] = {
            'total_samples': len(self.dataset),
            'total_columns': len(self.dataset.columns),
            'spectral_columns': len(self.spectral_columns),
            'metadata_columns': len(self.metadata_columns),
            'wavelength_range': f"{min(self.wavelengths):.1f} - {max(self.wavelengths):.1f} nm",
            'spectral_resolution': f"{np.mean(np.diff(self.wavelengths)):.1f} nm"
        }
        
        # Spectral data statistics
        spectral_data = self.dataset[self.spectral_columns]
        summary_stats['spectral_statistics'] = {
            'mean_reflectance': float(spectral_data.mean().mean()),
            'std_reflectance': float(spectral_data.std().mean()),
            'min_reflectance': float(spectral_data.min().min()),
            'max_reflectance': float(spectral_data.max().max()),
            'missing_values': int(spectral_data.isnull().sum().sum()),
            'missing_percentage': float(spectral_data.isnull().sum().sum() / spectral_data.size * 100)
        }
        
        # Metadata statistics
        summary_stats['metadata_statistics'] = {
            'crop_types': self.dataset['Crop'].nunique(),
            'crop_distribution': self.dataset['Crop'].value_counts().to_dict(),
            'stage_types': self.dataset['Stage'].nunique(),
            'stage_distribution': self.dataset['Stage'].value_counts().to_dict()
        }
        
        # Add geographic information if available
        if 'Country' in self.dataset.columns:
            summary_stats['metadata_statistics']['countries'] = self.dataset['Country'].nunique()
            summary_stats['metadata_statistics']['country_distribution'] = (
                self.dataset['Country'].value_counts().to_dict()
            )
        
        # Quality metrics
        summary_stats['quality_metrics'] = {
            'complete_samples': int((~spectral_data.isnull().any(axis=1)).sum()),
            'samples_with_missing': int(spectral_data.isnull().any(axis=1).sum()),
            'duplicate_samples': int(self.dataset.duplicated().sum())
        }
        
        # Memory usage
        memory_info = calculate_memory_usage(spectral_data.size, 'float64')
        summary_stats['quality_metrics']['estimated_memory_mb'] = memory_info['mb']
        
        self.logger.info("Generated comprehensive summary statistics")
        return summary_stats
    
    def detect_data_quality_issues(self) -> List[str]:
        """
        Detect and report data quality issues.
        
        Returns:
            List of detected data quality issues
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        issues = []
        
        # Check for missing values in critical columns
        for col in ['Crop', 'Stage']:
            missing_count = self.dataset[col].isnull().sum()
            if missing_count > 0:
                issues.append(f"Missing values in {col}: {missing_count} samples")
        
        # Check spectral data quality
        spectral_data = self.dataset[self.spectral_columns]
        
        # Negative reflectance values
        negative_count = (spectral_data < 0).sum().sum()
        if negative_count > 0:
            issues.append(f"Negative reflectance values: {negative_count} occurrences")
        
        # Extremely high reflectance values
        high_count = (spectral_data > 100).sum().sum()
        if high_count > 0:
            issues.append(f"Reflectance values >100%: {high_count} occurrences")
        
        # Check for constant spectral signatures (likely errors)
        constant_spectra = 0
        for idx in range(len(spectral_data)):
            spectrum = spectral_data.iloc[idx].values
            if len(np.unique(spectrum[~np.isnan(spectrum)])) <= 2:
                constant_spectra += 1
        
        if constant_spectra > 0:
            issues.append(f"Samples with constant/near-constant spectra: {constant_spectra}")
        
        # Check for samples with too many missing values
        missing_threshold = len(self.spectral_columns) * 0.1  # 10% threshold
        samples_high_missing = (spectral_data.isnull().sum(axis=1) > missing_threshold).sum()
        if samples_high_missing > 0:
            issues.append(f"Samples with >10% missing spectral values: {samples_high_missing}")
        
        # Check for unrealistic crop stage combinations
        stage_crop_combinations = self.dataset.groupby(['Crop', 'Stage']).size()
        unusual_combinations = []
        for (crop, stage), count in stage_crop_combinations.items():
            if count < 5:  # Very few samples for this combination
                unusual_combinations.append(f"{crop}-{stage}")
        
        if unusual_combinations:
            issues.append(f"Unusual crop-stage combinations with <5 samples: {len(unusual_combinations)}")
        
        self.logger.info(f"Data quality check completed. Found {len(issues)} issues.")
        return issues
    
    def get_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
        """
        Get first n samples for inspection.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            DataFrame with first n samples
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        return self.dataset.head(n_samples)
    
    def process(self, file_path: str) -> pd.DataFrame:
        """
        Main processing method that loads and validates data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded and validated DataFrame
        """
        # Load the dataset
        dataset = self.load_dataset(file_path)
        
        # Validate structure
        validation_results = self.validate_data_structure()
        
        # Generate statistics
        summary_stats = self.generate_summary_statistics()
        
        # Detect quality issues
        quality_issues = self.detect_data_quality_issues()
        
        # Generate comprehensive quality report
        quality_report = self.quality_assessor.generate_quality_report(dataset, self.spectral_columns)
        
        # Create visualization plots
        self.logger.info("Generating data exploration visualizations...")
        figures = self.visualizer.create_comprehensive_report_plots(
            dataset, self.spectral_columns, quality_report
        )
        
        # Log summary
        self.logger.info("="*50)
        self.logger.info("DATA LOADING SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Samples: {summary_stats['dataset_info']['total_samples']}")
        self.logger.info(f"Spectral bands: {summary_stats['dataset_info']['spectral_columns']}")
        self.logger.info(f"Wavelength range: {summary_stats['dataset_info']['wavelength_range']}")
        self.logger.info(f"Crop types: {summary_stats['metadata_statistics']['crop_types']}")
        self.logger.info(f"Growth stages: {summary_stats['metadata_statistics']['stage_types']}")
        self.logger.info(f"Data quality issues: {len(quality_issues)}")
        self.logger.info(f"Memory usage: {summary_stats['quality_metrics']['estimated_memory_mb']:.1f} MB")
        
        if not validation_results['valid']:
            self.logger.error("Data validation failed. Please check the issues before proceeding.")
            raise DataQualityError("Dataset validation failed")
        
        return dataset