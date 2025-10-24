"""
Advanced data quality assessment and validation for hyperspectral data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseProcessor
from .utils import validate_spectral_data


class DataQualityAssessor(BaseProcessor):
    """
    Advanced data quality assessment for hyperspectral crop data.
    
    Provides comprehensive analysis of data quality including outlier detection,
    missing value analysis, spectral signature validation, and statistical tests.
    """
    
    def __init__(self):
        """Initialize the data quality assessor."""
        super().__init__()
        self.assessment_results = {}
    
    def assess_missing_values(self, df: pd.DataFrame, spectral_columns: List[str]) -> Dict[str, Any]:
        """
        Comprehensive missing value analysis.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            
        Returns:
            Dictionary with missing value analysis results
        """
        self.logger.info("Analyzing missing values...")
        
        results = {
            'total_missing': 0,
            'missing_by_column': {},
            'missing_by_sample': {},
            'missing_patterns': {},
            'recommendations': []
        }
        
        # Overall missing value statistics
        spectral_data = df[spectral_columns]
        total_values = spectral_data.size
        total_missing = spectral_data.isnull().sum().sum()
        
        results['total_missing'] = int(total_missing)
        results['missing_percentage'] = float(total_missing / total_values * 100)
        
        # Missing values by column (wavelength)
        missing_by_col = spectral_data.isnull().sum()
        results['missing_by_column'] = {
            col: int(count) for col, count in missing_by_col.items() if count > 0
        }
        
        # Missing values by sample
        missing_by_sample = spectral_data.isnull().sum(axis=1)
        results['missing_by_sample'] = {
            'samples_with_missing': int((missing_by_sample > 0).sum()),
            'max_missing_per_sample': int(missing_by_sample.max()),
            'mean_missing_per_sample': float(missing_by_sample.mean())
        }
        
        # Identify samples with high missing percentages
        high_missing_threshold = len(spectral_columns) * 0.1  # 10%
        high_missing_samples = missing_by_sample > high_missing_threshold
        results['high_missing_samples'] = int(high_missing_samples.sum())
        
        # Missing value patterns
        if total_missing > 0:
            # Check for systematic missing patterns
            missing_matrix = spectral_data.isnull()
            
            # Consecutive missing values (spectral gaps)
            consecutive_gaps = []
            for idx in range(len(missing_matrix)):
                sample_missing = missing_matrix.iloc[idx]
                if sample_missing.any():
                    # Find consecutive missing regions
                    missing_indices = np.where(sample_missing)[0]
                    if len(missing_indices) > 1:
                        gaps = np.diff(missing_indices)
                        consecutive_gaps.extend(gaps[gaps == 1])
            
            results['missing_patterns']['consecutive_gaps'] = len(consecutive_gaps)
            
            # Edge effects (missing at spectrum edges)
            edge_missing = (
                missing_matrix.iloc[:, :5].any(axis=1).sum() +  # First 5 bands
                missing_matrix.iloc[:, -5:].any(axis=1).sum()   # Last 5 bands
            )
            results['missing_patterns']['edge_missing'] = int(edge_missing)
        
        # Generate recommendations
        if results['missing_percentage'] > 5:
            results['recommendations'].append("High missing value percentage detected")
        
        if results['high_missing_samples'] > 0:
            results['recommendations'].append(
                f"Consider removing {results['high_missing_samples']} samples with >10% missing values"
            )
        
        if results['missing_patterns'].get('consecutive_gaps', 0) > 0:
            results['recommendations'].append("Use interpolation for consecutive missing values")
        
        self.logger.info(f"Missing value analysis completed: {results['missing_percentage']:.2f}% missing")
        return results
    
    def detect_outliers(self, df: pd.DataFrame, spectral_columns: List[str], 
                       methods: List[str] = None) -> Dict[str, Any]:
        """
        Multi-method outlier detection for spectral data.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            methods: List of outlier detection methods to use
            
        Returns:
            Dictionary with outlier detection results
        """
        if methods is None:
            methods = ['iqr', 'zscore', 'isolation_forest']
        
        self.logger.info(f"Detecting outliers using methods: {methods}")
        
        spectral_data = df[spectral_columns].fillna(0)  # Fill NaN for outlier detection
        results = {
            'methods_used': methods,
            'outliers_by_method': {},
            'consensus_outliers': [],
            'outlier_statistics': {}
        }
        
        outlier_masks = {}
        
        # IQR method
        if 'iqr' in methods:
            outlier_masks['iqr'] = self._detect_outliers_iqr(spectral_data)
            results['outliers_by_method']['iqr'] = int(outlier_masks['iqr'].sum())
        
        # Z-score method
        if 'zscore' in methods:
            outlier_masks['zscore'] = self._detect_outliers_zscore(spectral_data)
            results['outliers_by_method']['zscore'] = int(outlier_masks['zscore'].sum())
        
        # Isolation Forest method
        if 'isolation_forest' in methods:
            outlier_masks['isolation_forest'] = self._detect_outliers_isolation_forest(spectral_data)
            results['outliers_by_method']['isolation_forest'] = int(outlier_masks['isolation_forest'].sum())
        
        # Consensus outliers (detected by multiple methods)
        if len(outlier_masks) > 1:
            consensus_mask = sum(outlier_masks.values()) >= 2  # At least 2 methods agree
            results['consensus_outliers'] = np.where(consensus_mask)[0].tolist()
            results['outlier_statistics']['consensus_count'] = len(results['consensus_outliers'])
        
        # Outlier characteristics analysis
        if outlier_masks:
            any_outlier_mask = np.logical_or.reduce(list(outlier_masks.values()))
            outlier_data = spectral_data[any_outlier_mask]
            normal_data = spectral_data[~any_outlier_mask]
            
            if len(outlier_data) > 0 and len(normal_data) > 0:
                results['outlier_statistics'].update({
                    'outlier_mean_reflectance': float(outlier_data.mean().mean()),
                    'normal_mean_reflectance': float(normal_data.mean().mean()),
                    'outlier_std_reflectance': float(outlier_data.std().mean()),
                    'normal_std_reflectance': float(normal_data.std().mean())
                })
        
        self.logger.info(f"Outlier detection completed. Methods: {results['outliers_by_method']}")
        return results
    
    def _detect_outliers_iqr(self, data: pd.DataFrame, factor: float = 1.5) -> np.ndarray:
        """Detect outliers using Interquartile Range method."""
        outliers = np.zeros(len(data), dtype=bool)
        
        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outliers |= col_outliers
        
        return outliers
    
    def _detect_outliers_zscore(self, data: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
        outliers = (z_scores > threshold).any(axis=1)
        return outliers
    
    def _detect_outliers_isolation_forest(self, data: pd.DataFrame, 
                                        contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        # Use a subset of features for efficiency with high-dimensional data
        n_features = min(50, data.shape[1])  # Use max 50 features
        feature_indices = np.linspace(0, data.shape[1]-1, n_features, dtype=int)
        subset_data = data.iloc[:, feature_indices]
        
        # Standardize data for Isolation Forest
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(subset_data)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        outlier_labels = iso_forest.fit_predict(scaled_data)
        
        # Convert to boolean mask (True for outliers)
        outliers = outlier_labels == -1
        return outliers
    
    def validate_spectral_ranges(self, df: pd.DataFrame, spectral_columns: List[str]) -> Dict[str, Any]:
        """
        Validate spectral reflectance ranges and physical constraints.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            
        Returns:
            Dictionary with spectral range validation results
        """
        self.logger.info("Validating spectral ranges...")
        
        spectral_data = df[spectral_columns]
        results = {
            'valid_ranges': True,
            'range_violations': {},
            'statistics': {},
            'recommendations': []
        }
        
        # Check for negative reflectance (physically impossible)
        negative_mask = spectral_data < 0
        negative_count = negative_mask.sum().sum()
        
        if negative_count > 0:
            results['valid_ranges'] = False
            results['range_violations']['negative_reflectance'] = {
                'count': int(negative_count),
                'percentage': float(negative_count / spectral_data.size * 100),
                'affected_samples': int(negative_mask.any(axis=1).sum())
            }
            results['recommendations'].append("Remove or correct negative reflectance values")
        
        # Check for extremely high reflectance (>100% is suspicious)
        high_mask = spectral_data > 100
        high_count = high_mask.sum().sum()
        
        if high_count > 0:
            results['range_violations']['high_reflectance'] = {
                'count': int(high_count),
                'percentage': float(high_count / spectral_data.size * 100),
                'affected_samples': int(high_mask.any(axis=1).sum()),
                'max_value': float(spectral_data.max().max())
            }
            results['recommendations'].append("Investigate reflectance values >100%")
        
        # Check for unrealistic reflectance (>200% definitely wrong)
        extreme_mask = spectral_data > 200
        extreme_count = extreme_mask.sum().sum()
        
        if extreme_count > 0:
            results['valid_ranges'] = False
            results['range_violations']['extreme_reflectance'] = {
                'count': int(extreme_count),
                'affected_samples': int(extreme_mask.any(axis=1).sum())
            }
            results['recommendations'].append("Remove samples with extreme reflectance values")
        
        # Overall statistics
        results['statistics'] = {
            'min_reflectance': float(spectral_data.min().min()),
            'max_reflectance': float(spectral_data.max().max()),
            'mean_reflectance': float(spectral_data.mean().mean()),
            'std_reflectance': float(spectral_data.std().mean())
        }
        
        # Check for constant spectra (likely measurement errors)
        constant_spectra = 0
        for idx in range(len(spectral_data)):
            spectrum = spectral_data.iloc[idx].dropna()
            if len(spectrum) > 0 and spectrum.std() < 0.01:  # Very low variation
                constant_spectra += 1
        
        if constant_spectra > 0:
            results['range_violations']['constant_spectra'] = constant_spectra
            results['recommendations'].append(f"Investigate {constant_spectra} samples with constant spectra")
        
        self.logger.info(f"Spectral range validation completed. Valid: {results['valid_ranges']}")
        return results
    
    def analyze_crop_stage_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze consistency of crop-stage combinations and temporal logic.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Dictionary with crop-stage consistency analysis
        """
        self.logger.info("Analyzing crop-stage consistency...")
        
        results = {
            'crop_stage_combinations': {},
            'unusual_combinations': [],
            'temporal_consistency': {},
            'recommendations': []
        }
        
        # Analyze crop-stage combinations
        crop_stage_counts = df.groupby(['Crop', 'Stage']).size().reset_index(name='count')
        results['crop_stage_combinations'] = crop_stage_counts.to_dict('records')
        
        # Identify unusual combinations (very few samples)
        min_samples_threshold = 5
        unusual = crop_stage_counts[crop_stage_counts['count'] < min_samples_threshold]
        results['unusual_combinations'] = unusual.to_dict('records')
        
        if len(unusual) > 0:
            results['recommendations'].append(
                f"Found {len(unusual)} crop-stage combinations with <{min_samples_threshold} samples"
            )
        
        # Temporal consistency (if date information is available)
        if 'Month' in df.columns and 'Year' in df.columns:
            # Check for logical stage progression within crops
            temporal_analysis = df.groupby(['Crop', 'Year', 'Month'])['Stage'].apply(list).reset_index()
            
            # Define expected stage order (approximate)
            stage_order = {
                'Emerge_VEarly': 1,
                'Early_Mid': 2,
                'Late': 3,
                'Critical': 4,  # Disease can occur at any stage
                'Mature_Senesc': 5,
                'Harvest': 6
            }
            
            inconsistent_temporal = 0
            for _, group in temporal_analysis.iterrows():
                stages = group['Stage']
                if len(stages) > 1:
                    # Check if stages follow logical order
                    stage_numbers = [stage_order.get(stage, 0) for stage in stages]
                    if not all(stage_numbers[i] <= stage_numbers[i+1] for i in range(len(stage_numbers)-1)):
                        inconsistent_temporal += 1
            
            results['temporal_consistency'] = {
                'inconsistent_sequences': inconsistent_temporal,
                'total_sequences': len(temporal_analysis)
            }
            
            if inconsistent_temporal > 0:
                results['recommendations'].append(
                    f"Found {inconsistent_temporal} temporally inconsistent stage sequences"
                )
        
        self.logger.info("Crop-stage consistency analysis completed")
        return results
    
    def generate_quality_report(self, df: pd.DataFrame, spectral_columns: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            
        Returns:
            Comprehensive quality assessment report
        """
        self.logger.info("Generating comprehensive data quality report...")
        
        report = {
            'overall_quality_score': 0.0,
            'quality_components': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'detailed_results': {}
        }
        
        # Run all quality assessments
        missing_results = self.assess_missing_values(df, spectral_columns)
        outlier_results = self.detect_outliers(df, spectral_columns)
        range_results = self.validate_spectral_ranges(df, spectral_columns)
        consistency_results = self.analyze_crop_stage_consistency(df)
        
        # Store detailed results
        report['detailed_results'] = {
            'missing_values': missing_results,
            'outliers': outlier_results,
            'spectral_ranges': range_results,
            'crop_stage_consistency': consistency_results
        }
        
        # Calculate quality scores for each component (0-100)
        quality_scores = {}
        
        # Missing values score (100 - missing_percentage)
        quality_scores['missing_values'] = max(0, 100 - missing_results['missing_percentage'])
        
        # Outlier score (based on consensus outliers)
        outlier_percentage = (outlier_results.get('outlier_statistics', {}).get('consensus_count', 0) / 
                            len(df) * 100)
        quality_scores['outliers'] = max(0, 100 - outlier_percentage * 2)  # Penalize outliers more
        
        # Range validity score
        quality_scores['spectral_ranges'] = 100 if range_results['valid_ranges'] else 50
        
        # Consistency score
        unusual_combinations = len(consistency_results['unusual_combinations'])
        total_combinations = len(consistency_results['crop_stage_combinations'])
        consistency_percentage = (unusual_combinations / max(total_combinations, 1)) * 100
        quality_scores['consistency'] = max(0, 100 - consistency_percentage * 10)
        
        report['quality_components'] = quality_scores
        
        # Calculate overall quality score (weighted average)
        weights = {'missing_values': 0.3, 'outliers': 0.3, 'spectral_ranges': 0.3, 'consistency': 0.1}
        overall_score = sum(score * weights[component] for component, score in quality_scores.items())
        report['overall_quality_score'] = overall_score
        
        # Identify critical issues
        if missing_results['missing_percentage'] > 10:
            report['critical_issues'].append("High percentage of missing values (>10%)")
        
        if not range_results['valid_ranges']:
            report['critical_issues'].append("Invalid spectral ranges detected")
        
        if outlier_percentage > 20:
            report['critical_issues'].append("High percentage of outliers (>20%)")
        
        # Compile all recommendations
        all_recommendations = (
            missing_results.get('recommendations', []) +
            outlier_results.get('recommendations', []) +
            range_results.get('recommendations', []) +
            consistency_results.get('recommendations', [])
        )
        report['recommendations'] = list(set(all_recommendations))  # Remove duplicates
        
        # Generate warnings
        if overall_score < 70:
            report['warnings'].append("Overall data quality score is below 70%")
        
        if quality_scores['missing_values'] < 80:
            report['warnings'].append("High missing value percentage detected")
        
        if quality_scores['outliers'] < 80:
            report['warnings'].append("Significant number of outliers detected")
        
        self.logger.info(f"Quality report generated. Overall score: {overall_score:.1f}/100")
        return report
    
    def process(self, df: pd.DataFrame, spectral_columns: List[str]) -> Dict[str, Any]:
        """
        Main processing method for data quality assessment.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            
        Returns:
            Comprehensive quality assessment report
        """
        return self.generate_quality_report(df, spectral_columns)