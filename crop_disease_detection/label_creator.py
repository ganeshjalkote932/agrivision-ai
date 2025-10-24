"""
Disease label creation module for hyperspectral crop data.

This module implements strategies to create binary disease labels from crop metadata
and spectral signatures, enabling supervised learning for disease detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging

from .base import BaseProcessor
from .config import DataConfig


class DiseaseLabelCreator(BaseProcessor):
    """
    Creates binary disease labels from crop metadata and spectral analysis.
    
    This class implements multiple strategies to identify diseased vs healthy samples:
    1. Crop stage-based labeling (primary method)
    2. Spectral anomaly detection (validation method)
    3. Statistical outlier analysis (supplementary method)
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the disease label creator.
        
        Args:
            config: DataConfig instance with labeling parameters
        """
        super().__init__()
        self.config = config
        self.labeling_strategy = "stage_based"  # Default strategy
        self.label_statistics = {}
        
    def create_stage_based_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create disease labels based on crop growth stages.
        
        The strategy assumes that certain growth stages indicate disease or stress:
        - 'Critical': Diseased (1) - indicates critical growth issues
        - 'Mature_Senesc': Diseased (1) - mature/senescent crops may show disease
        - 'Emerge_VEarly': Healthy (0) - early emergence is typically healthy
        - 'Early_Mid': Healthy (0) - early to mid growth is typically healthy
        - 'Late': Healthy (0) - late growth stage is typically healthy
        - 'Harvest': Healthy (0) - harvest-ready crops are typically healthy
        
        Args:
            df: DataFrame containing crop data with 'Stage' column
            
        Returns:
            Tuple of (labels array, labeling statistics)
        """
        self.logger.info("Creating disease labels based on crop growth stages...")
        
        if 'Stage' not in df.columns:
            raise ValueError("'Stage' column not found in dataset")
        
        # Define stage-to-label mapping
        stage_mapping = {
            'Critical': 1,        # Diseased - critical growth issues
            'Mature_Senesc': 1,   # Diseased - senescence may indicate disease
            'Emerge_VEarly': 0,   # Healthy - early emergence
            'Early_Mid': 0,       # Healthy - early to mid growth
            'Late': 0,            # Healthy - late growth
            'Harvest': 0          # Healthy - harvest ready
        }
        
        # Create labels array
        labels = np.zeros(len(df), dtype=int)
        stage_counts = {}
        
        for stage, label in stage_mapping.items():
            mask = df['Stage'] == stage
            labels[mask] = label
            stage_counts[stage] = {
                'count': int(mask.sum()),
                'label': label,
                'percentage': float(mask.sum() / len(df) * 100)
            }
        
        # Handle unknown stages
        known_stages = set(stage_mapping.keys())
        actual_stages = set(df['Stage'].unique())
        unknown_stages = actual_stages - known_stages
        
        if unknown_stages:
            self.logger.warning(f"Unknown stages found: {unknown_stages}")
            # Default unknown stages to healthy (0)
            for stage in unknown_stages:
                mask = df['Stage'] == stage
                labels[mask] = 0
                stage_counts[stage] = {
                    'count': int(mask.sum()),
                    'label': 0,
                    'percentage': float(mask.sum() / len(df) * 100),
                    'unknown': True
                }
        
        # Calculate overall statistics
        diseased_count = np.sum(labels == 1)
        healthy_count = np.sum(labels == 0)
        
        statistics = {
            'total_samples': len(labels),
            'diseased_samples': int(diseased_count),
            'healthy_samples': int(healthy_count),
            'diseased_percentage': float(diseased_count / len(labels) * 100),
            'healthy_percentage': float(healthy_count / len(labels) * 100),
            'class_balance_ratio': float(healthy_count / diseased_count) if diseased_count > 0 else float('inf'),
            'stage_breakdown': stage_counts,
            'labeling_method': 'stage_based'
        }
        
        self.logger.info(f"Stage-based labeling completed:")
        self.logger.info(f"  Diseased samples: {diseased_count} ({statistics['diseased_percentage']:.1f}%)")
        self.logger.info(f"  Healthy samples: {healthy_count} ({statistics['healthy_percentage']:.1f}%)")
        self.logger.info(f"  Class balance ratio: {statistics['class_balance_ratio']:.2f}:1")
        
        return labels, statistics
    
    def create_spectral_anomaly_labels(self, df: pd.DataFrame, spectral_columns: List[str],
                                     contamination: float = 0.1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create disease labels based on spectral anomaly detection.
        
        This method uses statistical analysis of spectral signatures to identify
        samples that deviate significantly from normal patterns, which may indicate disease.
        
        Args:
            df: DataFrame containing spectral data
            spectral_columns: List of spectral column names
            contamination: Expected fraction of anomalies (diseased samples)
            
        Returns:
            Tuple of (labels array, labeling statistics)
        """
        self.logger.info("Creating disease labels based on spectral anomaly detection...")
        
        spectral_data = df[spectral_columns].fillna(0)  # Fill NaN for analysis
        
        # Method 1: Isolation Forest for anomaly detection
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Use a subset of features for efficiency
        n_features = min(50, len(spectral_columns))
        feature_indices = np.linspace(0, len(spectral_columns)-1, n_features, dtype=int)
        subset_data = spectral_data.iloc[:, feature_indices]
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(subset_data)
        
        # Detect anomalies
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        labels = (anomaly_labels == -1).astype(int)  # Convert to binary (1 = diseased)
        
        # Method 2: Statistical outlier detection (validation)
        z_scores = np.abs(stats.zscore(spectral_data, axis=0, nan_policy='omit'))
        statistical_outliers = (z_scores > 3).any(axis=1)
        
        # Method 3: Spectral signature deviation analysis
        mean_spectrum = spectral_data.mean()
        std_spectrum = spectral_data.std()
        
        # Calculate deviation from mean spectrum for each sample
        deviations = []
        for idx in range(len(spectral_data)):
            sample_spectrum = spectral_data.iloc[idx]
            # Calculate normalized deviation
            deviation = np.sqrt(np.mean(((sample_spectrum - mean_spectrum) / std_spectrum) ** 2))
            deviations.append(deviation)
        
        deviations = np.array(deviations)
        deviation_threshold = np.percentile(deviations, 90)  # Top 10% as anomalies
        deviation_outliers = deviations > deviation_threshold
        
        # Combine methods for consensus
        consensus_labels = (
            labels + 
            statistical_outliers.astype(int) + 
            deviation_outliers.astype(int)
        )
        # Samples identified by 2+ methods are considered diseased
        final_labels = (consensus_labels >= 2).astype(int)
        
        # Calculate statistics
        diseased_count = np.sum(final_labels == 1)
        healthy_count = np.sum(final_labels == 0)
        
        statistics = {
            'total_samples': len(final_labels),
            'diseased_samples': int(diseased_count),
            'healthy_samples': int(healthy_count),
            'diseased_percentage': float(diseased_count / len(final_labels) * 100),
            'healthy_percentage': float(healthy_count / len(final_labels) * 100),
            'class_balance_ratio': float(healthy_count / diseased_count) if diseased_count > 0 else float('inf'),
            'method_breakdown': {
                'isolation_forest': int(np.sum(labels == 1)),
                'statistical_outliers': int(np.sum(statistical_outliers)),
                'deviation_outliers': int(np.sum(deviation_outliers)),
                'consensus_threshold': 2
            },
            'labeling_method': 'spectral_anomaly'
        }
        
        self.logger.info(f"Spectral anomaly labeling completed:")
        self.logger.info(f"  Diseased samples: {diseased_count} ({statistics['diseased_percentage']:.1f}%)")
        self.logger.info(f"  Healthy samples: {healthy_count} ({statistics['healthy_percentage']:.1f}%)")
        
        return final_labels, statistics
    
    def validate_labels_consistency(self, stage_labels: np.ndarray, spectral_labels: np.ndarray,
                                  df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate consistency between stage-based and spectral-based labels.
        
        Args:
            stage_labels: Labels from stage-based method
            spectral_labels: Labels from spectral anomaly method
            df: Original DataFrame
            
        Returns:
            Dictionary with consistency analysis results
        """
        self.logger.info("Validating label consistency between methods...")
        
        # Calculate agreement metrics
        agreement = (stage_labels == spectral_labels)
        agreement_rate = np.mean(agreement)
        
        # Analyze disagreements
        disagreements = ~agreement
        disagreement_indices = np.where(disagreements)[0]
        
        # Breakdown by crop type
        crop_consistency = {}
        for crop in df['Crop'].unique():
            crop_mask = df['Crop'] == crop
            crop_agreement = np.mean(agreement[crop_mask])
            crop_consistency[crop] = {
                'agreement_rate': float(crop_agreement),
                'total_samples': int(crop_mask.sum()),
                'disagreements': int(np.sum(disagreements[crop_mask]))
            }
        
        # Breakdown by stage
        stage_consistency = {}
        for stage in df['Stage'].unique():
            stage_mask = df['Stage'] == stage
            stage_agreement = np.mean(agreement[stage_mask])
            stage_consistency[stage] = {
                'agreement_rate': float(stage_agreement),
                'total_samples': int(stage_mask.sum()),
                'disagreements': int(np.sum(disagreements[stage_mask]))
            }
        
        # Confusion matrix between methods
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(stage_labels, spectral_labels)
        
        validation_results = {
            'overall_agreement_rate': float(agreement_rate),
            'total_disagreements': int(np.sum(disagreements)),
            'disagreement_percentage': float(np.mean(disagreements) * 100),
            'crop_consistency': crop_consistency,
            'stage_consistency': stage_consistency,
            'confusion_matrix': cm.tolist(),
            'disagreement_indices': disagreement_indices.tolist()
        }
        
        self.logger.info(f"Label consistency validation completed:")
        self.logger.info(f"  Agreement rate: {agreement_rate:.3f} ({agreement_rate*100:.1f}%)")
        self.logger.info(f"  Disagreements: {np.sum(disagreements)} samples")
        
        return validation_results
    
    def create_hybrid_labels(self, df: pd.DataFrame, spectral_columns: List[str],
                           stage_weight: float = 0.7, spectral_weight: float = 0.3) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create hybrid disease labels combining stage-based and spectral-based methods.
        
        Args:
            df: DataFrame containing crop data
            spectral_columns: List of spectral column names
            stage_weight: Weight for stage-based labels (0-1)
            spectral_weight: Weight for spectral-based labels (0-1)
            
        Returns:
            Tuple of (labels array, labeling statistics)
        """
        self.logger.info("Creating hybrid disease labels...")
        
        # Normalize weights
        total_weight = stage_weight + spectral_weight
        stage_weight = stage_weight / total_weight
        spectral_weight = spectral_weight / total_weight
        
        # Get labels from both methods
        stage_labels, stage_stats = self.create_stage_based_labels(df)
        spectral_labels, spectral_stats = self.create_spectral_anomaly_labels(df, spectral_columns)
        
        # Validate consistency
        consistency_results = self.validate_labels_consistency(stage_labels, spectral_labels, df)
        
        # Create weighted combination
        weighted_scores = (stage_labels * stage_weight + spectral_labels * spectral_weight)
        
        # Apply threshold for final binary labels
        threshold = 0.5
        hybrid_labels = (weighted_scores >= threshold).astype(int)
        
        # Calculate final statistics
        diseased_count = np.sum(hybrid_labels == 1)
        healthy_count = np.sum(hybrid_labels == 0)
        
        statistics = {
            'total_samples': len(hybrid_labels),
            'diseased_samples': int(diseased_count),
            'healthy_samples': int(healthy_count),
            'diseased_percentage': float(diseased_count / len(hybrid_labels) * 100),
            'healthy_percentage': float(healthy_count / len(hybrid_labels) * 100),
            'class_balance_ratio': float(healthy_count / diseased_count) if diseased_count > 0 else float('inf'),
            'labeling_method': 'hybrid',
            'weights': {
                'stage_weight': stage_weight,
                'spectral_weight': spectral_weight
            },
            'component_statistics': {
                'stage_based': stage_stats,
                'spectral_based': spectral_stats
            },
            'consistency_analysis': consistency_results,
            'threshold': threshold
        }
        
        self.logger.info(f"Hybrid labeling completed:")
        self.logger.info(f"  Diseased samples: {diseased_count} ({statistics['diseased_percentage']:.1f}%)")
        self.logger.info(f"  Healthy samples: {healthy_count} ({statistics['healthy_percentage']:.1f}%)")
        self.logger.info(f"  Method agreement: {consistency_results['overall_agreement_rate']:.3f}")
        
        return hybrid_labels, statistics
    
    def analyze_label_quality(self, labels: np.ndarray, df: pd.DataFrame, 
                            spectral_columns: List[str]) -> Dict[str, Any]:
        """
        Analyze the quality and characteristics of created labels.
        
        Args:
            labels: Binary disease labels
            df: Original DataFrame
            spectral_columns: List of spectral column names
            
        Returns:
            Dictionary with label quality analysis
        """
        self.logger.info("Analyzing label quality...")
        
        spectral_data = df[spectral_columns]
        
        # Separate diseased and healthy samples
        diseased_mask = labels == 1
        healthy_mask = labels == 0
        
        diseased_spectra = spectral_data[diseased_mask]
        healthy_spectra = spectral_data[healthy_mask]
        
        quality_analysis = {
            'class_separability': {},
            'spectral_characteristics': {},
            'crop_distribution': {},
            'stage_distribution': {},
            'recommendations': []
        }
        
        # Class separability analysis
        if len(diseased_spectra) > 0 and len(healthy_spectra) > 0:
            # Calculate mean spectra for each class
            diseased_mean = diseased_spectra.mean()
            healthy_mean = healthy_spectra.mean()
            
            # Calculate spectral distance between classes
            spectral_distance = np.sqrt(np.mean((diseased_mean - healthy_mean) ** 2))
            
            # Calculate within-class variance
            diseased_var = diseased_spectra.var().mean()
            healthy_var = healthy_spectra.var().mean()
            
            # Fisher's discriminant ratio (between-class / within-class variance)
            fisher_ratio = spectral_distance / (np.sqrt(diseased_var + healthy_var) + 1e-8)
            
            quality_analysis['class_separability'] = {
                'spectral_distance': float(spectral_distance),
                'diseased_variance': float(diseased_var),
                'healthy_variance': float(healthy_var),
                'fisher_ratio': float(fisher_ratio),
                'separability_score': min(fisher_ratio / 2.0, 1.0)  # Normalized score
            }
        
        # Spectral characteristics
        quality_analysis['spectral_characteristics'] = {
            'diseased_mean_reflectance': float(diseased_spectra.mean().mean()) if len(diseased_spectra) > 0 else 0,
            'healthy_mean_reflectance': float(healthy_spectra.mean().mean()) if len(healthy_spectra) > 0 else 0,
            'diseased_std_reflectance': float(diseased_spectra.std().mean()) if len(diseased_spectra) > 0 else 0,
            'healthy_std_reflectance': float(healthy_spectra.std().mean()) if len(healthy_spectra) > 0 else 0
        }
        
        # Distribution by crop type
        for crop in df['Crop'].unique():
            crop_mask = df['Crop'] == crop
            crop_labels = labels[crop_mask]
            quality_analysis['crop_distribution'][crop] = {
                'total': int(crop_mask.sum()),
                'diseased': int(np.sum(crop_labels == 1)),
                'healthy': int(np.sum(crop_labels == 0)),
                'diseased_rate': float(np.mean(crop_labels == 1))
            }
        
        # Distribution by growth stage
        for stage in df['Stage'].unique():
            stage_mask = df['Stage'] == stage
            stage_labels = labels[stage_mask]
            quality_analysis['stage_distribution'][stage] = {
                'total': int(stage_mask.sum()),
                'diseased': int(np.sum(stage_labels == 1)),
                'healthy': int(np.sum(stage_labels == 0)),
                'diseased_rate': float(np.mean(stage_labels == 1))
            }
        
        # Generate recommendations
        diseased_percentage = np.mean(labels == 1) * 100
        
        if diseased_percentage < 5:
            quality_analysis['recommendations'].append("Very low disease rate - consider class balancing techniques")
        elif diseased_percentage > 50:
            quality_analysis['recommendations'].append("High disease rate - validate labeling strategy")
        
        if 'class_separability' in quality_analysis and quality_analysis['class_separability']['fisher_ratio'] < 1.0:
            quality_analysis['recommendations'].append("Low class separability - consider feature selection or different labeling strategy")
        
        # Check for crop-specific issues
        crop_disease_rates = [info['diseased_rate'] for info in quality_analysis['crop_distribution'].values()]
        if max(crop_disease_rates) - min(crop_disease_rates) > 0.5:
            quality_analysis['recommendations'].append("Large variation in disease rates across crops - consider crop-specific models")
        
        self.logger.info("Label quality analysis completed")
        return quality_analysis
    
    def process(self, df: pd.DataFrame, spectral_columns: List[str], 
               method: str = "stage_based") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main processing method to create disease labels.
        
        Args:
            df: DataFrame containing crop data
            spectral_columns: List of spectral column names
            method: Labeling method ("stage_based", "spectral_anomaly", "hybrid")
            
        Returns:
            Tuple of (labels array, comprehensive statistics)
        """
        self.logger.info(f"Creating disease labels using method: {method}")
        
        if method == "stage_based":
            labels, stats = self.create_stage_based_labels(df)
        elif method == "spectral_anomaly":
            labels, stats = self.create_spectral_anomaly_labels(df, spectral_columns)
        elif method == "hybrid":
            labels, stats = self.create_hybrid_labels(df, spectral_columns)
        else:
            raise ValueError(f"Unknown labeling method: {method}")
        
        # Analyze label quality
        quality_analysis = self.analyze_label_quality(labels, df, spectral_columns)
        
        # Combine statistics
        comprehensive_stats = {
            **stats,
            'quality_analysis': quality_analysis,
            'method_used': method
        }
        
        # Store for future reference
        self.label_statistics = comprehensive_stats
        
        return labels, comprehensive_stats