"""
Data splitting and validation module for hyperspectral crop disease detection.

This module implements robust data splitting strategies with stratification,
cross-validation, and comprehensive validation to ensure reliable model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, StratifiedShuffleSplit,
    cross_val_score, validation_curve
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

from .base import BaseProcessor
from .config import DataConfig


class DataSplitter(BaseProcessor):
    """
    Comprehensive data splitting and validation system for hyperspectral data.
    
    Provides stratified splitting, cross-validation, and validation analysis
    to ensure robust model training and evaluation.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data splitter.
        
        Args:
            config: DataConfig instance with splitting parameters
        """
        super().__init__()
        self.config = config
        self.split_info = {}
        self.validation_results = {}
        
    def stratified_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                                  test_size: float = None, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified train-test split preserving class distribution.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Fraction of data for testing (default from config)
            random_state: Random seed (default from config)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.config.test_size
        if random_state is None:
            random_state = self.config.random_state
            
        self.logger.info(f"Performing stratified train-test split (test_size={test_size})...")
        
        # Check class distribution before split
        class_counts = Counter(y)
        self.logger.info(f"Original class distribution: {dict(class_counts)}")
        
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Verify class distribution preservation
        train_class_counts = Counter(y_train)
        test_class_counts = Counter(y_test)
        
        self.logger.info(f"Training set class distribution: {dict(train_class_counts)}")
        self.logger.info(f"Test set class distribution: {dict(test_class_counts)}")
        
        # Calculate class distribution ratios
        train_ratios = {cls: count/len(y_train) for cls, count in train_class_counts.items()}
        test_ratios = {cls: count/len(y_test) for cls, count in test_class_counts.items()}
        original_ratios = {cls: count/len(y) for cls, count in class_counts.items()}
        
        # Store split information
        self.split_info = {
            'total_samples': len(y),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'test_size': test_size,
            'random_state': random_state,
            'original_distribution': dict(class_counts),
            'train_distribution': dict(train_class_counts),
            'test_distribution': dict(test_class_counts),
            'original_ratios': original_ratios,
            'train_ratios': train_ratios,
            'test_ratios': test_ratios
        }
        
        # Validate stratification quality
        max_ratio_diff = max(abs(train_ratios[cls] - original_ratios[cls]) for cls in original_ratios)
        if max_ratio_diff > 0.02:  # 2% tolerance
            self.logger.warning(f"Stratification quality concern: max ratio difference = {max_ratio_diff:.3f}")
        else:
            self.logger.info(f"✓ Excellent stratification: max ratio difference = {max_ratio_diff:.3f}")
        
        self.logger.info(f"Split completed: {len(X_train)} training, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_cross_validation_folds(self, X: np.ndarray, y: np.ndarray, 
                                    n_folds: int = 5, random_state: int = None) -> StratifiedKFold:
        """
        Create stratified cross-validation folds.
        
        Args:
            X: Feature matrix
            y: Target labels
            n_folds: Number of CV folds
            random_state: Random seed
            
        Returns:
            StratifiedKFold object
        """
        if random_state is None:
            random_state = self.config.random_state
            
        self.logger.info(f"Creating {n_folds}-fold stratified cross-validation...")
        
        cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )
        
        # Validate fold quality
        fold_distributions = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            fold_y_train = y[train_idx]
            fold_y_val = y[val_idx]
            
            train_dist = Counter(fold_y_train)
            val_dist = Counter(fold_y_val)
            
            fold_distributions.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_distribution': dict(train_dist),
                'val_distribution': dict(val_dist)
            })
        
        # Store CV information
        self.split_info['cv_folds'] = fold_distributions
        self.split_info['n_folds'] = n_folds
        
        self.logger.info(f"Cross-validation folds created successfully")
        return cv
    
    def validate_data_leakage(self, X_train: np.ndarray, X_test: np.ndarray, 
                            tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Check for potential data leakage between train and test sets.
        
        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix
            tolerance: Numerical tolerance for duplicate detection
            
        Returns:
            Dictionary with leakage analysis results
        """
        self.logger.info("Checking for data leakage between train and test sets...")
        
        leakage_results = {
            'has_leakage': False,
            'duplicate_samples': 0,
            'similarity_analysis': {},
            'recommendations': []
        }
        
        # Check for exact duplicates (optimized for large datasets)
        duplicates = 0
        duplicate_indices = []
        
        # Use sampling for large datasets to speed up detection
        max_samples_to_check = 1000
        if len(X_test) > max_samples_to_check:
            test_indices = np.random.choice(len(X_test), max_samples_to_check, replace=False)
            X_test_sample = X_test[test_indices]
        else:
            test_indices = np.arange(len(X_test))
            X_test_sample = X_test
        
        if len(X_train) > max_samples_to_check:
            train_indices = np.random.choice(len(X_train), max_samples_to_check, replace=False)
            X_train_sample = X_train[train_indices]
        else:
            train_indices = np.arange(len(X_train))
            X_train_sample = X_train
        
        for i, test_sample in enumerate(X_test_sample):
            for j, train_sample in enumerate(X_train_sample):
                if np.allclose(test_sample, train_sample, atol=tolerance):
                    duplicates += 1
                    duplicate_indices.append((test_indices[i], train_indices[j]))
                    break
        
        leakage_results['duplicate_samples'] = duplicates
        leakage_results['duplicate_indices'] = duplicate_indices
        
        if duplicates > 0:
            leakage_results['has_leakage'] = True
            leakage_results['recommendations'].append(f"Remove {duplicates} duplicate samples")
            self.logger.warning(f"⚠️ Data leakage detected: {duplicates} duplicate samples")
        else:
            self.logger.info("✓ No exact duplicates found between train and test sets")
        
        # Check for high similarity (potential near-duplicates)
        if len(X_test) <= 1000 and len(X_train) <= 1000:  # Only for smaller datasets
            high_similarity_count = 0
            similarity_threshold = 0.99
            
            # Sample-based similarity check
            sample_size = min(100, len(X_test), len(X_train))
            test_sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            train_sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
            
            for i in test_sample_indices:
                test_sample = X_test[i]
                for j in train_sample_indices:
                    train_sample = X_train[j]
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(test_sample, train_sample)
                    norms = np.linalg.norm(test_sample) * np.linalg.norm(train_sample)
                    
                    if norms > 0:
                        similarity = dot_product / norms
                        if similarity > similarity_threshold:
                            high_similarity_count += 1
                            break
            
            leakage_results['similarity_analysis'] = {
                'high_similarity_pairs': high_similarity_count,
                'similarity_threshold': similarity_threshold,
                'samples_checked': sample_size
            }
            
            if high_similarity_count > sample_size * 0.1:  # >10% high similarity
                leakage_results['recommendations'].append("Investigate high similarity samples")
                self.logger.warning(f"⚠️ High similarity detected in {high_similarity_count} sample pairs")
        
        # Statistical distribution comparison
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)
        mean_correlation = np.corrcoef(train_mean, test_mean)[0, 1]
        
        leakage_results['similarity_analysis']['mean_correlation'] = float(mean_correlation)
        
        if mean_correlation > 0.99:
            leakage_results['recommendations'].append("Very high correlation between train/test means")
        
        self.logger.info(f"Data leakage validation completed")
        return leakage_results
    
    def analyze_split_quality(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive analysis of split quality and characteristics.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary with split quality analysis
        """
        self.logger.info("Analyzing split quality and characteristics...")
        
        quality_analysis = {
            'size_analysis': {},
            'distribution_analysis': {},
            'feature_analysis': {},
            'quality_score': 0.0,
            'recommendations': []
        }
        
        # Size analysis
        total_samples = len(X_train) + len(X_test)
        train_percentage = len(X_train) / total_samples * 100
        test_percentage = len(X_test) / total_samples * 100
        
        quality_analysis['size_analysis'] = {
            'total_samples': total_samples,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_percentage': train_percentage,
            'test_percentage': test_percentage
        }
        
        # Distribution analysis
        train_class_dist = Counter(y_train)
        test_class_dist = Counter(y_test)
        
        # Calculate distribution similarity
        train_ratios = np.array([count/len(y_train) for count in train_class_dist.values()])
        test_ratios = np.array([count/len(y_test) for count in test_class_dist.values()])
        
        distribution_similarity = 1 - np.mean(np.abs(train_ratios - test_ratios))
        
        quality_analysis['distribution_analysis'] = {
            'train_class_distribution': dict(train_class_dist),
            'test_class_distribution': dict(test_class_dist),
            'distribution_similarity': float(distribution_similarity)
        }
        
        # Feature analysis
        train_feature_means = np.mean(X_train, axis=0)
        test_feature_means = np.mean(X_test, axis=0)
        train_feature_stds = np.std(X_train, axis=0)
        test_feature_stds = np.std(X_test, axis=0)
        
        mean_correlation = np.corrcoef(train_feature_means, test_feature_means)[0, 1]
        std_correlation = np.corrcoef(train_feature_stds, test_feature_stds)[0, 1]
        
        quality_analysis['feature_analysis'] = {
            'mean_correlation': float(mean_correlation),
            'std_correlation': float(std_correlation),
            'train_feature_range': [float(X_train.min()), float(X_train.max())],
            'test_feature_range': [float(X_test.min()), float(X_test.max())]
        }
        
        # Calculate overall quality score (0-100)
        quality_score = 0
        
        # Size balance (target 80/20)
        size_score = 100 - abs(train_percentage - 80) * 2  # Penalty for deviation from 80/20
        quality_score += size_score * 0.2
        
        # Distribution similarity
        dist_score = distribution_similarity * 100
        quality_score += dist_score * 0.4
        
        # Feature correlation (should be high but not too high)
        feature_score = min(mean_correlation * 100, 95)  # Cap at 95 to avoid perfect correlation
        quality_score += feature_score * 0.3
        
        # No data leakage bonus
        leakage_results = self.validate_data_leakage(X_train, X_test)
        if not leakage_results['has_leakage']:
            quality_score += 10  # 10 point bonus
        
        quality_analysis['quality_score'] = min(quality_score, 100)
        
        # Generate recommendations
        if size_score < 90:
            quality_analysis['recommendations'].append("Consider adjusting train/test split ratio")
        
        if dist_score < 90:
            quality_analysis['recommendations'].append("Class distribution imbalance between train/test")
        
        if mean_correlation < 0.8:
            quality_analysis['recommendations'].append("Low feature correlation - check for systematic differences")
        
        if leakage_results['has_leakage']:
            quality_analysis['recommendations'].extend(leakage_results['recommendations'])
        
        # Store results
        self.validation_results['split_quality'] = quality_analysis
        self.validation_results['data_leakage'] = leakage_results
        
        self.logger.info(f"Split quality analysis completed. Quality score: {quality_score:.1f}/100")
        return quality_analysis
    
    def create_validation_visualization(self, X_train: np.ndarray, X_test: np.ndarray,
                                      y_train: np.ndarray, y_test: np.ndarray,
                                      save_plot: bool = True, output_dir: str = "results/plots") -> plt.Figure:
        """
        Create comprehensive visualization of data split characteristics.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            save_plot: Whether to save the plot
            output_dir: Directory to save plots
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating data split visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Split Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sample size comparison
        sizes = [len(X_train), len(X_test)]
        labels = ['Training', 'Test']
        colors = ['skyblue', 'lightcoral']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Train/Test Split Ratio')
        
        # 2. Class distribution comparison
        train_class_counts = Counter(y_train)
        test_class_counts = Counter(y_test)
        
        classes = list(train_class_counts.keys())
        train_counts = [train_class_counts[cls] for cls in classes]
        test_counts = [test_class_counts[cls] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, train_counts, width, label='Training', color='skyblue')
        axes[0, 1].bar(x + width/2, test_counts, width, label='Test', color='lightcoral')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Sample Count')
        axes[0, 1].set_title('Class Distribution Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'Class {cls}' for cls in classes])
        axes[0, 1].legend()
        
        # 3. Feature mean comparison (sample of features)
        n_features_to_plot = min(20, X_train.shape[1])
        feature_indices = np.linspace(0, X_train.shape[1]-1, n_features_to_plot, dtype=int)
        
        train_means = np.mean(X_train[:, feature_indices], axis=0)
        test_means = np.mean(X_test[:, feature_indices], axis=0)
        
        axes[0, 2].plot(feature_indices, train_means, 'o-', label='Training', color='skyblue', markersize=4)
        axes[0, 2].plot(feature_indices, test_means, 's-', label='Test', color='lightcoral', markersize=4)
        axes[0, 2].set_xlabel('Feature Index')
        axes[0, 2].set_ylabel('Mean Value')
        axes[0, 2].set_title('Feature Means Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature correlation analysis
        if X_train.shape[1] > 1:
            # Sample features for correlation analysis
            sample_features = min(50, X_train.shape[1])
            sample_indices = np.random.choice(X_train.shape[1], sample_features, replace=False)
            
            train_corr = np.corrcoef(X_train[:, sample_indices].T)
            test_corr = np.corrcoef(X_test[:, sample_indices].T)
            
            # Plot correlation difference
            corr_diff = np.abs(train_corr - test_corr)
            im = axes[1, 0].imshow(corr_diff, cmap='Reds', aspect='auto')
            axes[1, 0].set_title('Feature Correlation Difference')
            axes[1, 0].set_xlabel('Feature Index')
            axes[1, 0].set_ylabel('Feature Index')
            plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        
        # 5. PCA visualization
        if X_train.shape[1] > 2:
            from sklearn.decomposition import PCA
            
            # Combine data for consistent PCA
            X_combined = np.vstack([X_train, X_test])
            y_combined = np.hstack([y_train, y_test])
            split_labels = ['Train'] * len(X_train) + ['Test'] * len(X_test)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_combined)
            
            # Plot by split
            train_mask = np.array(split_labels) == 'Train'
            test_mask = np.array(split_labels) == 'Test'
            
            axes[1, 1].scatter(X_pca[train_mask, 0], X_pca[train_mask, 1], 
                              c='skyblue', alpha=0.6, s=20, label='Training')
            axes[1, 1].scatter(X_pca[test_mask, 0], X_pca[test_mask, 1], 
                              c='lightcoral', alpha=0.6, s=20, label='Test')
            
            axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[1, 1].set_title('PCA Visualization of Train/Test Split')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Quality summary
        quality_analysis = self.validation_results.get('split_quality', {})
        quality_score = quality_analysis.get('quality_score', 0)
        
        quality_text = f"""Split Quality Summary:
        
Quality Score: {quality_score:.1f}/100

Sample Sizes:
  Training: {len(X_train):,} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)
  Test: {len(X_test):,} ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)

Class Balance:
  Training: {dict(Counter(y_train))}
  Test: {dict(Counter(y_test))}

Data Leakage: {'✓ None' if not self.validation_results.get('data_leakage', {}).get('has_leakage', True) else '⚠ Detected'}
        """
        
        axes[1, 2].text(0.1, 0.9, quality_text, transform=axes[1, 2].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Quality Summary')
        
        plt.tight_layout()
        
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'data_split_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Data split visualization saved to: {plot_path}")
        
        return fig
    
    def get_split_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of data splitting results.
        
        Returns:
            Dictionary with complete split analysis
        """
        return {
            'split_info': self.split_info,
            'validation_results': self.validation_results
        }
    
    def process(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Main processing method for data splitting.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Perform stratified split
        X_train, X_test, y_train, y_test = self.stratified_train_test_split(X, y)
        
        # Analyze split quality
        self.analyze_split_quality(X_train, X_test, y_train, y_test)
        
        # Create visualization
        self.create_validation_visualization(X_train, X_test, y_train, y_test)
        
        return X_train, X_test, y_train, y_test