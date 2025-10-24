"""
Visualization tools for preprocessing pipeline analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os

from .base import BaseProcessor


class PreprocessingVisualizer(BaseProcessor):
    """
    Visualization tools for analyzing preprocessing effects and data transformations.
    """
    
    def __init__(self, output_dir: str = "results/plots/preprocessing"):
        """
        Initialize the preprocessing visualizer.
        
        Args:
            output_dir: Directory to save preprocessing plots
        """
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_preprocessing_comparison(self, X_original: np.ndarray, X_processed: np.ndarray,
                                   wavelengths: List[float], save_plot: bool = True) -> plt.Figure:
        """
        Create before/after comparison plots for preprocessing effects.
        
        Args:
            X_original: Original spectral data
            X_processed: Processed spectral data
            wavelengths: List of wavelength values
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating preprocessing comparison plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Preprocessing Effects Comparison', fontsize=16, fontweight='bold')
        
        # 1. Mean spectra comparison
        mean_original = np.mean(X_original, axis=0)
        mean_processed = np.mean(X_processed, axis=0)
        
        axes[0, 0].plot(wavelengths, mean_original, 'b-', linewidth=2, label='Original', alpha=0.7)
        axes[0, 0].plot(wavelengths, mean_processed, 'r-', linewidth=2, label='Processed', alpha=0.7)
        axes[0, 0].set_xlabel('Wavelength (nm)')
        axes[0, 0].set_ylabel('Mean Reflectance')
        axes[0, 0].set_title('Mean Spectral Signatures')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Standard deviation comparison
        std_original = np.std(X_original, axis=0)
        std_processed = np.std(X_processed, axis=0)
        
        axes[0, 1].plot(wavelengths, std_original, 'b-', linewidth=2, label='Original', alpha=0.7)
        axes[0, 1].plot(wavelengths, std_processed, 'r-', linewidth=2, label='Processed', alpha=0.7)
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_title('Spectral Variability')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution comparison (sample wavelength)
        mid_idx = len(wavelengths) // 2
        
        axes[0, 2].hist(X_original[:, mid_idx], bins=50, alpha=0.6, label='Original', color='blue', density=True)
        axes[0, 2].hist(X_processed[:, mid_idx], bins=50, alpha=0.6, label='Processed', color='red', density=True)
        axes[0, 2].set_xlabel(f'Reflectance at {wavelengths[mid_idx]:.0f} nm')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Value Distribution Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Sample spectra overlay (first 10 samples)
        n_samples = min(10, X_original.shape[0])
        for i in range(n_samples):
            axes[1, 0].plot(wavelengths, X_original[i, :], 'b-', alpha=0.3, linewidth=0.5)
            axes[1, 0].plot(wavelengths, X_processed[i, :], 'r-', alpha=0.3, linewidth=0.5)
        
        # Add mean lines
        axes[1, 0].plot(wavelengths, mean_original, 'b-', linewidth=3, label='Original Mean')
        axes[1, 0].plot(wavelengths, mean_processed, 'r-', linewidth=3, label='Processed Mean')
        axes[1, 0].set_xlabel('Wavelength (nm)')
        axes[1, 0].set_ylabel('Reflectance')
        axes[1, 0].set_title(f'Sample Spectra Overlay (n={n_samples})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Correlation matrix comparison
        # Sample wavelengths for correlation analysis
        sample_indices = np.linspace(0, len(wavelengths)-1, 20, dtype=int)
        
        corr_original = np.corrcoef(X_original[:, sample_indices].T)
        corr_processed = np.corrcoef(X_processed[:, sample_indices].T)
        
        im1 = axes[1, 1].imshow(corr_original, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1, 1].set_title('Original Data Correlation')
        axes[1, 1].set_xlabel('Wavelength Index')
        axes[1, 1].set_ylabel('Wavelength Index')
        
        im2 = axes[1, 2].imshow(corr_processed, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1, 2].set_title('Processed Data Correlation')
        axes[1, 2].set_xlabel('Wavelength Index')
        axes[1, 2].set_ylabel('Wavelength Index')
        
        # Add colorbars
        plt.colorbar(im1, ax=axes[1, 1], shrink=0.8)
        plt.colorbar(im2, ax=axes[1, 2], shrink=0.8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'preprocessing_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Preprocessing comparison plot saved to: {plot_path}")
        
        return fig
    
    def plot_missing_values_analysis(self, X_original: np.ndarray, X_imputed: np.ndarray,
                                   wavelengths: List[float], save_plot: bool = True) -> plt.Figure:
        """
        Visualize missing value patterns and imputation effects.
        
        Args:
            X_original: Original data with missing values
            X_imputed: Data after imputation
            wavelengths: List of wavelength values
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating missing values analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold')
        
        # 1. Missing values pattern
        missing_mask = np.isnan(X_original)
        
        if np.any(missing_mask):
            # Sample for visualization if too many samples
            sample_size = min(100, X_original.shape[0])
            sample_indices = np.random.choice(X_original.shape[0], sample_size, replace=False)
            missing_sample = missing_mask[sample_indices]
            
            im = axes[0, 0].imshow(missing_sample.T, cmap='Reds', aspect='auto')
            axes[0, 0].set_title(f'Missing Values Pattern (Sample of {sample_size})')
            axes[0, 0].set_xlabel('Sample Index')
            axes[0, 0].set_ylabel('Wavelength Index')
            plt.colorbar(im, ax=axes[0, 0])
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                           transform=axes[0, 0].transAxes, fontsize=14)
            axes[0, 0].set_title('Missing Values Pattern')
        
        # 2. Missing values by wavelength
        missing_by_wavelength = np.sum(missing_mask, axis=0)
        axes[0, 1].plot(wavelengths, missing_by_wavelength, 'ro-', markersize=3)
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('Number of Missing Values')
        axes[0, 1].set_title('Missing Values by Wavelength')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Imputation effect on mean spectrum
        if np.any(missing_mask):
            # Calculate mean excluding missing values
            mean_original = np.nanmean(X_original, axis=0)
            mean_imputed = np.mean(X_imputed, axis=0)
            
            axes[1, 0].plot(wavelengths, mean_original, 'b-', linewidth=2, label='Original (excl. NaN)')
            axes[1, 0].plot(wavelengths, mean_imputed, 'r-', linewidth=2, label='After Imputation')
            axes[1, 0].set_xlabel('Wavelength (nm)')
            axes[1, 0].set_ylabel('Mean Reflectance')
            axes[1, 0].set_title('Imputation Effect on Mean Spectrum')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Imputation quality assessment
            # Compare imputed values with nearby valid values
            imputed_mask = missing_mask & ~np.isnan(X_imputed)
            if np.any(imputed_mask):
                original_valid = X_original[~missing_mask]
                imputed_values = X_imputed[imputed_mask]
                
                axes[1, 1].hist(original_valid.flatten(), bins=50, alpha=0.6, 
                               label='Original Valid Values', density=True, color='blue')
                axes[1, 1].hist(imputed_values, bins=50, alpha=0.6, 
                               label='Imputed Values', density=True, color='red')
                axes[1, 1].set_xlabel('Reflectance Value')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].set_title('Imputed vs Original Value Distribution')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Imputed Values', ha='center', va='center',
                               transform=axes[1, 1].transAxes, fontsize=14)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Values to Impute', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 1].text(0.5, 0.5, 'No Missing Values to Impute', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=14)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'missing_values_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Missing values analysis plot saved to: {plot_path}")
        
        return fig
    
    def plot_outlier_detection_results(self, X: np.ndarray, outlier_mask: np.ndarray,
                                     wavelengths: List[float], save_plot: bool = True) -> plt.Figure:
        """
        Visualize outlier detection results.
        
        Args:
            X: Spectral data
            outlier_mask: Boolean mask indicating outliers
            wavelengths: List of wavelength values
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating outlier detection visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Outlier Detection Results', fontsize=16, fontweight='bold')
        
        # Separate normal and outlier data
        normal_data = X[~outlier_mask]
        outlier_data = X[outlier_mask]
        
        # 1. Spectral signatures comparison
        if len(normal_data) > 0:
            normal_mean = np.mean(normal_data, axis=0)
            normal_std = np.std(normal_data, axis=0)
            
            axes[0, 0].plot(wavelengths, normal_mean, 'b-', linewidth=2, label='Normal Mean')
            axes[0, 0].fill_between(wavelengths, 
                                   normal_mean - normal_std, 
                                   normal_mean + normal_std,
                                   alpha=0.3, color='blue', label='Normal ±1σ')
        
        if len(outlier_data) > 0:
            # Plot individual outlier spectra
            for i in range(min(10, len(outlier_data))):
                axes[0, 0].plot(wavelengths, outlier_data[i], 'r-', alpha=0.5, linewidth=1)
            
            outlier_mean = np.mean(outlier_data, axis=0)
            axes[0, 0].plot(wavelengths, outlier_mean, 'r-', linewidth=3, label='Outlier Mean')
        
        axes[0, 0].set_xlabel('Wavelength (nm)')
        axes[0, 0].set_ylabel('Reflectance')
        axes[0, 0].set_title('Normal vs Outlier Spectral Signatures')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Outlier distribution by sample index
        outlier_indices = np.where(outlier_mask)[0]
        axes[0, 1].scatter(outlier_indices, np.ones(len(outlier_indices)), 
                          c='red', alpha=0.6, s=20)
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Outlier Status')
        axes[0, 1].set_title(f'Outlier Distribution ({len(outlier_indices)} outliers)')
        axes[0, 1].set_ylim(0.5, 1.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. PCA visualization (if possible)
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            
            # Use subset of features for PCA
            n_components = min(2, X.shape[1])
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            # Plot normal and outlier points
            axes[1, 0].scatter(X_pca[~outlier_mask, 0], X_pca[~outlier_mask, 1], 
                              c='blue', alpha=0.6, s=20, label='Normal')
            if len(outlier_data) > 0:
                axes[1, 0].scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], 
                                  c='red', alpha=0.8, s=30, label='Outliers')
            
            axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[1, 0].set_title('PCA Visualization of Outliers')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient features for PCA', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
        
        # 4. Outlier statistics
        outlier_percentage = np.mean(outlier_mask) * 100
        
        stats_text = f"""Outlier Detection Summary:
        
Total Samples: {len(X):,}
Outliers Detected: {np.sum(outlier_mask):,}
Outlier Percentage: {outlier_percentage:.2f}%
Normal Samples: {np.sum(~outlier_mask):,}

Outlier Characteristics:
Mean Reflectance: {np.mean(outlier_data):.3f} (outliers)
Mean Reflectance: {np.mean(normal_data):.3f} (normal)
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Outlier Statistics')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'outlier_detection_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Outlier detection plot saved to: {plot_path}")
        
        return fig
    
    def plot_normalization_effects(self, X_original: np.ndarray, X_normalized: np.ndarray,
                                 wavelengths: List[float], method: str, save_plot: bool = True) -> plt.Figure:
        """
        Visualize the effects of normalization on spectral data.
        
        Args:
            X_original: Original spectral data
            X_normalized: Normalized spectral data
            wavelengths: List of wavelength values
            method: Normalization method used
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating normalization effects visualization ({method})...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Normalization Effects ({method.upper()})', fontsize=16, fontweight='bold')
        
        # 1. Distribution comparison (sample wavelength)
        mid_idx = len(wavelengths) // 2
        
        axes[0, 0].hist(X_original[:, mid_idx], bins=50, alpha=0.6, label='Original', 
                       color='blue', density=True)
        axes[0, 0].hist(X_normalized[:, mid_idx], bins=50, alpha=0.6, label='Normalized', 
                       color='red', density=True)
        axes[0, 0].set_xlabel(f'Value at {wavelengths[mid_idx]:.0f} nm')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Value Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Mean and std across wavelengths
        mean_original = np.mean(X_original, axis=0)
        std_original = np.std(X_original, axis=0)
        mean_normalized = np.mean(X_normalized, axis=0)
        std_normalized = np.std(X_normalized, axis=0)
        
        axes[0, 1].plot(wavelengths, mean_original, 'b-', linewidth=2, label='Original Mean')
        axes[0, 1].plot(wavelengths, mean_normalized, 'r-', linewidth=2, label='Normalized Mean')
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('Mean Value')
        axes[0, 1].set_title('Mean Values Across Wavelengths')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Standard deviation comparison
        axes[1, 0].plot(wavelengths, std_original, 'b-', linewidth=2, label='Original Std')
        axes[1, 0].plot(wavelengths, std_normalized, 'r-', linewidth=2, label='Normalized Std')
        axes[1, 0].set_xlabel('Wavelength (nm)')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].set_title('Standard Deviation Across Wavelengths')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sample spectra comparison
        n_samples = min(5, X_original.shape[0])
        for i in range(n_samples):
            axes[1, 1].plot(wavelengths, X_original[i, :], 'b-', alpha=0.6, linewidth=1)
            axes[1, 1].plot(wavelengths, X_normalized[i, :], 'r-', alpha=0.6, linewidth=1)
        
        # Add dummy lines for legend
        axes[1, 1].plot([], [], 'b-', linewidth=2, label='Original')
        axes[1, 1].plot([], [], 'r-', linewidth=2, label='Normalized')
        axes[1, 1].set_xlabel('Wavelength (nm)')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title(f'Sample Spectra Comparison (n={n_samples})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, f'normalization_effects_{method}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Normalization effects plot saved to: {plot_path}")
        
        return fig
    
    def create_preprocessing_report(self, preprocessing_results: Dict[str, Any], 
                                  save_plot: bool = True) -> List[plt.Figure]:
        """
        Create comprehensive preprocessing visualization report.
        
        Args:
            preprocessing_results: Dictionary containing preprocessing results
            save_plot: Whether to save plots
            
        Returns:
            List of matplotlib figure objects
        """
        self.logger.info("Creating comprehensive preprocessing report...")
        
        figures = []
        
        # Extract data from results
        X_original = preprocessing_results.get('X_original')
        X_processed = preprocessing_results.get('X_processed')
        wavelengths = preprocessing_results.get('wavelengths')
        outlier_mask = preprocessing_results.get('outlier_mask')
        
        if X_original is not None and X_processed is not None and wavelengths is not None:
            # Main comparison plot
            figures.append(self.plot_preprocessing_comparison(X_original, X_processed, wavelengths, save_plot))
            
            # Missing values analysis (if applicable)
            if 'X_before_imputation' in preprocessing_results:
                X_before_imputation = preprocessing_results['X_before_imputation']
                figures.append(self.plot_missing_values_analysis(X_before_imputation, X_processed, wavelengths, save_plot))
            
            # Outlier detection results
            if outlier_mask is not None:
                figures.append(self.plot_outlier_detection_results(X_original, outlier_mask, wavelengths, save_plot))
            
            # Normalization effects
            if 'normalization_method' in preprocessing_results:
                method = preprocessing_results['normalization_method']
                figures.append(self.plot_normalization_effects(X_original, X_processed, wavelengths, method, save_plot))
        
        self.logger.info(f"Created {len(figures)} preprocessing visualization plots")
        return figures
    
    def process(self, preprocessing_results: Dict[str, Any]) -> List[plt.Figure]:
        """
        Main processing method for preprocessing visualization.
        
        Args:
            preprocessing_results: Dictionary containing preprocessing results
            
        Returns:
            List of created figure objects
        """
        return self.create_preprocessing_report(preprocessing_results)