"""
Visualization module for hyperspectral data exploration and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path

from .base import BaseProcessor


class DataVisualizer(BaseProcessor):
    """
    Comprehensive visualization tools for hyperspectral crop data exploration.
    
    Provides methods to create informative plots for data distribution analysis,
    spectral signature visualization, and quality assessment.
    """
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        Initialize the data visualizer.
        
        Args:
            output_dir: Directory to save generated plots
        """
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    def plot_dataset_overview(self, df: pd.DataFrame, spectral_columns: List[str], 
                            save_plot: bool = True) -> plt.Figure:
        """
        Create an overview plot showing dataset characteristics.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            save_plot: Whether to save the plot to file
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating dataset overview plot...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperspectral Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Crop distribution
        crop_counts = df['Crop'].value_counts()
        axes[0, 0].pie(crop_counts.values, labels=crop_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Crop Distribution')
        
        # 2. Stage distribution
        stage_counts = df['Stage'].value_counts()
        axes[0, 1].bar(range(len(stage_counts)), stage_counts.values)
        axes[0, 1].set_xticks(range(len(stage_counts)))
        axes[0, 1].set_xticklabels(stage_counts.index, rotation=45, ha='right')
        axes[0, 1].set_title('Growth Stage Distribution')
        axes[0, 1].set_ylabel('Number of Samples')
        
        # 3. Crop-Stage heatmap
        crop_stage_matrix = pd.crosstab(df['Crop'], df['Stage'])
        sns.heatmap(crop_stage_matrix, annot=True, fmt='d', ax=axes[0, 2], cmap='Blues')
        axes[0, 2].set_title('Crop-Stage Combinations')
        axes[0, 2].set_xlabel('Growth Stage')
        axes[0, 2].set_ylabel('Crop Type')
        
        # 4. Spectral data statistics
        spectral_data = df[spectral_columns]
        wavelengths = [float(col[1:]) for col in spectral_columns]
        
        mean_spectrum = spectral_data.mean()
        std_spectrum = spectral_data.std()
        
        axes[1, 0].plot(wavelengths, mean_spectrum, 'b-', linewidth=2, label='Mean')
        axes[1, 0].fill_between(wavelengths, 
                               mean_spectrum - std_spectrum, 
                               mean_spectrum + std_spectrum, 
                               alpha=0.3, label='±1 Std')
        axes[1, 0].set_xlabel('Wavelength (nm)')
        axes[1, 0].set_ylabel('Reflectance')
        axes[1, 0].set_title('Average Spectral Signature')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Missing values heatmap
        missing_data = spectral_data.isnull()
        if missing_data.any().any():
            # Sample subset for visualization if too many samples
            sample_size = min(100, len(missing_data))
            sample_indices = np.random.choice(len(missing_data), sample_size, replace=False)
            missing_sample = missing_data.iloc[sample_indices]
            
            sns.heatmap(missing_sample.T, cbar=True, ax=axes[1, 1], 
                       cmap='Reds', yticklabels=False)
            axes[1, 1].set_title(f'Missing Values Pattern (Sample of {sample_size})')
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Wavelength')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Values', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=14, fontweight='bold')
            axes[1, 1].set_title('Missing Values Pattern')
        
        # 6. Data quality summary
        total_samples = len(df)
        total_features = len(spectral_columns)
        missing_percentage = spectral_data.isnull().sum().sum() / spectral_data.size * 100
        
        quality_text = f"""Dataset Summary:

Samples: {total_samples:,}
Spectral Bands: {total_features}
Wavelength Range: {min(wavelengths):.0f}-{max(wavelengths):.0f} nm
Missing Values: {missing_percentage:.2f}%
Crop Types: {df['Crop'].nunique()}
Growth Stages: {df['Stage'].nunique()}"""
        
        axes[1, 2].text(0.1, 0.9, quality_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Dataset Statistics')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'dataset_overview.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dataset overview plot saved to: {plot_path}")
        
        return fig
    
    def plot_spectral_signatures_by_crop(self, df: pd.DataFrame, spectral_columns: List[str],
                                       n_samples_per_crop: int = 10, save_plot: bool = True) -> plt.Figure:
        """
        Plot representative spectral signatures for each crop type.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            n_samples_per_crop: Number of samples to plot per crop
            save_plot: Whether to save the plot to file
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating spectral signatures by crop plot...")
        
        wavelengths = [float(col[1:]) for col in spectral_columns]
        crops = df['Crop'].unique()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Spectral Signatures by Crop Type', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, crop in enumerate(crops[:6]):  # Limit to 6 crops for visualization
            if i >= len(axes):
                break
                
            crop_data = df[df['Crop'] == crop][spectral_columns]
            
            # Sample random spectra for this crop
            n_samples = min(n_samples_per_crop, len(crop_data))
            sample_indices = np.random.choice(len(crop_data), n_samples, replace=False)
            sample_spectra = crop_data.iloc[sample_indices]
            
            # Plot individual spectra with transparency
            for idx in range(len(sample_spectra)):
                axes[i].plot(wavelengths, sample_spectra.iloc[idx], 
                           alpha=0.3, color='blue', linewidth=0.5)
            
            # Plot mean spectrum
            mean_spectrum = crop_data.mean()
            axes[i].plot(wavelengths, mean_spectrum, 'red', linewidth=2, label='Mean')
            
            # Plot confidence interval
            std_spectrum = crop_data.std()
            axes[i].fill_between(wavelengths,
                               mean_spectrum - std_spectrum,
                               mean_spectrum + std_spectrum,
                               alpha=0.2, color='red')
            
            axes[i].set_title(f'{crop} (n={len(crop_data)})')
            axes[i].set_xlabel('Wavelength (nm)')
            axes[i].set_ylabel('Reflectance')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(crops), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'spectral_signatures_by_crop.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Spectral signatures by crop plot saved to: {plot_path}")
        
        return fig
    
    def plot_spectral_signatures_by_stage(self, df: pd.DataFrame, spectral_columns: List[str],
                                        save_plot: bool = True) -> plt.Figure:
        """
        Plot average spectral signatures for each growth stage.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            save_plot: Whether to save the plot to file
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating spectral signatures by growth stage plot...")
        
        wavelengths = [float(col[1:]) for col in spectral_columns]
        stages = df['Stage'].unique()
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(stages)))
        
        for i, stage in enumerate(stages):
            stage_data = df[df['Stage'] == stage][spectral_columns]
            mean_spectrum = stage_data.mean()
            std_spectrum = stage_data.std()
            
            # Plot mean spectrum
            ax.plot(wavelengths, mean_spectrum, color=colors[i], linewidth=2, 
                   label=f'{stage} (n={len(stage_data)})')
            
            # Plot confidence interval
            ax.fill_between(wavelengths,
                          mean_spectrum - std_spectrum,
                          mean_spectrum + std_spectrum,
                          alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_title('Average Spectral Signatures by Growth Stage')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'spectral_signatures_by_stage.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Spectral signatures by stage plot saved to: {plot_path}")
        
        return fig
    
    def create_comprehensive_report_plots(self, df: pd.DataFrame, spectral_columns: List[str],
                                        quality_report: Optional[Dict[str, Any]] = None) -> List[plt.Figure]:
        """
        Create all visualization plots for comprehensive data exploration.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            quality_report: Optional quality assessment report
            
        Returns:
            List of matplotlib figure objects
        """
        self.logger.info("Creating comprehensive visualization report...")
        
        figures = []
        
        # Create all plots
        figures.append(self.plot_dataset_overview(df, spectral_columns))
        figures.append(self.plot_spectral_signatures_by_crop(df, spectral_columns))
        figures.append(self.plot_spectral_signatures_by_stage(df, spectral_columns))
        
        self.logger.info(f"Created {len(figures)} visualization plots")
        return figures
    
    def process(self, df: pd.DataFrame, spectral_columns: List[str]) -> List[plt.Figure]:
        """
        Main processing method for data visualization.
        
        Args:
            df: DataFrame containing the data
            spectral_columns: List of spectral column names
            
        Returns:
            List of created figure objects
        """
        return self.create_comprehensive_report_plots(df, spectral_columns)