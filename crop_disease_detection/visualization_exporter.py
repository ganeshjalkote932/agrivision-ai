"""
Visualization export system for hyperspectral crop disease detection.

This module implements automatic saving of all plots with descriptive filenames
and organized directory structure for comprehensive results storage.
"""

import os
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json

from .base import BaseProcessor
from .logger import ProcessLogger


class VisualizationExporter(BaseProcessor):
    """
    Comprehensive visualization export system.
    
    Automatically saves all plots with descriptive filenames and creates
    organized directory structure for results storage and documentation.
    """
    
    def __init__(self, output_dir: str = "results/visualizations",
                 session_id: Optional[str] = None,
                 logger: Optional[ProcessLogger] = None):
        """
        Initialize the visualization exporter.
        
        Args:
            output_dir: Base directory for saving visualizations
            session_id: Optional session identifier for organization
            logger: Optional ProcessLogger instance
        """
        super().__init__()
        
        # Setup directories
        self.base_dir = Path(output_dir)
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / f"session_{self.session_id}"
        
        # Create organized subdirectories
        self.directories = {
            'data_analysis': self.session_dir / "01_data_analysis",
            'preprocessing': self.session_dir / "02_preprocessing", 
            'model_training': self.session_dir / "03_model_training",
            'model_evaluation': self.session_dir / "04_model_evaluation",
            'feature_analysis': self.session_dir / "05_feature_analysis",
            'deployment': self.session_dir / "06_deployment",
            'summary': self.session_dir / "07_summary"
        }
        
        # Create all directories
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.process_logger = logger
        
        # Track exported visualizations
        self.exported_plots = {}
        self.plot_counter = 0
        
        # Default plot settings
        self.default_dpi = 300
        self.default_format = 'png'
        self.default_bbox_inches = 'tight'
        
        # Create session info
        self._create_session_info()
        
        if self.process_logger:
            self.process_logger.log_file_operation(
                "create", str(self.session_dir), True,
                session_id=self.session_id,
                directories_created=len(self.directories)
            )
    
    def _create_session_info(self):
        """Create session information file."""
        session_info = {
            'session_id': self.session_id,
            'created_at': datetime.datetime.now().isoformat(),
            'base_directory': str(self.base_dir),
            'session_directory': str(self.session_dir),
            'subdirectories': {name: str(path) for name, path in self.directories.items()},
            'export_settings': {
                'default_dpi': self.default_dpi,
                'default_format': self.default_format,
                'default_bbox_inches': self.default_bbox_inches
            }
        }
        
        info_file = self.session_dir / "session_info.json"
        with open(info_file, 'w') as f:
            json.dump(session_info, f, indent=2, default=str)
    
    def _generate_filename(self, plot_type: str, description: str, 
                          category: str = "general") -> Tuple[str, Path]:
        """
        Generate descriptive filename for a plot.
        
        Args:
            plot_type: Type of plot (histogram, scatter, line, etc.)
            description: Description of what the plot shows
            category: Category for organization
            
        Returns:
            Tuple of (filename, full_path)
        """
        self.plot_counter += 1
        
        # Clean description for filename
        clean_description = "".join(c if c.isalnum() or c in "._-" else "_" for c in description)
        clean_description = clean_description.strip("_").lower()
        
        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        
        # Generate filename
        filename = f"{self.plot_counter:03d}_{plot_type}_{clean_description}_{timestamp}.{self.default_format}"
        
        # Determine directory based on category
        if category in self.directories:
            directory = self.directories[category]
        else:
            directory = self.session_dir
        
        full_path = directory / filename
        
        return filename, full_path
    
    def save_plot(self, fig: matplotlib.figure.Figure, plot_type: str, 
                 description: str, category: str = "general",
                 metadata: Optional[Dict[str, Any]] = None,
                 dpi: Optional[int] = None,
                 format: Optional[str] = None) -> str:
        """
        Save a matplotlib figure with organized naming and metadata.
        
        Args:
            fig: Matplotlib figure to save
            plot_type: Type of plot (histogram, scatter, line, etc.)
            description: Description of what the plot shows
            category: Category for organization
            metadata: Additional metadata to store
            dpi: DPI for saving (uses default if None)
            format: File format (uses default if None)
            
        Returns:
            Path to saved file
        """
        filename, full_path = self._generate_filename(plot_type, description, category)
        
        # Save the plot
        save_kwargs = {
            'dpi': dpi or self.default_dpi,
            'format': format or self.default_format,
            'bbox_inches': self.default_bbox_inches
        }
        
        try:
            fig.savefig(full_path, **save_kwargs)
            
            # Store plot information
            plot_info = {
                'filename': filename,
                'full_path': str(full_path),
                'plot_type': plot_type,
                'description': description,
                'category': category,
                'created_at': datetime.datetime.now().isoformat(),
                'file_size': os.path.getsize(full_path),
                'save_settings': save_kwargs,
                'metadata': metadata or {}
            }
            
            self.exported_plots[filename] = plot_info
            
            if self.process_logger:
                self.process_logger.log_file_operation(
                    "save_plot", str(full_path), True,
                    plot_type=plot_type,
                    description=description,
                    category=category
                )
            
            return str(full_path)
            
        except Exception as e:
            if self.process_logger:
                self.process_logger.log_file_operation(
                    "save_plot", str(full_path), False,
                    error=str(e)
                )
            raise
    
    def save_current_figure(self, plot_type: str, description: str, 
                           category: str = "general", **kwargs) -> str:
        """
        Save the current matplotlib figure.
        
        Args:
            plot_type: Type of plot
            description: Description of the plot
            category: Category for organization
            **kwargs: Additional arguments for save_plot
            
        Returns:
            Path to saved file
        """
        fig = plt.gcf()
        return self.save_plot(fig, plot_type, description, category, **kwargs)
    
    def create_data_analysis_plots(self, df: pd.DataFrame, target_column: str = None) -> List[str]:
        """
        Create and save standard data analysis plots.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column for classification plots
            
        Returns:
            List of saved plot paths
        """
        saved_plots = []
        
        # Dataset overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Dataset shape info
        axes[0, 0].text(0.1, 0.8, f"Dataset Shape: {df.shape}", fontsize=14, transform=axes[0, 0].transAxes)
        axes[0, 0].text(0.1, 0.6, f"Columns: {len(df.columns)}", fontsize=12, transform=axes[0, 0].transAxes)
        axes[0, 0].text(0.1, 0.4, f"Rows: {len(df)}", fontsize=12, transform=axes[0, 0].transAxes)
        axes[0, 0].text(0.1, 0.2, f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", 
                       fontsize=12, transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("Dataset Overview")
        axes[0, 0].axis('off')
        
        # Missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0].head(10)
            axes[0, 1].bar(range(len(missing_data)), missing_data.values)
            axes[0, 1].set_xticks(range(len(missing_data)))
            axes[0, 1].set_xticklabels(missing_data.index, rotation=45)
            axes[0, 1].set_title("Missing Values by Column")
        else:
            axes[0, 1].text(0.5, 0.5, "No Missing Values", ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title("Missing Values")
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title("Data Types Distribution")
        
        # Target distribution (if provided)
        if target_column and target_column in df.columns:
            target_counts = df[target_column].value_counts()
            axes[1, 1].bar(target_counts.index, target_counts.values)
            axes[1, 1].set_title(f"Target Distribution ({target_column})")
            axes[1, 1].set_xlabel(target_column)
            axes[1, 1].set_ylabel("Count")
        else:
            axes[1, 1].text(0.5, 0.5, "No Target Column", ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title("Target Distribution")
        
        plt.tight_layout()
        saved_plots.append(self.save_plot(fig, "overview", "dataset_analysis", "data_analysis"))
        plt.close(fig)
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Sample columns if too many
            if len(numeric_cols) > 50:
                numeric_cols = numeric_cols[:50]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Feature Correlation Matrix")
            plt.tight_layout()
            saved_plots.append(self.save_plot(fig, "heatmap", "correlation_matrix", "data_analysis"))
            plt.close(fig)
        
        return saved_plots
    
    def create_preprocessing_plots(self, before_data: np.ndarray, after_data: np.ndarray,
                                 feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Create and save preprocessing comparison plots.
        
        Args:
            before_data: Data before preprocessing
            after_data: Data after preprocessing
            feature_names: Names of features
            
        Returns:
            List of saved plot paths
        """
        saved_plots = []
        
        # Distribution comparison
        n_features_to_plot = min(6, before_data.shape[1])
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(n_features_to_plot):
            ax = axes[i]
            
            # Plot histograms
            ax.hist(before_data[:, i], alpha=0.7, label='Before', bins=30, density=True)
            ax.hist(after_data[:, i], alpha=0.7, label='After', bins=30, density=True)
            
            feature_name = feature_names[i] if feature_names else f"Feature {i}"
            ax.set_title(f"{feature_name}")
            ax.legend()
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
        
        plt.suptitle("Feature Distributions: Before vs After Preprocessing")
        plt.tight_layout()
        saved_plots.append(self.save_plot(fig, "histogram", "preprocessing_comparison", "preprocessing"))
        plt.close(fig)
        
        # Statistics comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean comparison
        before_means = np.mean(before_data, axis=0)
        after_means = np.mean(after_data, axis=0)
        
        x_pos = np.arange(min(20, len(before_means)))
        ax1.bar(x_pos - 0.2, before_means[:len(x_pos)], 0.4, label='Before', alpha=0.7)
        ax1.bar(x_pos + 0.2, after_means[:len(x_pos)], 0.4, label='After', alpha=0.7)
        ax1.set_title("Feature Means Comparison")
        ax1.set_xlabel("Feature Index")
        ax1.set_ylabel("Mean Value")
        ax1.legend()
        
        # Standard deviation comparison
        before_stds = np.std(before_data, axis=0)
        after_stds = np.std(after_data, axis=0)
        
        ax2.bar(x_pos - 0.2, before_stds[:len(x_pos)], 0.4, label='Before', alpha=0.7)
        ax2.bar(x_pos + 0.2, after_stds[:len(x_pos)], 0.4, label='After', alpha=0.7)
        ax2.set_title("Feature Standard Deviations Comparison")
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Standard Deviation")
        ax2.legend()
        
        plt.tight_layout()
        saved_plots.append(self.save_plot(fig, "bar", "preprocessing_statistics", "preprocessing"))
        plt.close(fig)
        
        return saved_plots
    
    def create_model_training_plots(self, training_history: Dict[str, List[float]]) -> List[str]:
        """
        Create and save model training plots.
        
        Args:
            training_history: Dictionary with training metrics over epochs/iterations
            
        Returns:
            List of saved plot paths
        """
        saved_plots = []
        
        if not training_history:
            return saved_plots
        
        # Training curves
        fig, axes = plt.subplots(1, len(training_history), figsize=(6*len(training_history), 6))
        if len(training_history) == 1:
            axes = [axes]
        
        for i, (metric_name, values) in enumerate(training_history.items()):
            ax = axes[i]
            ax.plot(values, marker='o', linewidth=2, markersize=4)
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ax.set_xlabel("Epoch/Iteration")
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        saved_plots.append(self.save_plot(fig, "line", "training_curves", "model_training"))
        plt.close(fig)
        
        return saved_plots
    
    def export_plot_summary(self) -> str:
        """
        Export a summary of all created plots.
        
        Returns:
            Path to summary file
        """
        summary = {
            'session_id': self.session_id,
            'export_timestamp': datetime.datetime.now().isoformat(),
            'total_plots': len(self.exported_plots),
            'plots_by_category': {},
            'plots_by_type': {},
            'plots': self.exported_plots
        }
        
        # Aggregate by category and type
        for plot_info in self.exported_plots.values():
            category = plot_info['category']
            plot_type = plot_info['plot_type']
            
            summary['plots_by_category'][category] = summary['plots_by_category'].get(category, 0) + 1
            summary['plots_by_type'][plot_type] = summary['plots_by_type'].get(plot_type, 0) + 1
        
        # Save summary
        summary_file = self.directories['summary'] / "plot_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if self.process_logger:
            self.process_logger.log_file_operation(
                "export_summary", str(summary_file), True,
                total_plots=len(self.exported_plots)
            )
        
        return str(summary_file)
    
    def process(self, data: Any) -> Any:
        """
        Process method required by BaseProcessor.
        
        Args:
            data: Input data (figure, plot info, etc.)
            
        Returns:
            Processed results
        """
        if isinstance(data, matplotlib.figure.Figure):
            # Auto-save figure with generic info
            return self.save_plot(data, "figure", "auto_saved", "general")
        elif isinstance(data, dict) and 'figure' in data:
            # Save figure with provided metadata
            fig = data['figure']
            plot_type = data.get('plot_type', 'figure')
            description = data.get('description', 'auto_saved')
            category = data.get('category', 'general')
            return self.save_plot(fig, plot_type, description, category, data.get('metadata'))
        
        return data