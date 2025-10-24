#!/usr/bin/env python3
"""
Web Utilities for Crop Disease Detection Web Application
Handles visualizatiion and results export for web interface
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better web display
plt.style.use('default')
sns.set_palette("husl")

class WebVisualizationGenerator:
    """Generates visualizations optimized for web display."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """Initialize visualization generator."""
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'healthy': '#2ecc71',
            'diseased': '#e74c3c',
            'primary': '#3498db',
            'secondary': '#9b59b6',
            'warning': '#f39c12',
            'info': '#17a2b8'
        }
    
    def create_analysis_visualizations(self, 
                                     spectral_data: np.ndarray,
                                     processed_data: np.ndarray,
                                     predictions: List[Dict],
                                     metadata: Dict) -> Dict[str, str]:
        """
        Create comprehensive visualizations for web display.
        
        Returns:
            Dictionary of visualization names to base64 encoded images
        """
        visualizations = {}
        
        try:
            # 1. Spectral signature plot
            visualizations['spectral_signatures'] = self._create_spectral_signatures_plot(
                spectral_data, predictions, metadata
            )
            
            # 2. Prediction confidence distribution
            visualizations['confidence_distribution'] = self._create_confidence_distribution_plot(predictions)
            
            # 3. Disease probability heatmap (if spatial data)
            if spectral_data.ndim == 3:
                visualizations['disease_heatmap'] = self._create_disease_heatmap(
                    spectral_data, predictions, metadata
                )
            
            # 4. Data quality assessment
            visualizations['data_quality'] = self._create_data_quality_plot(spectral_data, metadata)
            
            # 5. Prediction summary
            visualizations['prediction_summary'] = self._create_prediction_summary_plot(predictions)
            
            # 6. Wavelength importance (if available)
            if len(predictions) > 0 and 'feature_importance' in predictions[0]:
                visualizations['feature_importance'] = self._create_feature_importance_plot(predictions[0])
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            # Create error visualization
            visualizations['error'] = self._create_error_plot(str(e))
        
        return visualizations
    
    def _create_spectral_signatures_plot(self, spectral_data: np.ndarray, 
                                       predictions: List[Dict], 
                                       metadata: Dict) -> str:
        """Create spectral signatures plot."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        try:
            # Generate wavelengths if not available
            if 'wavelengths' in metadata and metadata['wavelengths'] is not None:
                wavelengths = metadata['wavelengths']
                if isinstance(wavelengths, np.ndarray):
                    wavelengths = wavelengths.tolist()
            else:
                # Default wavelength range for hyperspectral data
                n_bands = spectral_data.shape[-1] if spectral_data.ndim > 1 else len(spectral_data)
                wavelengths = np.linspace(400, 2500, n_bands).tolist()
            
            # Plot spectral signatures
            if spectral_data.ndim == 1:
                # Single spectrum
                color = self.colors['diseased'] if predictions[0]['prediction'] == 1 else self.colors['healthy']
                label = f"Sample ({'Diseased' if predictions[0]['prediction'] == 1 else 'Healthy'})"
                ax.plot(wavelengths, spectral_data, color=color, linewidth=2, label=label)
                
            elif spectral_data.ndim == 2:
                # Multiple spectra - plot up to 10 samples
                n_samples = min(10, len(spectral_data))
                for i in range(n_samples):
                    pred = predictions[i] if i < len(predictions) else {'prediction': 0}
                    color = self.colors['diseased'] if pred['prediction'] == 1 else self.colors['healthy']
                    alpha = 0.7 if n_samples > 5 else 1.0
                    label = f"Sample {i+1} ({'D' if pred['prediction'] == 1 else 'H'})"
                    ax.plot(wavelengths, spectral_data[i], color=color, alpha=alpha, 
                           linewidth=1.5, label=label if i < 5 else "")
                    
            elif spectral_data.ndim == 3:
                # Hyperspectral cube - plot mean spectra for diseased and healthy regions
                h, w, bands = spectral_data.shape
                
                # Reshape predictions to spatial grid
                pred_grid = np.array([p['prediction'] for p in predictions]).reshape(h, w)
                
                # Calculate mean spectra
                diseased_mask = pred_grid == 1
                healthy_mask = pred_grid == 0
                
                if diseased_mask.any():
                    diseased_spectra = spectral_data[diseased_mask].mean(axis=0)
                    ax.plot(wavelengths, diseased_spectra, color=self.colors['diseased'], 
                           linewidth=3, label='Mean Diseased Spectrum')
                
                if healthy_mask.any():
                    healthy_spectra = spectral_data[healthy_mask].mean(axis=0)
                    ax.plot(wavelengths, healthy_spectra, color=self.colors['healthy'], 
                           linewidth=3, label='Mean Healthy Spectrum')
            
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Reflectance', fontsize=12)
            ax.set_title('Spectral Signatures', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Highlight important wavelength regions
            ax.axvspan(680, 750, alpha=0.1, color='red', label='Red Edge')
            ax.axvspan(750, 1300, alpha=0.1, color='green', label='NIR')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
        
        return self._fig_to_base64(fig)
    
    def _create_confidence_distribution_plot(self, predictions: List[Dict]) -> str:
        """Create confidence distribution plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        try:
            confidences = [p.get('confidence', 0) for p in predictions]
            disease_probs = [p.get('disease_probability', p.get('confidence', 0)) for p in predictions]
            
            # Confidence histogram
            ax1.hist(confidences, bins=20, alpha=0.7, color=self.colors['primary'], edgecolor='black')
            ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.3f}')
            ax1.set_xlabel('Prediction Confidence')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Confidence Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Disease probability distribution
            ax2.hist(disease_probs, bins=20, alpha=0.7, color=self.colors['warning'], edgecolor='black')
            ax2.axvline(np.mean(disease_probs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(disease_probs):.3f}')
            ax2.axvline(0.5, color='black', linestyle='-', alpha=0.5, label='Decision Threshold')
            ax2.set_xlabel('Disease Probability')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Disease Probability Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)}', transform=ax1.transAxes, ha='center')
            ax2.text(0.5, 0.5, f'Error: {str(e)}', transform=ax2.transAxes, ha='center')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_disease_heatmap(self, spectral_data: np.ndarray, 
                              predictions: List[Dict], 
                              metadata: Dict) -> str:
        """Create disease probability heatmap for spatial data."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        try:
            h, w, bands = spectral_data.shape
            
            # Reshape predictions to spatial grid
            pred_grid = np.array([p['prediction'] for p in predictions]).reshape(h, w)
            prob_grid = np.array([p.get('disease_probability', p.get('confidence', 0)) 
                                for p in predictions]).reshape(h, w)
            
            # Disease prediction map
            im1 = ax1.imshow(pred_grid, cmap='RdYlGn_r', vmin=0, vmax=1)
            ax1.set_title('Disease Prediction Map')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            plt.colorbar(im1, ax=ax1, label='Diseased (1) / Healthy (0)')
            
            # Disease probability heatmap
            im2 = ax2.imshow(prob_grid, cmap='Reds', vmin=0, vmax=1)
            ax2.set_title('Disease Probability Heatmap')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            plt.colorbar(im2, ax=ax2, label='Disease Probability')
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)}', transform=ax1.transAxes, ha='center')
            ax2.text(0.5, 0.5, f'Error: {str(e)}', transform=ax2.transAxes, ha='center')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_data_quality_plot(self, spectral_data: np.ndarray, metadata: Dict) -> str:
        """Create data quality assessment plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        
        try:
            # Flatten data for analysis
            if spectral_data.ndim > 2:
                flat_data = spectral_data.reshape(-1, spectral_data.shape[-1])
            elif spectral_data.ndim == 2:
                flat_data = spectral_data
            else:
                flat_data = spectral_data.reshape(1, -1)
            
            # 1. Data range per band
            band_means = np.mean(flat_data, axis=0)
            band_stds = np.std(flat_data, axis=0)
            bands = range(len(band_means))
            
            ax1.plot(bands, band_means, color=self.colors['primary'], label='Mean')
            ax1.fill_between(bands, band_means - band_stds, band_means + band_stds, 
                           alpha=0.3, color=self.colors['primary'], label='±1 Std')
            ax1.set_xlabel('Band Index')
            ax1.set_ylabel('Reflectance')
            ax1.set_title('Spectral Band Statistics')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Data distribution
            ax2.hist(flat_data.flatten(), bins=50, alpha=0.7, color=self.colors['secondary'])
            ax2.set_xlabel('Reflectance Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Data Value Distribution')
            ax2.grid(True, alpha=0.3)
            
            # 3. Missing/invalid data
            nan_counts = np.isnan(flat_data).sum(axis=0)
            inf_counts = np.isinf(flat_data).sum(axis=0)
            
            ax3.bar(bands, nan_counts, alpha=0.7, color=self.colors['warning'], label='NaN')
            ax3.bar(bands, inf_counts, alpha=0.7, color=self.colors['diseased'], 
                   bottom=nan_counts, label='Inf')
            ax3.set_xlabel('Band Index')
            ax3.set_ylabel('Count')
            ax3.set_title('Invalid Values per Band')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Correlation heatmap (sample of bands)
            n_bands_sample = min(20, flat_data.shape[1])
            step = max(1, flat_data.shape[1] // n_bands_sample)
            sample_bands = flat_data[:, ::step]
            
            corr_matrix = np.corrcoef(sample_bands.T)
            im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_title('Band Correlation Matrix (Sample)')
            ax4.set_xlabel('Band Index (Sampled)')
            ax4.set_ylabel('Band Index (Sampled)')
            plt.colorbar(im, ax=ax4)
            
        except Exception as e:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_prediction_summary_plot(self, predictions: List[Dict]) -> str:
        """Create prediction summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        
        try:
            # 1. Disease vs Healthy pie chart
            diseased_count = sum(1 for p in predictions if p['prediction'] == 1)
            healthy_count = len(predictions) - diseased_count
            
            labels = ['Healthy', 'Diseased']
            sizes = [healthy_count, diseased_count]
            colors = [self.colors['healthy'], self.colors['diseased']]
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'Disease Detection Results\n(Total: {len(predictions)} samples)')
            
            # 2. Confidence levels
            confidences = [p.get('confidence', 0) for p in predictions]
            conf_bins = ['Low (<0.6)', 'Medium (0.6-0.8)', 'High (>0.8)']
            conf_counts = [
                sum(1 for c in confidences if c < 0.6),
                sum(1 for c in confidences if 0.6 <= c <= 0.8),
                sum(1 for c in confidences if c > 0.8)
            ]
            
            bars = ax2.bar(conf_bins, conf_counts, color=[self.colors['diseased'], 
                                                         self.colors['warning'], 
                                                         self.colors['healthy']])
            ax2.set_title('Prediction Confidence Levels')
            ax2.set_ylabel('Number of Samples')
            
            # Add value labels on bars
            for bar, count in zip(bars, conf_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
            
            # 3. Disease probability vs confidence scatter
            disease_probs = [p.get('disease_probability', p.get('confidence', 0)) for p in predictions]
            colors_scatter = [self.colors['diseased'] if p['prediction'] == 1 else self.colors['healthy'] 
                            for p in predictions]
            
            ax3.scatter(confidences, disease_probs, c=colors_scatter, alpha=0.6, s=50)
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Correlation')
            ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Threshold')
            ax3.set_xlabel('Prediction Confidence')
            ax3.set_ylabel('Disease Probability')
            ax3.set_title('Confidence vs Disease Probability')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Summary statistics table
            ax4.axis('off')
            
            stats_data = [
                ['Total Samples', len(predictions)],
                ['Diseased Samples', diseased_count],
                ['Healthy Samples', healthy_count],
                ['Disease Rate', f'{(diseased_count/len(predictions)*100):.1f}%'],
                ['Avg Confidence', f'{np.mean(confidences):.3f}'],
                ['High Confidence', f'{sum(1 for c in confidences if c > 0.8)}'],
                ['Low Confidence', f'{sum(1 for c in confidences if c < 0.6)}']
            ]
            
            table = ax4.table(cellText=stats_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax4.set_title('Summary Statistics', pad=20)
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)}', transform=ax1.transAxes, ha='center')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_feature_importance_plot(self, prediction: Dict) -> str:
        """Create feature importance plot if available."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        try:
            importance = prediction['feature_importance']
            n_features = len(importance)
            
            # Show top 20 features
            top_n = min(20, n_features)
            top_indices = np.argsort(importance)[-top_n:]
            
            ax.barh(range(top_n), importance[top_indices], color=self.colors['primary'])
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Feature Index')
            ax.set_title('Top Feature Importance')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
        
        return self._fig_to_base64(fig)
    
    def _create_error_plot(self, error_message: str) -> str:
        """Create error visualization."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.text(0.5, 0.5, f'Visualization Error:\n{error_message}', 
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=self.dpi)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return image_base64


class ResultsExporter:
    """Exports analysis results in various formats."""
    
    def __init__(self):
        """Initialize results exporter."""
        pass
    
    def export_results(self, 
                      predictions: List[Dict],
                      visualizations: Dict[str, str],
                      metadata: Dict,
                      output_dir: str) -> Dict[str, str]:
        """
        Export complete analysis results.
        
        Returns:
            Dictionary of export file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        export_paths = {}
        
        try:
            # 1. Export predictions as CSV
            predictions_df = pd.DataFrame(predictions)
            csv_path = output_dir / 'predictions.csv'
            predictions_df.to_csv(csv_path, index=False)
            export_paths['predictions_csv'] = str(csv_path)
            
            # 2. Export predictions as JSON
            json_path = output_dir / 'predictions.json'
            with open(json_path, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            export_paths['predictions_json'] = str(json_path)
            
            # 3. Export metadata
            metadata_path = output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            export_paths['metadata'] = str(metadata_path)
            
            # 4. Export visualizations as PNG files
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            for viz_name, viz_base64 in visualizations.items():
                if viz_base64:
                    viz_path = viz_dir / f'{viz_name}.png'
                    with open(viz_path, 'wb') as f:
                        f.write(base64.b64decode(viz_base64))
                    export_paths[f'viz_{viz_name}'] = str(viz_path)
            
            # 5. Create summary report
            report_path = self._create_summary_report(
                predictions, metadata, output_dir / 'analysis_report.html'
            )
            export_paths['report'] = str(report_path)
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            export_paths['error'] = str(e)
        
        return export_paths
    
    def _create_summary_report(self, 
                              predictions: List[Dict], 
                              metadata: Dict, 
                              output_path: Path) -> Path:
        """Create HTML summary report."""
        
        # Calculate summary statistics
        diseased_count = sum(1 for p in predictions if p['prediction'] == 1)
        healthy_count = len(predictions) - diseased_count
        confidences = [p.get('confidence', 0) for p in predictions]
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crop Disease Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .diseased {{ color: #e74c3c; font-weight: bold; }}
                .healthy {{ color: #2ecc71; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌱 Crop Disease Detection Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Analysis Summary</h2>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Samples Analyzed</td><td>{len(predictions)}</td></tr>
                    <tr><td>Diseased Samples</td><td class="diseased">{diseased_count}</td></tr>
                    <tr><td>Healthy Samples</td><td class="healthy">{healthy_count}</td></tr>
                    <tr><td>Disease Rate</td><td>{(diseased_count/len(predictions)*100):.1f}%</td></tr>
                    <tr><td>Average Confidence</td><td>{np.mean(confidences):.3f}</td></tr>
                    <tr><td>High Confidence Predictions (>0.8)</td><td>{sum(1 for c in confidences if c > 0.8)}</td></tr>
                    <tr><td>Low Confidence Predictions (<0.6)</td><td>{sum(1 for c in confidences if c < 0.6)}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📁 File Information</h2>
                <table class="stats-table">
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>File Type</td><td>{metadata.get('file_type', 'Unknown')}</td></tr>
                    <tr><td>File Size</td><td>{metadata.get('file_size_mb', 0):.2f} MB</td></tr>
                    <tr><td>Original Shape</td><td>{metadata.get('original_shape', 'Unknown')}</td></tr>
                    <tr><td>Processed Shape</td><td>{metadata.get('processed_shape', 'Unknown')}</td></tr>
                    <tr><td>Data Range</td><td>{metadata.get('data_range', 'Unknown')}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🔍 Detailed Results</h2>
                <p>Detailed predictions have been exported to:</p>
                <ul>
                    <li><strong>predictions.csv</strong> - Tabular format for analysis</li>
                    <li><strong>predictions.json</strong> - Structured format for applications</li>
                    <li><strong>visualizations/</strong> - Analysis plots and charts</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>⚠️ Important Notes</h2>
                <ul>
                    <li>This analysis is based on machine learning predictions and should be validated by agricultural experts.</li>
                    <li>Low confidence predictions (< 0.6) should be reviewed manually.</li>
                    <li>Environmental factors and crop variety may affect prediction accuracy.</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path