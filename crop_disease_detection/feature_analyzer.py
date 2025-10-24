"""
Feature importance analysis module for hyperspectral crop disease detection.

This module implements comprehensive feature importance analysis to identify
the most critical wavelengths for disease detection and validate biological relevance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
import os

from .base import BaseProcessor


class FeatureAnalyzer(BaseProcessor):
    """
    Comprehensive feature importance analysis for hyperspectral data.
    
    Identifies critical wavelengths for disease detection using multiple methods
    and validates biological relevance of important features.
    """
    
    def __init__(self, output_dir: str = "results/plots/features"):
        """
        Initialize the feature analyzer.
        
        Args:
            output_dir: Directory to save feature analysis plots
        """
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Known disease-related wavelengths (literature-based)
        self.known_disease_wavelengths = {
            'chlorophyll_a': [430, 663, 678],
            'chlorophyll_b': [453, 642],
            'carotenoids': [400, 500],
            'water_stress': [970, 1200, 1450, 1940],
            'nitrogen_stress': [550, 708, 750],
            'red_edge': [680, 700, 740],
            'near_infrared': [800, 900, 1000],
            'shortwave_infrared': [1550, 1750, 2200]
        }
        
        self.analysis_results = {}
    
    def extract_model_feature_importance(self, model: Any, method: str = "auto") -> np.ndarray:
        """
        Extract feature importance from trained models.
        
        Args:
            model: Trained model object
            method: Method to extract importance ("auto", "coefficients", "feature_importances", "permutation")
            
        Returns:
            Array of feature importance scores
        """
        self.logger.info(f"Extracting feature importance using method: {method}")
        
        if method == "auto":
            # Automatically determine best method based on model type
            if hasattr(model, 'feature_importances_'):
                method = "feature_importances"
            elif hasattr(model, 'coef_'):
                method = "coefficients"
            else:
                method = "permutation"
        
        if method == "feature_importances":
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                raise ValueError("Model does not have feature_importances_ attribute")
                
        elif method == "coefficients":
            if hasattr(model, 'coef_'):
                # Use absolute values of coefficients
                importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                raise ValueError("Model does not have coef_ attribute")
                
        elif method == "permutation":
            # This requires X and y data, will be implemented in a separate method
            raise ValueError("Permutation importance requires separate method call")
            
        else:
            raise ValueError(f"Unknown importance extraction method: {method}")
        
        # Normalize importance scores
        importance = importance / np.sum(importance)
        
        self.logger.info(f"Extracted feature importance: {len(importance)} features")
        return importance
    
    def calculate_permutation_importance(self, model: Any, X: np.ndarray, y: np.ndarray, 
                                       n_repeats: int = 10, random_state: int = 42) -> np.ndarray:
        """
        Calculate permutation-based feature importance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            n_repeats: Number of permutation repeats
            random_state: Random seed for reproducibility
            
        Returns:
            Array of permutation importance scores
        """
        self.logger.info("Calculating permutation importance...")
        
        perm_importance = permutation_importance(
            model, X, y, 
            n_repeats=n_repeats, 
            random_state=random_state,
            scoring='accuracy'
        )
        
        # Use mean importance across repeats
        importance = perm_importance.importances_mean
        
        # Normalize scores
        importance = importance / np.sum(importance) if np.sum(importance) > 0 else importance
        
        self.logger.info(f"Calculated permutation importance for {len(importance)} features")
        return importance
    
    def calculate_statistical_importance(self, X: np.ndarray, y: np.ndarray, 
                                       method: str = "f_classif") -> np.ndarray:
        """
        Calculate statistical feature importance using univariate tests.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Statistical method ("f_classif", "mutual_info", "correlation")
            
        Returns:
            Array of statistical importance scores
        """
        self.logger.info(f"Calculating statistical importance using {method}")
        
        if method == "f_classif":
            # F-test for classification
            scores, _ = f_classif(X, y)
            
        elif method == "mutual_info":
            # Mutual information
            scores = mutual_info_classif(X, y, random_state=42)
            
        elif method == "correlation":
            # Point-biserial correlation for binary classification
            scores = np.abs([stats.pointbiserialr(X[:, i], y)[0] for i in range(X.shape[1])])
            scores = np.nan_to_num(scores)  # Handle NaN values
            
        else:
            raise ValueError(f"Unknown statistical method: {method}")
        
        # Normalize scores
        scores = scores / np.sum(scores) if np.sum(scores) > 0 else scores
        
        self.logger.info(f"Calculated statistical importance for {len(scores)} features")
        return scores
    
    def identify_top_wavelengths(self, importance_scores: np.ndarray, 
                               wavelengths: Optional[List[float]] = None,
                               n_top: int = 10) -> List[Tuple[float, float]]:
        """
        Identify top contributing wavelengths for disease classification.
        
        Args:
            importance_scores: Array of feature importance scores
            wavelengths: List of corresponding wavelengths (if None, uses default range)
            n_top: Number of top wavelengths to return
            
        Returns:
            List of tuples (wavelength, importance_score) sorted by importance
        """
        self.logger.info(f"Identifying top {n_top} wavelengths")
        
        if wavelengths is None:
            # Generate default wavelength range (437-2345 nm)
            wavelengths = np.linspace(437, 2345, len(importance_scores))
        
        if len(wavelengths) != len(importance_scores):
            raise ValueError("Wavelengths and importance scores must have same length")
        
        # Create pairs and sort by importance
        wavelength_importance = list(zip(wavelengths, importance_scores))
        top_wavelengths = sorted(wavelength_importance, key=lambda x: x[1], reverse=True)[:n_top]
        
        self.logger.info(f"Top wavelengths identified: {[w[0] for w in top_wavelengths[:3]]}")
        return top_wavelengths
    
    def analyze_multiple_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                              wavelengths: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Analyze feature importance across multiple models.
        
        Args:
            models: Dictionary of model_name -> trained_model
            X: Feature matrix
            y: Target labels
            wavelengths: List of wavelengths
            
        Returns:
            Dictionary of model_name -> importance_scores
        """
        self.logger.info(f"Analyzing feature importance for {len(models)} models")
        
        importance_results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Analyzing model: {model_name}")
            
            try:
                # Try model-specific importance first
                importance = self.extract_model_feature_importance(model)
                importance_results[f"{model_name}_intrinsic"] = importance
                
            except ValueError:
                self.logger.warning(f"No intrinsic importance for {model_name}")
            
            # Calculate permutation importance for all models
            try:
                perm_importance = self.calculate_permutation_importance(model, X, y)
                importance_results[f"{model_name}_permutation"] = perm_importance
                
            except Exception as e:
                self.logger.error(f"Failed to calculate permutation importance for {model_name}: {e}")
        
        # Calculate statistical importance (model-independent)
        if "statistical_f_test" not in importance_results:
            stat_importance = self.calculate_statistical_importance(X, y, "f_classif")
            importance_results["statistical_f_test"] = stat_importance
        
        self.analysis_results = importance_results
        self.logger.info(f"Completed importance analysis for {len(importance_results)} methods")
        
        return importance_results
    
    def create_ensemble_importance(self, importance_dict: Dict[str, np.ndarray],
                                 weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Create ensemble importance by combining multiple importance measures.
        
        Args:
            importance_dict: Dictionary of method_name -> importance_scores
            weights: Optional weights for each method
            
        Returns:
            Combined ensemble importance scores
        """
        self.logger.info("Creating ensemble importance scores")
        
        if weights is None:
            # Equal weights for all methods
            weights = {method: 1.0 for method in importance_dict.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Combine importance scores
        ensemble_importance = np.zeros(len(next(iter(importance_dict.values()))))
        
        for method, importance in importance_dict.items():
            weight = weights.get(method, 0.0)
            ensemble_importance += weight * importance
        
        # Normalize final scores
        ensemble_importance = ensemble_importance / np.sum(ensemble_importance)
        
        self.logger.info("Created ensemble importance scores")
        return ensemble_importance
    
    def validate_biological_relevance(self, top_wavelengths: List[Tuple[float, float]],
                                    tolerance: float = 20.0) -> Dict[str, Any]:
        """
        Validate biological relevance of important wavelengths.
        
        Args:
            top_wavelengths: List of (wavelength, importance) tuples
            tolerance: Tolerance in nm for matching known wavelengths
            
        Returns:
            Dictionary with biological relevance analysis
        """
        self.logger.info("Validating biological relevance of important wavelengths")
        
        relevance_analysis = {
            'matches': [],
            'unmatched_wavelengths': [],
            'biological_categories': {},
            'relevance_score': 0.0
        }
        
        # Check each top wavelength against known disease indicators
        for wavelength, importance in top_wavelengths:
            matched = False
            
            for category, known_wavelengths in self.known_disease_wavelengths.items():
                for known_wl in known_wavelengths:
                    if abs(wavelength - known_wl) <= tolerance:
                        relevance_analysis['matches'].append({
                            'wavelength': wavelength,
                            'importance': importance,
                            'known_wavelength': known_wl,
                            'category': category,
                            'difference': abs(wavelength - known_wl)
                        })
                        
                        if category not in relevance_analysis['biological_categories']:
                            relevance_analysis['biological_categories'][category] = []
                        relevance_analysis['biological_categories'][category].append(wavelength)
                        
                        matched = True
                        break
                
                if matched:
                    break
            
            if not matched:
                relevance_analysis['unmatched_wavelengths'].append({
                    'wavelength': wavelength,
                    'importance': importance
                })
        
        # Calculate relevance score
        total_importance = sum([imp for _, imp in top_wavelengths])
        matched_importance = sum([match['importance'] for match in relevance_analysis['matches']])
        relevance_analysis['relevance_score'] = matched_importance / total_importance if total_importance > 0 else 0.0
        
        self.logger.info(f"Biological relevance score: {relevance_analysis['relevance_score']:.3f}")
        self.logger.info(f"Matched {len(relevance_analysis['matches'])} wavelengths to known indicators")
        
        return relevance_analysis
    
    def generate_interpretable_explanations(self, model: Any, X_sample: np.ndarray,
                                          wavelengths: Optional[List[float]] = None,
                                          top_n: int = 5) -> Dict[str, Any]:
        """
        Generate interpretable explanations for model predictions.
        
        Args:
            model: Trained model
            X_sample: Single sample for explanation
            wavelengths: List of wavelengths
            top_n: Number of top contributing features to explain
            
        Returns:
            Dictionary with prediction explanation
        """
        self.logger.info("Generating interpretable prediction explanation")
        
        if wavelengths is None:
            wavelengths = np.linspace(437, 2345, X_sample.shape[0])
        
        # Get prediction and probability
        prediction = model.predict(X_sample.reshape(1, -1))[0]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X_sample.reshape(1, -1))[0]
        else:
            probability = [1.0 - prediction, prediction] if prediction == 0 else [prediction, 1.0 - prediction]
        
        # Get feature importance for this model
        try:
            importance = self.extract_model_feature_importance(model)
        except ValueError:
            # Use statistical importance as fallback
            importance = np.ones(len(X_sample)) / len(X_sample)
        
        # Calculate feature contributions (importance * feature_value)
        contributions = importance * np.abs(X_sample)
        
        # Get top contributing features
        top_indices = np.argsort(contributions)[-top_n:][::-1]
        
        explanation = {
            'prediction': int(prediction),
            'prediction_label': 'Diseased' if prediction == 1 else 'Healthy',
            'confidence': float(max(probability)),
            'class_probabilities': {
                'healthy': float(probability[0]),
                'diseased': float(probability[1])
            },
            'top_contributors': []
        }
        
        for idx in top_indices:
            explanation['top_contributors'].append({
                'wavelength': float(wavelengths[idx]),
                'spectral_value': float(X_sample[idx]),
                'importance': float(importance[idx]),
                'contribution': float(contributions[idx])
            })
        
        self.logger.info(f"Generated explanation for {explanation['prediction_label']} prediction")
        return explanation
    
    def save_analysis_results(self, filename: str = "feature_analysis_results.json") -> str:
        """
        Save feature analysis results to JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        import json
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for method, importance in self.analysis_results.items():
            serializable_results[method] = importance.tolist()
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved analysis results to {output_path}")
        return output_path
    
    def load_analysis_results(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load previously saved analysis results.
        
        Args:
            filepath: Path to saved results file
            
        Returns:
            Dictionary of method_name -> importance_scores
        """
        import json
        
        with open(filepath, 'r') as f:
            loaded_results = json.load(f)
        
        # Convert lists back to numpy arrays
        self.analysis_results = {
            method: np.array(importance) 
            for method, importance in loaded_results.items()
        }
        
        self.logger.info(f"Loaded analysis results from {filepath}")
        return self.analysis_results    

    def plot_spectral_importance(self, importance_scores: np.ndarray, 
                               wavelengths: Optional[List[float]] = None,
                               title: str = "Spectral Feature Importance",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create plots showing importance across the spectral range.
        
        Args:
            importance_scores: Array of feature importance scores
            wavelengths: List of corresponding wavelengths
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating spectral importance visualization")
        
        if wavelengths is None:
            wavelengths = np.linspace(437, 2345, len(importance_scores))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Main spectral importance plot
        ax1.plot(wavelengths, importance_scores, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(wavelengths, importance_scores, alpha=0.3)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Feature Importance')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # Add known disease wavelength regions
        colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive']
        for i, (category, known_wls) in enumerate(self.known_disease_wavelengths.items()):
            color = colors[i % len(colors)]
            for wl in known_wls:
                if min(wavelengths) <= wl <= max(wavelengths):
                    ax1.axvline(x=wl, color=color, linestyle='--', alpha=0.6, linewidth=1)
        
        # Create legend for known wavelengths
        legend_elements = [plt.Line2D([0], [0], color=colors[i % len(colors)], 
                                    linestyle='--', label=cat.replace('_', ' ').title())
                         for i, cat in enumerate(self.known_disease_wavelengths.keys())]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Top wavelengths bar plot
        top_wavelengths = self.identify_top_wavelengths(importance_scores, wavelengths, n_top=15)
        top_wls, top_scores = zip(*top_wavelengths)
        
        bars = ax2.bar(range(len(top_wls)), top_scores, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Top Wavelengths (nm)')
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Top 15 Most Important Wavelengths')
        ax2.set_xticks(range(len(top_wls)))
        ax2.set_xticklabels([f'{wl:.0f}' for wl in top_wls], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved spectral importance plot to {save_path}")
        elif self.output_dir:
            default_path = os.path.join(self.output_dir, f"spectral_importance_{title.lower().replace(' ', '_')}.png")
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved spectral importance plot to {default_path}")
        
        return fig
    
    def plot_wavelength_contributions(self, importance_dict: Dict[str, np.ndarray],
                                    wavelengths: Optional[List[float]] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison plot of wavelength contributions across different methods.
        
        Args:
            importance_dict: Dictionary of method_name -> importance_scores
            wavelengths: List of wavelengths
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating wavelength contribution comparison plot")
        
        if wavelengths is None:
            wavelengths = np.linspace(437, 2345, len(next(iter(importance_dict.values()))))
        
        fig, axes = plt.subplots(len(importance_dict), 1, figsize=(14, 4 * len(importance_dict)))
        if len(importance_dict) == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (method, importance) in enumerate(importance_dict.items()):
            ax = axes[i]
            color = colors[i % len(colors)]
            
            # Plot importance curve
            ax.plot(wavelengths, importance, color=color, linewidth=2, alpha=0.8, label=method)
            ax.fill_between(wavelengths, importance, alpha=0.3, color=color)
            
            # Highlight top wavelengths
            top_wavelengths = self.identify_top_wavelengths(importance, wavelengths, n_top=5)
            for wl, score in top_wavelengths:
                ax.scatter(wl, score, color='red', s=50, zorder=5)
                ax.annotate(f'{wl:.0f}nm', (wl, score), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8, alpha=0.8)
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Importance Score')
            ax.set_title(f'Feature Importance: {method.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved wavelength contribution plot to {save_path}")
        elif self.output_dir:
            default_path = os.path.join(self.output_dir, "wavelength_contributions_comparison.png")
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved wavelength contribution plot to {default_path}")
        
        return fig
    
    def plot_biological_relevance_analysis(self, relevance_analysis: Dict[str, Any],
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of biological relevance analysis.
        
        Args:
            relevance_analysis: Results from validate_biological_relevance
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating biological relevance analysis plot")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Relevance score gauge
        score = relevance_analysis['relevance_score']
        ax1.pie([score, 1-score], labels=['Biologically Relevant', 'Unknown Relevance'], 
               colors=['green', 'lightgray'], autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Biological Relevance Score: {score:.1%}')
        
        # 2. Category distribution
        categories = relevance_analysis['biological_categories']
        if categories:
            cat_names = list(categories.keys())
            cat_counts = [len(wls) for wls in categories.values()]
            
            bars = ax2.bar(cat_names, cat_counts, color='skyblue', alpha=0.7)
            ax2.set_xlabel('Biological Categories')
            ax2.set_ylabel('Number of Matching Wavelengths')
            ax2.set_title('Distribution of Biologically Relevant Wavelengths')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, cat_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No biological matches found', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Distribution of Biologically Relevant Wavelengths')
        
        # 3. Matched vs unmatched wavelengths
        matched_wls = [match['wavelength'] for match in relevance_analysis['matches']]
        matched_importance = [match['importance'] for match in relevance_analysis['matches']]
        unmatched_wls = [wl['wavelength'] for wl in relevance_analysis['unmatched_wavelengths']]
        unmatched_importance = [wl['importance'] for wl in relevance_analysis['unmatched_wavelengths']]
        
        if matched_wls:
            ax3.scatter(matched_wls, matched_importance, color='green', alpha=0.7, 
                       s=60, label='Biologically Relevant')
        if unmatched_wls:
            ax3.scatter(unmatched_wls, unmatched_importance, color='red', alpha=0.7, 
                       s=60, label='Unknown Relevance')
        
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Importance Score')
        ax3.set_title('Wavelength Importance vs Biological Relevance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Detailed matches table (as text)
        ax4.axis('off')
        if relevance_analysis['matches']:
            matches_text = "Top Biological Matches:\n\n"
            for i, match in enumerate(relevance_analysis['matches'][:10]):  # Show top 10
                matches_text += f"{i+1}. {match['wavelength']:.0f}nm → {match['category'].replace('_', ' ').title()}\n"
                matches_text += f"   Known: {match['known_wavelength']}nm, Diff: {match['difference']:.1f}nm\n"
                matches_text += f"   Importance: {match['importance']:.4f}\n\n"
        else:
            matches_text = "No biological matches found.\n\nThis could indicate:\n"
            matches_text += "• Novel disease indicators\n• Data quality issues\n• Model artifacts"
        
        ax4.text(0.05, 0.95, matches_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved biological relevance plot to {save_path}")
        elif self.output_dir:
            default_path = os.path.join(self.output_dir, "biological_relevance_analysis.png")
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved biological relevance plot to {default_path}")
        
        return fig
    
    def create_comprehensive_feature_report(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                                          wavelengths: Optional[List[float]] = None,
                                          save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive feature importance analysis report with all visualizations.
        
        Args:
            models: Dictionary of trained models
            X: Feature matrix
            y: Target labels
            wavelengths: List of wavelengths
            save_dir: Directory to save all outputs
            
        Returns:
            Dictionary with analysis results and file paths
        """
        self.logger.info("Creating comprehensive feature importance report")
        
        if save_dir:
            self.output_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        
        report = {
            'analysis_results': {},
            'top_wavelengths': {},
            'biological_relevance': {},
            'plots': {},
            'summary': {}
        }
        
        # 1. Analyze all models
        importance_results = self.analyze_multiple_models(models, X, y, wavelengths)
        report['analysis_results'] = importance_results
        
        # 2. Create ensemble importance
        ensemble_importance = self.create_ensemble_importance(importance_results)
        importance_results['ensemble'] = ensemble_importance
        
        # 3. Identify top wavelengths for each method
        for method, importance in importance_results.items():
            top_wls = self.identify_top_wavelengths(importance, wavelengths, n_top=10)
            report['top_wavelengths'][method] = top_wls
        
        # 4. Biological relevance analysis
        ensemble_top_wls = self.identify_top_wavelengths(ensemble_importance, wavelengths, n_top=15)
        relevance_analysis = self.validate_biological_relevance(ensemble_top_wls)
        report['biological_relevance'] = relevance_analysis
        
        # 5. Create all visualizations
        plots = {}
        
        # Individual importance plots
        for method, importance in importance_results.items():
            fig = self.plot_spectral_importance(importance, wavelengths, 
                                              title=f"Feature Importance: {method.replace('_', ' ').title()}")
            plot_path = os.path.join(self.output_dir, f"importance_{method}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots[f"importance_{method}"] = plot_path
            plt.close(fig)
        
        # Comparison plot
        fig = self.plot_wavelength_contributions(importance_results, wavelengths)
        comparison_path = os.path.join(self.output_dir, "importance_comparison.png")
        fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plots['comparison'] = comparison_path
        plt.close(fig)
        
        # Biological relevance plot
        fig = self.plot_biological_relevance_analysis(relevance_analysis)
        relevance_path = os.path.join(self.output_dir, "biological_relevance.png")
        fig.savefig(relevance_path, dpi=300, bbox_inches='tight')
        plots['biological_relevance'] = relevance_path
        plt.close(fig)
        
        report['plots'] = plots
        
        # 6. Create summary
        report['summary'] = {
            'total_features': len(ensemble_importance),
            'top_wavelength_ensemble': ensemble_top_wls[0][0] if ensemble_top_wls else None,
            'biological_relevance_score': relevance_analysis['relevance_score'],
            'matched_categories': len(relevance_analysis['biological_categories']),
            'total_matches': len(relevance_analysis['matches']),
            'methods_analyzed': list(importance_results.keys())
        }
        
        # 7. Save analysis results
        results_path = self.save_analysis_results("comprehensive_feature_analysis.json")
        report['results_file'] = results_path
        
        self.logger.info("Completed comprehensive feature importance report")
        return report 
   
    def process(self, data: Any) -> Any:
        """
        Process method required by BaseProcessor.
        
        Args:
            data: Input data (can be models dict, importance scores, etc.)
            
        Returns:
            Processed results
        """
        # This is a placeholder implementation to satisfy the abstract method requirement
        # The actual processing is done through specific methods like analyze_multiple_models
        if isinstance(data, dict) and 'models' in data:
            # If data contains models, perform comprehensive analysis
            models = data['models']
            X = data.get('X')
            y = data.get('y')
            wavelengths = data.get('wavelengths')
            
            if X is not None and y is not None:
                return self.create_comprehensive_feature_report(models, X, y, wavelengths)
        
        return data