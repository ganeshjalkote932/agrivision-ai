"""
Model evaluation module for hyperspectral crop disease detection.

This module implements comprehensive model evaluation with detailed metrics,
visualizations, and statistical analysis to validate >95% accuracy targets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss
)
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from scipy import stats
import os
from collections import Counter

from .base import BaseEvaluator
from .config import EvaluationConfig


class ModelEvaluator(BaseEvaluator):
    """
    Comprehensive model evaluation system for hyperspectral crop disease detection.
    
    Provides detailed performance metrics, statistical analysis, visualizations,
    and validation to ensure model reliability and accuracy targets.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the model evaluator.
        
        Args:
            config: EvaluationConfig instance with evaluation parameters
        """
        super().__init__()
        self.config = config
        self.evaluation_results = {}
        self.figures = []
        
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with comprehensive metrics
        """
        self.logger.info("Calculating comprehensive evaluation metrics...")
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, (prec, rec, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            metrics[f'precision_class_{i}'] = prec
            metrics[f'recall_class_{i}'] = rec
            metrics[f'f1_class_{i}'] = f1
        
        # Advanced metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Probability-based metrics (if available)
        if y_proba is not None:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                # Binary classification
                metrics['roc_auc'] = auc(*roc_curve(y_true, y_proba[:, 1])[:2])
                metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_proba)
            elif y_proba.ndim == 2:
                # Multi-class classification
                try:
                    from sklearn.metrics import roc_auc_score
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
                except ValueError:
                    pass  # Skip if not applicable
                metrics['log_loss'] = log_loss(y_true, y_proba)
        
        # Class distribution analysis
        class_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        
        # Convert numpy int64 keys to regular int for JSON serialization
        metrics['true_class_distribution'] = {int(k): int(v) for k, v in class_counts.items()}
        metrics['pred_class_distribution'] = {int(k): int(v) for k, v in pred_counts.items()}
        
        # Error analysis
        correct_predictions = (y_true == y_pred)
        metrics['correct_predictions'] = int(np.sum(correct_predictions))
        metrics['incorrect_predictions'] = int(np.sum(~correct_predictions))
        metrics['error_rate'] = 1 - metrics['accuracy']
        
        self.logger.info(f"Calculated {len(metrics)} evaluation metrics")
        return metrics
    
    def generate_confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Generate detailed confusion matrix analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with confusion matrix analysis
        """
        self.logger.info("Generating confusion matrix analysis...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        analysis = {
            'confusion_matrix': cm.tolist(),
            'normalized_cm': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist(),
            'class_accuracies': {},
            'class_errors': {},
            'total_samples': len(y_true)
        }
        
        # Per-class analysis
        for i in range(cm.shape[0]):
            true_positives = cm[i, i]
            false_negatives = cm[i, :].sum() - true_positives
            false_positives = cm[:, i].sum() - true_positives
            true_negatives = cm.sum() - true_positives - false_negatives - false_positives
            
            class_accuracy = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            analysis['class_accuracies'][f'class_{i}'] = {
                'accuracy': class_accuracy,
                'true_positives': int(true_positives),
                'false_negatives': int(false_negatives),
                'false_positives': int(false_positives),
                'true_negatives': int(true_negatives),
                'sensitivity': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
                'specificity': true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            }
        
        # Overall confusion matrix statistics
        analysis['overall_stats'] = {
            'total_correct': int(np.trace(cm)),
            'total_incorrect': int(cm.sum() - np.trace(cm)),
            'most_confused_classes': self._find_most_confused_classes(cm),
            'class_balance': self._analyze_class_balance(cm)
        }
        
        return analysis
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray, 
                       class_names: Optional[List[str]] = None, save_plot: bool = True) -> plt.Figure:
        """
        Plot ROC curves for model evaluation.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            class_names: Optional class names for labeling
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating ROC curve plots...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(y_proba.shape[1])]
        
        # Plot ROC curve for each class
        for i in range(y_proba.shape[1]):
            # Create binary labels for current class
            y_binary = (y_true == i).astype(int)
            
            if len(np.unique(y_binary)) > 1:  # Only plot if both classes present
                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, linewidth=2, 
                       label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.config.plots_dir, 'roc_curves.png')
            os.makedirs(self.config.plots_dir, exist_ok=True)
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            self.logger.info(f"ROC curves saved to: {plot_path}")
        
        self.figures.append(fig)
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None, save_plot: bool = True) -> plt.Figure:
        """
        Plot confusion matrix with detailed annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names for labeling
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating confusion matrix plot...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=class_names, yticklabels=class_names)
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=ax2,
                   xticklabels=class_names, yticklabels=class_names)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.config.plots_dir, 'confusion_matrix.png')
            os.makedirs(self.config.plots_dir, exist_ok=True)
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to: {plot_path}")
        
        self.figures.append(fig)
        return fig
    
    def plot_learning_curves(self, model: Any, X: np.ndarray, y: np.ndarray,
                           cv: int = 5, save_plot: bool = True) -> plt.Figure:
        """
        Plot learning curves to analyze model performance vs training size.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating learning curves...")
        
        # Calculate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot learning curves
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                       alpha=0.1, color='blue')
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                       alpha=0.1, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Learning Curves')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add performance annotations
        final_train_score = train_mean[-1]
        final_val_score = val_mean[-1]
        gap = final_train_score - final_val_score
        
        ax.text(0.02, 0.98, f'Final Training Score: {final_train_score:.3f}\n'
                           f'Final Validation Score: {final_val_score:.3f}\n'
                           f'Overfitting Gap: {gap:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.config.plots_dir, 'learning_curves.png')
            os.makedirs(self.config.plots_dir, exist_ok=True)
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            self.logger.info(f"Learning curves saved to: {plot_path}")
        
        self.figures.append(fig)
        return fig
    
    def perform_statistical_significance_tests(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Perform statistical significance tests for model comparison.
        
        Args:
            model_results: Dictionary with model names and their CV scores
            
        Returns:
            Dictionary with statistical test results
        """
        self.logger.info("Performing statistical significance tests...")
        
        significance_results = {
            'pairwise_tests': {},
            'overall_analysis': {},
            'recommendations': []
        }
        
        model_names = list(model_results.keys())
        
        # Pairwise t-tests
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                scores1 = model_results[model1].get('cv_scores', [])
                scores2 = model_results[model2].get('cv_scores', [])
                
                if len(scores1) > 1 and len(scores2) > 1:
                    # Perform paired t-test
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                    
                    significance_results['pairwise_tests'][f'{model1}_vs_{model2}'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'effect_size': float(np.mean(scores1) - np.mean(scores2)),
                        'model1_mean': float(np.mean(scores1)),
                        'model2_mean': float(np.mean(scores2))
                    }
        
        # Overall ANOVA if more than 2 models
        if len(model_names) > 2:
            all_scores = [model_results[name].get('cv_scores', []) for name in model_names]
            if all(len(scores) > 1 for scores in all_scores):
                f_stat, p_value = stats.f_oneway(*all_scores)
                
                significance_results['overall_analysis'] = {
                    'anova_f_statistic': float(f_stat),
                    'anova_p_value': float(p_value),
                    'significant_difference': p_value < 0.05
                }
        
        # Generate recommendations
        best_model = max(model_names, key=lambda x: np.mean(model_results[x].get('cv_scores', [0])))
        significance_results['recommendations'].append(f"Best performing model: {best_model}")
        
        # Check for significant differences
        significant_pairs = [pair for pair, result in significance_results['pairwise_tests'].items() 
                           if result['significant']]
        
        if significant_pairs:
            significance_results['recommendations'].append(
                f"Significant performance differences found in {len(significant_pairs)} model pairs"
            )
        else:
            significance_results['recommendations'].append(
                "No statistically significant differences between models"
            )
        
        return significance_results
    
    def create_comprehensive_evaluation_report(self, model_results: Dict[str, Any],
                                             save_report: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report with all metrics and analysis.
        
        Args:
            model_results: Dictionary with model evaluation results
            save_report: Whether to save the report
            
        Returns:
            Comprehensive evaluation report
        """
        self.logger.info("Creating comprehensive evaluation report...")
        
        report = {
            'summary': {},
            'detailed_metrics': {},
            'model_comparison': {},
            'recommendations': [],
            'target_achievement': {}
        }
        
        # Extract best model information
        best_model_name = None
        best_accuracy = 0
        
        for model_name, results in model_results.items():
            if 'metrics' in results:
                accuracy = results['metrics'].get('accuracy', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
        
        # Summary statistics
        report['summary'] = {
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'target_accuracy': 0.95,
            'target_achieved': best_accuracy >= 0.95,
            'models_evaluated': len(model_results),
            'evaluation_date': pd.Timestamp.now().isoformat()
        }
        
        # Detailed metrics for each model
        for model_name, results in model_results.items():
            if 'metrics' in results:
                report['detailed_metrics'][model_name] = results['metrics']
        
        # Model comparison
        if len(model_results) > 1:
            comparison_metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
            
            for metric in comparison_metrics:
                metric_values = {}
                for model_name, results in model_results.items():
                    if 'metrics' in results and metric in results['metrics']:
                        metric_values[model_name] = results['metrics'][metric]
                
                if metric_values:
                    best_model_for_metric = max(metric_values.keys(), key=lambda x: metric_values[x])
                    report['model_comparison'][metric] = {
                        'values': metric_values,
                        'best_model': best_model_for_metric,
                        'best_value': metric_values[best_model_for_metric]
                    }
        
        # Target achievement analysis
        target_accuracy = 0.95
        
        report['target_achievement'] = {
            'target_accuracy': target_accuracy,
            'achieved': best_accuracy >= target_accuracy,
            'gap_to_target': max(0, target_accuracy - best_accuracy),
            'performance_level': self._classify_performance_level(best_accuracy)
        }
        
        # Generate recommendations
        if best_accuracy >= target_accuracy:
            report['recommendations'].append("✓ Target accuracy achieved - model ready for deployment")
        else:
            gap = target_accuracy - best_accuracy
            if gap <= 0.02:
                report['recommendations'].append("⚠ Close to target - minor hyperparameter tuning recommended")
            elif gap <= 0.05:
                report['recommendations'].append("⚠ Moderate gap - consider ensemble methods or feature engineering")
            else:
                report['recommendations'].append("✗ Significant gap - review data quality and model architecture")
        
        # Model-specific recommendations
        if best_model_name:
            best_results = model_results[best_model_name]
            if 'confusion_matrix_analysis' in best_results:
                cm_analysis = best_results['confusion_matrix_analysis']
                most_confused = cm_analysis['overall_stats'].get('most_confused_classes', [])
                if most_confused:
                    report['recommendations'].append(
                        f"Focus on improving classification between classes: {most_confused}"
                    )
        
        # Save report if requested
        if save_report:
            report_path = os.path.join(self.config.reports_dir, 'evaluation_report.json')
            os.makedirs(self.config.reports_dir, exist_ok=True)
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Evaluation report saved to: {report_path}")
        
        return report
    
    def _find_most_confused_classes(self, cm: np.ndarray) -> List[Tuple[int, int]]:
        """Find the most confused class pairs from confusion matrix."""
        confused_pairs = []
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((i, j, cm[i, j]))
        
        # Sort by confusion count and return top pairs
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        return [(pair[0], pair[1]) for pair in confused_pairs[:3]]
    
    def _analyze_class_balance(self, cm: np.ndarray) -> Dict[str, float]:
        """Analyze class balance from confusion matrix."""
        class_totals = cm.sum(axis=1)
        total_samples = cm.sum()
        
        return {
            f'class_{i}_proportion': float(count / total_samples)
            for i, count in enumerate(class_totals)
        }
    
    def _classify_performance_level(self, accuracy: float) -> str:
        """Classify performance level based on accuracy."""
        if accuracy >= 0.95:
            return "Excellent"
        elif accuracy >= 0.90:
            return "Good"
        elif accuracy >= 0.85:
            return "Fair"
        else:
            return "Poor"
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main evaluation method implementing BaseEvaluator interface.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_proba)
        
        # Generate confusion matrix analysis
        cm_analysis = self.generate_confusion_matrix_analysis(y_true, y_pred)
        
        # Create visualizations
        if self.config.plot_confusion_matrix:
            self.plot_confusion_matrix(y_true, y_pred)
        
        if self.config.plot_roc_curve and y_proba is not None:
            self.plot_roc_curves(y_true, y_proba)
        
        return {
            'metrics': metrics,
            'confusion_matrix_analysis': cm_analysis,
            'figures': self.figures
        }
    
    def process(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for comprehensive model evaluation.
        
        Args:
            model_results: Dictionary with model evaluation data
            
        Returns:
            Comprehensive evaluation results
        """
        return self.create_comprehensive_evaluation_report(model_results)