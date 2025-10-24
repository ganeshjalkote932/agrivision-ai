"""
Comprehensive report generation system for hyperspectral crop disease detection.

This module implements comprehensive performance reports with recommendations,
model improvement suggestions, and next steps for the entire pipeline.
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseProcessor
from .logger import ProcessLogger
from .visualization_exporter import VisualizationExporter


class ReportGenerator(BaseProcessor):
    """
    Comprehensive report generation system.
    
    Creates detailed performance reports with recommendations, model improvement
    suggestions, and actionable next steps for the hyperspectral disease detection pipeline.
    """
    
    def __init__(self, output_dir: str = "results/reports",
                 session_id: Optional[str] = None,
                 logger: Optional[ProcessLogger] = None,
                 viz_exporter: Optional[VisualizationExporter] = None):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for saving reports
            session_id: Session identifier for organization
            logger: ProcessLogger instance
            viz_exporter: VisualizationExporter for saving plots
        """
        super().__init__()
        
        self.output_dir = Path(output_dir)
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.process_logger = logger
        self.viz_exporter = viz_exporter
        
        # Report data storage
        self.report_data = {
            'session_info': {
                'session_id': self.session_id,
                'created_at': datetime.datetime.now().isoformat(),
                'report_directory': str(self.session_dir)
            },
            'data_analysis': {},
            'preprocessing': {},
            'model_training': {},
            'model_evaluation': {},
            'feature_analysis': {},
            'deployment': {},
            'recommendations': {},
            'next_steps': []
        }
        
        # Performance thresholds for recommendations
        self.performance_thresholds = {
            'excellent': 0.95,
            'good': 0.90,
            'acceptable': 0.85,
            'needs_improvement': 0.80
        }
    
    def add_data_analysis_results(self, dataset_info: Dict[str, Any], 
                                quality_metrics: Dict[str, Any]):
        """
        Add data analysis results to the report.
        
        Args:
            dataset_info: Information about the dataset
            quality_metrics: Data quality metrics
        """
        self.report_data['data_analysis'] = {
            'dataset_info': dataset_info,
            'quality_metrics': quality_metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if self.process_logger:
            self.process_logger.log_data_info("report_data_analysis", dataset_info)
    
    def add_preprocessing_results(self, preprocessing_info: Dict[str, Any],
                                before_stats: Dict[str, Any],
                                after_stats: Dict[str, Any]):
        """
        Add preprocessing results to the report.
        
        Args:
            preprocessing_info: Information about preprocessing steps
            before_stats: Statistics before preprocessing
            after_stats: Statistics after preprocessing
        """
        self.report_data['preprocessing'] = {
            'preprocessing_info': preprocessing_info,
            'before_stats': before_stats,
            'after_stats': after_stats,
            'improvement_metrics': self._calculate_preprocessing_improvements(before_stats, after_stats),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def add_model_training_results(self, training_results: Dict[str, Any]):
        """
        Add model training results to the report.
        
        Args:
            training_results: Results from model training
        """
        self.report_data['model_training'] = {
            'training_results': training_results,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def add_model_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """
        Add model evaluation results to the report.
        
        Args:
            evaluation_results: Results from model evaluation
        """
        self.report_data['model_evaluation'] = {
            'evaluation_results': evaluation_results,
            'performance_analysis': self._analyze_model_performance(evaluation_results),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def add_feature_analysis_results(self, feature_results: Dict[str, Any]):
        """
        Add feature analysis results to the report.
        
        Args:
            feature_results: Results from feature importance analysis
        """
        self.report_data['feature_analysis'] = {
            'feature_results': feature_results,
            'biological_relevance': feature_results.get('biological_relevance', {}),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def add_deployment_results(self, deployment_info: Dict[str, Any]):
        """
        Add deployment results to the report.
        
        Args:
            deployment_info: Information about model deployment
        """
        self.report_data['deployment'] = {
            'deployment_info': deployment_info,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def _calculate_preprocessing_improvements(self, before_stats: Dict[str, Any],
                                           after_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvements from preprocessing."""
        improvements = {}
        
        # Compare standard deviations (normalization effect)
        if 'std_dev' in before_stats and 'std_dev' in after_stats:
            before_std_range = np.max(before_stats['std_dev']) - np.min(before_stats['std_dev'])
            after_std_range = np.max(after_stats['std_dev']) - np.min(after_stats['std_dev'])
            improvements['std_dev_normalization'] = (before_std_range - after_std_range) / before_std_range
        
        # Compare means (centering effect)
        if 'mean' in before_stats and 'mean' in after_stats:
            before_mean_range = np.max(before_stats['mean']) - np.min(before_stats['mean'])
            after_mean_range = np.max(after_stats['mean']) - np.min(after_stats['mean'])
            if before_mean_range > 0:
                improvements['mean_centering'] = (before_mean_range - after_mean_range) / before_mean_range
        
        return improvements
    
    def _analyze_model_performance(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model performance and categorize results."""
        analysis = {}
        
        # Get accuracy metrics
        accuracy = evaluation_results.get('accuracy', 0)
        precision = evaluation_results.get('precision', 0)
        recall = evaluation_results.get('recall', 0)
        f1_score = evaluation_results.get('f1_score', 0)
        
        # Categorize performance
        def categorize_performance(score):
            if score >= self.performance_thresholds['excellent']:
                return 'excellent'
            elif score >= self.performance_thresholds['good']:
                return 'good'
            elif score >= self.performance_thresholds['acceptable']:
                return 'acceptable'
            elif score >= self.performance_thresholds['needs_improvement']:
                return 'needs_improvement'
            else:
                return 'poor'
        
        analysis['performance_categories'] = {
            'accuracy': categorize_performance(accuracy),
            'precision': categorize_performance(precision),
            'recall': categorize_performance(recall),
            'f1_score': categorize_performance(f1_score)
        }
        
        # Overall performance assessment
        avg_score = np.mean([accuracy, precision, recall, f1_score])
        analysis['overall_performance'] = categorize_performance(avg_score)
        analysis['average_score'] = avg_score
        
        # Identify strengths and weaknesses
        scores = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
        analysis['best_metric'] = max(scores, key=scores.get)
        analysis['worst_metric'] = min(scores, key=scores.get)
        
        return analysis
    
    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate recommendations based on analysis results."""
        recommendations = {
            'data_quality': [],
            'preprocessing': [],
            'model_improvement': [],
            'feature_engineering': [],
            'deployment': [],
            'general': []
        }
        
        # Data quality recommendations
        data_analysis = self.report_data.get('data_analysis', {})
        if data_analysis:
            dataset_info = data_analysis.get('dataset_info', {})
            quality_metrics = data_analysis.get('quality_metrics', {})
            
            # Check dataset size
            n_samples = dataset_info.get('n_samples', 0)
            if n_samples < 1000:
                recommendations['data_quality'].append(
                    "Consider collecting more data samples (current: {}, recommended: >1000)".format(n_samples)
                )
            
            # Check missing values
            missing_percentage = quality_metrics.get('missing_percentage', 0)
            if missing_percentage > 5:
                recommendations['data_quality'].append(
                    f"High missing data percentage ({missing_percentage:.1f}%) - consider data imputation strategies"
                )
        
        # Model performance recommendations
        evaluation = self.report_data.get('model_evaluation', {})
        if evaluation:
            performance_analysis = evaluation.get('performance_analysis', {})
            overall_performance = performance_analysis.get('overall_performance', 'unknown')
            
            if overall_performance in ['poor', 'needs_improvement']:
                recommendations['model_improvement'].extend([
                    "Consider hyperparameter tuning to improve model performance",
                    "Try ensemble methods (Random Forest, Gradient Boosting)",
                    "Experiment with different algorithms (SVM, Neural Networks)",
                    "Increase training data if possible"
                ])
            
            elif overall_performance == 'acceptable':
                recommendations['model_improvement'].extend([
                    "Fine-tune hyperparameters for marginal improvements",
                    "Consider feature selection to reduce overfitting",
                    "Validate model on additional test sets"
                ])
            
            # Specific metric recommendations
            worst_metric = performance_analysis.get('worst_metric', '')
            if worst_metric == 'recall':
                recommendations['model_improvement'].append(
                    "Low recall detected - consider adjusting classification threshold or class weights"
                )
            elif worst_metric == 'precision':
                recommendations['model_improvement'].append(
                    "Low precision detected - focus on reducing false positives"
                )
        
        # Feature analysis recommendations
        feature_analysis = self.report_data.get('feature_analysis', {})
        if feature_analysis:
            biological_relevance = feature_analysis.get('biological_relevance', {})
            relevance_score = biological_relevance.get('relevance_score', 0)
            
            if relevance_score < 0.3:
                recommendations['feature_engineering'].extend([
                    "Low biological relevance score - validate feature importance results",
                    "Consider domain expert consultation for feature interpretation",
                    "Investigate potential data quality issues in spectral measurements"
                ])
            elif relevance_score > 0.7:
                recommendations['feature_engineering'].append(
                    "High biological relevance - model is learning meaningful patterns"
                )
        
        # Preprocessing recommendations
        preprocessing = self.report_data.get('preprocessing', {})
        if preprocessing:
            improvement_metrics = preprocessing.get('improvement_metrics', {})
            
            std_normalization = improvement_metrics.get('std_dev_normalization', 0)
            if std_normalization < 0.5:
                recommendations['preprocessing'].append(
                    "Consider additional normalization techniques for better feature scaling"
                )
        
        # General recommendations
        recommendations['general'].extend([
            "Document all preprocessing steps for reproducibility",
            "Implement cross-validation for robust performance estimation",
            "Create automated testing pipeline for model validation",
            "Monitor model performance in production environment"
        ])
        
        return recommendations
    
    def _generate_next_steps(self) -> List[Dict[str, str]]:
        """Generate actionable next steps."""
        next_steps = []
        
        # Based on model performance
        evaluation = self.report_data.get('model_evaluation', {})
        if evaluation:
            performance_analysis = evaluation.get('performance_analysis', {})
            overall_performance = performance_analysis.get('overall_performance', 'unknown')
            
            if overall_performance in ['poor', 'needs_improvement']:
                next_steps.extend([
                    {
                        'priority': 'high',
                        'category': 'model_improvement',
                        'action': 'Conduct comprehensive hyperparameter optimization',
                        'timeline': '1-2 weeks',
                        'resources': 'Data scientist, computational resources'
                    },
                    {
                        'priority': 'high',
                        'category': 'data_collection',
                        'action': 'Collect additional training data',
                        'timeline': '2-4 weeks',
                        'resources': 'Field team, hyperspectral equipment'
                    }
                ])
            
            elif overall_performance in ['good', 'excellent']:
                next_steps.extend([
                    {
                        'priority': 'medium',
                        'category': 'deployment',
                        'action': 'Prepare model for production deployment',
                        'timeline': '1-2 weeks',
                        'resources': 'DevOps engineer, cloud infrastructure'
                    },
                    {
                        'priority': 'low',
                        'category': 'monitoring',
                        'action': 'Implement model performance monitoring',
                        'timeline': '2-3 weeks',
                        'resources': 'MLOps engineer, monitoring tools'
                    }
                ])
        
        # Feature analysis next steps
        feature_analysis = self.report_data.get('feature_analysis', {})
        if feature_analysis:
            next_steps.append({
                'priority': 'medium',
                'category': 'validation',
                'action': 'Validate important wavelengths with domain experts',
                'timeline': '1 week',
                'resources': 'Plant pathologist, spectroscopy expert'
            })
        
        # General next steps
        next_steps.extend([
            {
                'priority': 'medium',
                'category': 'documentation',
                'action': 'Create comprehensive model documentation',
                'timeline': '1 week',
                'resources': 'Technical writer, model developer'
            },
            {
                'priority': 'low',
                'category': 'research',
                'action': 'Investigate advanced deep learning approaches',
                'timeline': '4-6 weeks',
                'resources': 'Research team, GPU resources'
            }
        ])
        
        return next_steps
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive HTML report with all analysis results.
        
        Returns:
            Path to generated HTML report
        """
        # Generate recommendations and next steps
        self.report_data['recommendations'] = self._generate_recommendations()
        self.report_data['next_steps'] = self._generate_next_steps()
        
        # Create HTML report
        html_content = self._create_html_report()
        
        # Save HTML report
        report_file = self.session_dir / f"comprehensive_report_{self.session_id}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save JSON data
        json_file = self.session_dir / f"report_data_{self.session_id}.json"
        with open(json_file, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        if self.process_logger:
            self.process_logger.log_file_operation(
                "generate_report", str(report_file), True,
                report_type="comprehensive_html",
                sections=len(self.report_data)
            )
        
        return str(report_file)
    
    def _create_html_report(self) -> str:
        """Create HTML content for the comprehensive report."""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hyperspectral Crop Disease Detection - Comprehensive Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
        .excellent {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
        .good {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; }}
        .acceptable {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
        .needs-improvement {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
        .poor {{ background-color: #f5c6cb; border-left: 4px solid #721c24; }}
        .recommendation {{ margin: 5px 0; padding: 8px; background-color: #e9ecef; border-radius: 3px; }}
        .next-step {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007bff; background-color: #f8f9fa; }}
        .priority-high {{ border-left-color: #dc3545; }}
        .priority-medium {{ border-left-color: #ffc107; }}
        .priority-low {{ border-left-color: #28a745; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Hyperspectral Crop Disease Detection</h1>
        <h2>Comprehensive Analysis Report</h2>
        <p>Session ID: {session_id}</p>
        <p>Generated: {timestamp}</p>
    </div>

    {content}

    <div class="section">
        <h2>Report Summary</h2>
        <p>This comprehensive report provides detailed analysis of the hyperspectral crop disease detection pipeline, 
        including data quality assessment, model performance evaluation, and actionable recommendations for improvement.</p>
        <p>For technical details and raw data, refer to the accompanying JSON file: report_data_{session_id}.json</p>
    </div>
</body>
</html>
        """
        
        # Generate content sections
        content_sections = []
        
        # Executive Summary
        content_sections.append(self._create_executive_summary_section())
        
        # Data Analysis Section
        if self.report_data.get('data_analysis'):
            content_sections.append(self._create_data_analysis_section())
        
        # Model Performance Section
        if self.report_data.get('model_evaluation'):
            content_sections.append(self._create_model_performance_section())
        
        # Feature Analysis Section
        if self.report_data.get('feature_analysis'):
            content_sections.append(self._create_feature_analysis_section())
        
        # Recommendations Section
        content_sections.append(self._create_recommendations_section())
        
        # Next Steps Section
        content_sections.append(self._create_next_steps_section())
        
        content = '\n'.join(content_sections)
        
        return html_template.format(
            session_id=self.session_id,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content=content
        )
    
    def _create_executive_summary_section(self) -> str:
        """Create executive summary section."""
        evaluation = self.report_data.get('model_evaluation', {})
        performance_analysis = evaluation.get('performance_analysis', {})
        
        overall_performance = performance_analysis.get('overall_performance', 'unknown')
        average_score = performance_analysis.get('average_score', 0)
        
        performance_class = overall_performance.replace('_', '-')
        
        return f"""
    <div class="section {performance_class}">
        <h2>Executive Summary</h2>
        <h3>Overall Performance: {overall_performance.replace('_', ' ').title()}</h3>
        <p>Average Score: {average_score:.3f}</p>
        <p>The hyperspectral crop disease detection model has been trained and evaluated. 
        Based on the analysis, the model shows <strong>{overall_performance.replace('_', ' ')}</strong> performance 
        with an average score of <strong>{average_score:.1%}</strong>.</p>
    </div>
        """
    
    def _create_data_analysis_section(self) -> str:
        """Create data analysis section."""
        data_analysis = self.report_data['data_analysis']
        dataset_info = data_analysis.get('dataset_info', {})
        
        return f"""
    <div class="section">
        <h2>Data Analysis</h2>
        <div class="metric">
            <strong>Dataset Size:</strong> {dataset_info.get('n_samples', 'N/A')} samples
        </div>
        <div class="metric">
            <strong>Features:</strong> {dataset_info.get('n_features', 'N/A')} spectral bands
        </div>
        <div class="metric">
            <strong>Classes:</strong> {dataset_info.get('n_classes', 'N/A')} (Healthy/Diseased)
        </div>
        <p>The dataset contains hyperspectral measurements across {dataset_info.get('n_features', 'N/A')} 
        wavelengths for crop disease detection.</p>
    </div>
        """
    
    def _create_model_performance_section(self) -> str:
        """Create model performance section."""
        evaluation = self.report_data['model_evaluation']
        eval_results = evaluation.get('evaluation_results', {})
        performance_analysis = evaluation.get('performance_analysis', {})
        
        metrics_html = ""
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            value = eval_results.get(metric, 0)
            category = performance_analysis.get('performance_categories', {}).get(metric, 'unknown')
            category_class = category.replace('_', '-')
            
            metrics_html += f"""
            <div class="metric {category_class}">
                <strong>{metric.replace('_', ' ').title()}:</strong> {value:.3f} ({category.replace('_', ' ')})
            </div>
            """
        
        return f"""
    <div class="section">
        <h2>Model Performance</h2>
        {metrics_html}
        <p><strong>Best Metric:</strong> {performance_analysis.get('best_metric', 'N/A').replace('_', ' ').title()}</p>
        <p><strong>Needs Attention:</strong> {performance_analysis.get('worst_metric', 'N/A').replace('_', ' ').title()}</p>
    </div>
        """
    
    def _create_feature_analysis_section(self) -> str:
        """Create feature analysis section."""
        feature_analysis = self.report_data['feature_analysis']
        biological_relevance = feature_analysis.get('biological_relevance', {})
        
        relevance_score = biological_relevance.get('relevance_score', 0)
        matches = biological_relevance.get('matches', [])
        
        return f"""
    <div class="section">
        <h2>Feature Analysis</h2>
        <div class="metric">
            <strong>Biological Relevance Score:</strong> {relevance_score:.1%}
        </div>
        <div class="metric">
            <strong>Biologically Relevant Features:</strong> {len(matches)}
        </div>
        <p>The model identified {len(matches)} wavelengths that match known disease indicators, 
        achieving a biological relevance score of {relevance_score:.1%}.</p>
    </div>
        """
    
    def _create_recommendations_section(self) -> str:
        """Create recommendations section."""
        recommendations = self.report_data['recommendations']
        
        sections_html = ""
        for category, recs in recommendations.items():
            if recs:
                recs_html = "".join([f'<div class="recommendation">• {rec}</div>' for rec in recs])
                sections_html += f"""
                <h4>{category.replace('_', ' ').title()}</h4>
                {recs_html}
                """
        
        return f"""
    <div class="section">
        <h2>Recommendations</h2>
        {sections_html}
    </div>
        """
    
    def _create_next_steps_section(self) -> str:
        """Create next steps section."""
        next_steps = self.report_data['next_steps']
        
        steps_html = ""
        for step in next_steps:
            priority_class = f"priority-{step.get('priority', 'medium')}"
            steps_html += f"""
            <div class="next-step {priority_class}">
                <strong>{step.get('action', 'N/A')}</strong><br>
                <small>Priority: {step.get('priority', 'medium').title()} | 
                Timeline: {step.get('timeline', 'TBD')} | 
                Resources: {step.get('resources', 'TBD')}</small>
            </div>
            """
        
        return f"""
    <div class="section">
        <h2>Next Steps</h2>
        {steps_html}
    </div>
        """
    
    def process(self, data: Any) -> Any:
        """
        Process method required by BaseProcessor.
        
        Args:
            data: Input data for report generation
            
        Returns:
            Generated report path
        """
        if isinstance(data, dict):
            # Add data to appropriate section
            if 'dataset_info' in data:
                self.add_data_analysis_results(data['dataset_info'], data.get('quality_metrics', {}))
            elif 'evaluation_results' in data:
                self.add_model_evaluation_results(data['evaluation_results'])
            elif 'feature_results' in data:
                self.add_feature_analysis_results(data['feature_results'])
            
            # Generate report if requested
            if data.get('generate_report', False):
                return self.generate_comprehensive_report()
        
        return data