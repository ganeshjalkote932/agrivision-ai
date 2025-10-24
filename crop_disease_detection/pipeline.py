"""
Complete pipeline orchestration for hyperspectral crop disease detection.

This module implements end-to-end pipeline orchestration from data loading
to model export with comprehensive logging, error handling, and reporting.
"""

import os
import sys
import argparse
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
import traceback

from .config import DataConfig, PreprocessingConfig, ModelConfig, HyperparameterConfig
from .logger import ProcessLogger
from .visualization_exporter import VisualizationExporter
from .report_generator import ReportGenerator
from .data_loader import DataLoader
from .label_creator import DiseaseLabelCreator
from .preprocessor import Preprocessor
from .data_splitter import DataSplitter
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .feature_analyzer import FeatureAnalyzer
from .model_exporter import ModelExporter
from .disease_detector import DiseaseDetector


class CropDiseaseDetectionPipeline:
    """
    Complete end-to-end pipeline for hyperspectral crop disease detection.
    
    Orchestrates the entire workflow from data loading to model deployment
    with comprehensive logging, visualization, and reporting.
    """
    
    def __init__(self, config_file: Optional[str] = None,
                 output_dir: str = "results",
                 session_id: Optional[str] = None,
                 enable_logging: bool = True,
                 enable_visualization: bool = True,
                 enable_reporting: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            config_file: Path to configuration file (JSON)
            output_dir: Base output directory for all results
            session_id: Optional session identifier
            enable_logging: Enable comprehensive logging
            enable_visualization: Enable visualization export
            enable_reporting: Enable report generation
        """
        # Setup session
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging first
        self.logger = None
        if enable_logging:
            self.logger = ProcessLogger(
                name="crop_disease_pipeline",
                log_dir=str(self.session_dir / "logs"),
                log_level="INFO"
            )
        
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Initialize visualization exporter
        self.viz_exporter = None
        if enable_visualization:
            self.viz_exporter = VisualizationExporter(
                output_dir=str(self.session_dir / "visualizations"),
                session_id=self.session_id,
                logger=self.logger
            )
        
        # Initialize report generator
        self.report_generator = None
        if enable_reporting:
            self.report_generator = ReportGenerator(
                output_dir=str(self.session_dir / "reports"),
                session_id=self.session_id,
                logger=self.logger,
                viz_exporter=self.viz_exporter
            )
        
        # Pipeline components
        self.components = {}
        self.results = {}
        
        # Pipeline state
        self.is_initialized = False
        self.current_step = None
        
        if self.logger:
            self.logger.logger.info(f"Pipeline initialized for session: {self.session_id}")
    
    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'data': {
                'dataset_path': 'GHISACONUS_2008_001_speclib.csv',
                'target_accuracy': 0.95,
                'validation_split': 0.2,
                'random_seed': 42
            },
            'preprocessing': {
                'normalization_method': 'standard',
                'handle_outliers': True,
                'outlier_method': 'isolation_forest'
            },
            'models': {
                'algorithms': ['random_forest', 'mlp', 'svm'],
                'hyperparameter_optimization': True,
                'cross_validation_folds': 5
            },
            'feature_analysis': {
                'enable_importance_analysis': True,
                'biological_validation': True,
                'top_features': 15
            },
            'deployment': {
                'export_best_model': True,
                'create_deployment_package': True,
                'include_examples': True
            },
            'hardware': {
                'max_memory_gb': 4,
                'use_multiprocessing': True,
                'batch_size': 1000
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Merge configurations (user config overrides defaults)
                def merge_configs(default, user):
                    for key, value in user.items():
                        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                            merge_configs(default[key], value)
                        else:
                            default[key] = value
                
                merge_configs(default_config, user_config)
                
                if self.logger:
                    self.logger.log_file_operation("load_config", config_file, True)
                    
            except Exception as e:
                if self.logger:
                    self.logger.logger.error(f"Failed to load config file {config_file}: {e}")
                print(f"Warning: Failed to load config file {config_file}, using defaults")
        
        return default_config
    
    def initialize_components(self):
        """Initialize all pipeline components."""
        if self.logger:
            with self.logger.log_step("component_initialization", "Initializing pipeline components"):
                self._initialize_components_internal()
        else:
            self._initialize_components_internal()
        
        self.is_initialized = True
    
    def _initialize_components_internal(self):
        """Internal method to initialize components."""
        # Create configuration objects
        data_config = DataConfig()
        preprocessing_config = PreprocessingConfig()
        model_config = ModelConfig()
        hyperparameter_config = HyperparameterConfig()
        
        # Initialize components
        self.components = {
            'data_loader': DataLoader(data_config),
            'label_creator': DiseaseLabelCreator(data_config),
            'preprocessor': Preprocessor(preprocessing_config),
            'data_splitter': DataSplitter(data_config),
            'model_trainer': ModelTrainer(model_config, hyperparameter_config),
            'model_evaluator': ModelEvaluator(model_config),
            'feature_analyzer': FeatureAnalyzer(str(self.session_dir / "feature_analysis")),
            'model_exporter': ModelExporter(str(self.session_dir / "models"), model_config)
        }
        
        if self.logger:
            self.logger.logger.info(f"Initialized {len(self.components)} pipeline components")
    
    def run_complete_pipeline(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete end-to-end pipeline.
        
        Args:
            dataset_path: Path to dataset file (overrides config)
            
        Returns:
            Dictionary with pipeline results
        """
        if not self.is_initialized:
            self.initialize_components()
        
        try:
            # Use provided dataset path or config default
            data_path = dataset_path or self.config['data']['dataset_path']
            
            if self.logger:
                self.logger.logger.info("="*60)
                self.logger.logger.info("STARTING COMPLETE PIPELINE EXECUTION")
                self.logger.logger.info("="*60)
            
            # Step 1: Data Loading and Analysis
            self._run_data_loading_step(data_path)
            
            # Step 2: Label Creation
            self._run_label_creation_step()
            
            # Step 3: Data Preprocessing
            self._run_preprocessing_step()
            
            # Step 4: Data Splitting
            self._run_data_splitting_step()
            
            # Step 5: Model Training
            self._run_model_training_step()
            
            # Step 6: Model Evaluation
            self._run_model_evaluation_step()
            
            # Step 7: Feature Analysis
            self._run_feature_analysis_step()
            
            # Step 8: Model Export and Deployment
            self._run_model_export_step()
            
            # Step 9: Generate Final Report
            self._run_report_generation_step()
            
            # Pipeline completion
            if self.logger:
                self.logger.logger.info("="*60)
                self.logger.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
                self.logger.logger.info("="*60)
                self.logger.log_process_summary()
            
            return self.results
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Pipeline execution failed: {str(e)}")
                self.logger.logger.error(traceback.format_exc())
            raise
    
    def _run_data_loading_step(self, data_path: str):
        """Run data loading and initial analysis step."""
        if self.logger:
            with self.logger.log_step("data_loading", f"Loading dataset from {data_path}"):
                self._execute_data_loading(data_path)
        else:
            self._execute_data_loading(data_path)
    
    def _execute_data_loading(self, data_path: str):
        """Execute data loading logic."""
        # Load dataset
        df = self.components['data_loader'].load_dataset(data_path)
        
        # Perform data quality analysis
        quality_report = {
            'n_samples': len(df),
            'n_features': len([col for col in df.columns if col.startswith('X')]),
            'missing_values': df.isnull().sum().sum(),
            'data_types': df.dtypes.value_counts().to_dict()
        }
        
        # Store results
        self.results['raw_data'] = df
        self.results['data_quality'] = quality_report
        
        # Log data information
        if self.logger:
            self.logger.log_data_info("raw_dataset", df)
            self.logger.log_performance_metrics(quality_report, "data_quality")
        
        # Create visualizations
        if self.viz_exporter:
            spectral_columns = [col for col in df.columns if col.startswith('X')]
            target_col = 'stage' if 'stage' in df.columns else None
            
            data_plots = self.viz_exporter.create_data_analysis_plots(df, target_col)
            self.results['data_analysis_plots'] = data_plots
        
        # Add to report
        if self.report_generator:
            dataset_info = {
                'n_samples': len(df),
                'n_features': len([col for col in df.columns if col.startswith('X')]),
                'n_classes': 2,  # Binary classification
                'file_path': data_path
            }
            self.report_generator.add_data_analysis_results(dataset_info, quality_report)
    
    def _run_label_creation_step(self):
        """Run disease label creation step."""
        if self.logger:
            with self.logger.log_step("label_creation", "Creating disease labels from crop stages"):
                self._execute_label_creation()
        else:
            self._execute_label_creation()
    
    def _execute_label_creation(self):
        """Execute label creation logic."""
        df = self.results['raw_data']
        spectral_columns = [col for col in df.columns if col.startswith('X')]
        
        # Create hybrid labels
        labels, label_stats = self.components['label_creator'].create_hybrid_labels(df, spectral_columns)
        
        # Add labels to dataframe
        df_with_labels = df.copy()
        df_with_labels['disease_label'] = labels
        
        # Store results
        self.results['labeled_data'] = df_with_labels
        self.results['label_statistics'] = label_stats
        
        # Log label information
        if self.logger:
            self.logger.log_performance_metrics(label_stats, "label_creation")
            self.logger.log_data_info("labeled_dataset", df_with_labels)
    
    def _run_preprocessing_step(self):
        """Run data preprocessing step."""
        if self.logger:
            with self.logger.log_step("preprocessing", "Preprocessing spectral features"):
                self._execute_preprocessing()
        else:
            self._execute_preprocessing()
    
    def _execute_preprocessing(self):
        """Execute preprocessing logic."""
        df = self.results['labeled_data']
        spectral_columns = [col for col in df.columns if col.startswith('X')]
        
        # Extract features and labels
        X_raw = self.components['preprocessor'].extract_spectral_features(df, spectral_columns)
        y = df['disease_label'].values
        
        # Store raw statistics
        before_stats = {
            'mean': np.mean(X_raw, axis=0),
            'std_dev': np.std(X_raw, axis=0),
            'min': np.min(X_raw, axis=0),
            'max': np.max(X_raw, axis=0)
        }
        
        # Apply preprocessing
        X_processed = self.components['preprocessor'].normalize_features(X_raw)
        
        # Store processed statistics
        after_stats = {
            'mean': np.mean(X_processed, axis=0),
            'std_dev': np.std(X_processed, axis=0),
            'min': np.min(X_processed, axis=0),
            'max': np.max(X_processed, axis=0)
        }
        
        # Create wavelengths array
        wavelengths = [float(col[1:]) for col in spectral_columns]
        
        # Store results
        self.results['X_raw'] = X_raw
        self.results['X_processed'] = X_processed
        self.results['y'] = y
        self.results['wavelengths'] = wavelengths
        self.results['feature_names'] = spectral_columns
        self.results['preprocessing_stats'] = {'before': before_stats, 'after': after_stats}
        
        # Log preprocessing information
        if self.logger:
            self.logger.log_data_info("processed_features", X_processed)
        
        # Create visualizations
        if self.viz_exporter:
            preprocessing_plots = self.viz_exporter.create_preprocessing_plots(
                X_raw[:100], X_processed[:100], spectral_columns[:10]
            )
            self.results['preprocessing_plots'] = preprocessing_plots
        
        # Add to report
        if self.report_generator:
            preprocessing_info = {
                'methods_applied': ['StandardScaler'],
                'n_features': len(spectral_columns),
                'wavelength_range': [min(wavelengths), max(wavelengths)]
            }
            self.report_generator.add_preprocessing_results(
                preprocessing_info, before_stats, after_stats
            )
    
    def _run_data_splitting_step(self):
        """Run data splitting step."""
        if self.logger:
            with self.logger.log_step("data_splitting", "Splitting data for training and validation"):
                self._execute_data_splitting()
        else:
            self._execute_data_splitting()
    
    def _execute_data_splitting(self):
        """Execute data splitting logic."""
        X = self.results['X_processed']
        y = self.results['y']
        
        # Split data
        X_train, X_val, y_train, y_val = self.components['data_splitter'].stratified_train_test_split(X, y)
        
        # Store results
        self.results['X_train'] = X_train
        self.results['X_val'] = X_val
        self.results['y_train'] = y_train
        self.results['y_val'] = y_val
        
        # Calculate split statistics
        split_stats = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_class_distribution': np.bincount(y_train) / len(y_train),
            'val_class_distribution': np.bincount(y_val) / len(y_val)
        }
        
        self.results['split_statistics'] = split_stats
        
        # Log split information
        if self.logger:
            self.logger.log_performance_metrics(split_stats, "data_splitting")
    
    def _run_model_training_step(self):
        """Run model training step."""
        if self.logger:
            with self.logger.log_step("model_training", "Training multiple classifier models"):
                self._execute_model_training()
        else:
            self._execute_model_training()
    
    def _execute_model_training(self):
        """Execute model training logic."""
        X_train = self.results['X_train']
        y_train = self.results['y_train']
        
        # Train multiple models
        models = {}
        training_results = {}
        
        # Random Forest
        rf_model = self.components['model_trainer'].train_random_forest(X_train, y_train)
        models['random_forest'] = rf_model
        
        # MLP
        mlp_model = self.components['model_trainer'].train_mlp_classifier(X_train, y_train)
        models['mlp'] = mlp_model
        
        # SVM
        svm_model = self.components['model_trainer'].train_svm_classifier(X_train, y_train)
        models['svm'] = svm_model
        
        # Store results
        self.results['trained_models'] = models
        self.results['training_results'] = training_results
        
        # Log model information
        if self.logger:
            for model_name, model in models.items():
                self.logger.log_model_info(model_name, model)
        
        # Add to report
        if self.report_generator:
            training_info = {
                'models_trained': list(models.keys()),
                'training_samples': len(X_train),
                'algorithms_used': ['RandomForest', 'MLP', 'SVM']
            }
            self.report_generator.add_model_training_results(training_info)
    
    def _run_model_evaluation_step(self):
        """Run model evaluation step."""
        if self.logger:
            with self.logger.log_step("model_evaluation", "Evaluating trained models"):
                self._execute_model_evaluation()
        else:
            self._execute_model_evaluation()
    
    def _execute_model_evaluation(self):
        """Execute model evaluation logic."""
        models = self.results['trained_models']
        X_val = self.results['X_val']
        y_val = self.results['y_val']
        
        # Evaluate all models
        evaluation_results = {}
        best_model = None
        best_score = 0
        
        for model_name, model in models.items():
            # Evaluate model
            metrics = self.components['model_evaluator'].evaluate_model(model, X_val, y_val)
            evaluation_results[model_name] = metrics
            
            # Track best model
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_model = model_name
            
            # Log model performance
            if self.logger:
                self.logger.log_performance_metrics(metrics, f"{model_name}_evaluation")
        
        # Store results
        self.results['evaluation_results'] = evaluation_results
        self.results['best_model_name'] = best_model
        self.results['best_model'] = models[best_model]
        
        # Create evaluation visualizations
        if self.viz_exporter:
            # Create ROC curves for all models
            for model_name, model in models.items():
                try:
                    fig = self.components['model_evaluator'].plot_roc_curve(
                        model, X_val, y_val, title=f"ROC Curve - {model_name}"
                    )
                    self.viz_exporter.save_plot(
                        fig, "roc_curve", f"{model_name}_roc_curve", "model_evaluation"
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.logger.warning(f"Failed to create ROC curve for {model_name}: {e}")
        
        # Add to report
        if self.report_generator:
            best_metrics = evaluation_results[best_model]
            self.report_generator.add_model_evaluation_results(best_metrics)
    
    def _run_feature_analysis_step(self):
        """Run feature importance analysis step."""
        if self.logger:
            with self.logger.log_step("feature_analysis", "Analyzing feature importance"):
                self._execute_feature_analysis()
        else:
            self._execute_feature_analysis()
    
    def _execute_feature_analysis(self):
        """Execute feature analysis logic."""
        models = self.results['trained_models']
        X_val = self.results['X_val']
        y_val = self.results['y_val']
        wavelengths = self.results['wavelengths']
        
        # Analyze feature importance across models
        importance_results = self.components['feature_analyzer'].analyze_multiple_models(
            models, X_val, y_val, wavelengths
        )
        
        # Create ensemble importance
        ensemble_importance = self.components['feature_analyzer'].create_ensemble_importance(importance_results)
        
        # Identify top wavelengths
        top_wavelengths = self.components['feature_analyzer'].identify_top_wavelengths(
            ensemble_importance, wavelengths, n_top=15
        )
        
        # Validate biological relevance
        relevance_analysis = self.components['feature_analyzer'].validate_biological_relevance(top_wavelengths)
        
        # Store results
        self.results['feature_importance'] = importance_results
        self.results['ensemble_importance'] = ensemble_importance
        self.results['top_wavelengths'] = top_wavelengths
        self.results['biological_relevance'] = relevance_analysis
        
        # Create feature analysis visualizations
        if self.viz_exporter:
            # Spectral importance plot
            fig = self.components['feature_analyzer'].plot_spectral_importance(
                ensemble_importance, wavelengths, "Ensemble Feature Importance"
            )
            self.viz_exporter.save_plot(
                fig, "spectral_importance", "ensemble_importance", "feature_analysis"
            )
            
            # Biological relevance plot
            fig = self.components['feature_analyzer'].plot_biological_relevance_analysis(relevance_analysis)
            self.viz_exporter.save_plot(
                fig, "biological_analysis", "relevance_validation", "feature_analysis"
            )
        
        # Add to report
        if self.report_generator:
            feature_results = {
                'top_wavelengths': top_wavelengths,
                'biological_relevance': relevance_analysis,
                'importance_methods': list(importance_results.keys())
            }
            self.report_generator.add_feature_analysis_results(feature_results)
    
    def _run_model_export_step(self):
        """Run model export and deployment preparation step."""
        if self.logger:
            with self.logger.log_step("model_export", "Exporting best model for deployment"):
                self._execute_model_export()
        else:
            self._execute_model_export()
    
    def _execute_model_export(self):
        """Execute model export logic."""
        best_model = self.results['best_model']
        best_model_name = self.results['best_model_name']
        preprocessor = self.components['preprocessor']
        wavelengths = self.results['wavelengths']
        feature_names = self.results['feature_names']
        evaluation_results = self.results['evaluation_results']
        
        # Export best model
        export_paths = self.components['model_exporter'].save_model_with_preprocessing(
            model=best_model,
            preprocessor=preprocessor,
            model_name=f"best_model_{best_model_name}",
            wavelengths=wavelengths,
            performance_metrics=evaluation_results[best_model_name],
            training_config={
                'algorithm': best_model_name,
                'session_id': self.session_id,
                'training_samples': len(self.results['X_train'])
            },
            feature_names=feature_names
        )
        
        # Create deployment package
        deployment_paths = self.components['model_exporter'].export_model_for_deployment(
            model_name=f"best_model_{best_model_name}",
            deployment_format="joblib",
            include_examples=True
        )
        
        # Store results
        self.results['model_export_paths'] = export_paths
        self.results['deployment_paths'] = deployment_paths
        
        # Test deployment
        try:
            detector = DiseaseDetector(
                model_path=export_paths['model'],
                preprocessor_path=export_paths['preprocessor'],
                metadata_path=export_paths['metadata']
            )
            
            # Test with a sample
            sample_data = self.results['X_val'][0]
            test_result = detector.predict_single_sample(sample_data)
            
            self.results['deployment_test'] = {
                'success': True,
                'test_prediction': test_result
            }
            
        except Exception as e:
            self.results['deployment_test'] = {
                'success': False,
                'error': str(e)
            }
            
            if self.logger:
                self.logger.logger.warning(f"Deployment test failed: {e}")
        
        # Add to report
        if self.report_generator:
            deployment_info = {
                'model_exported': True,
                'best_model': best_model_name,
                'export_paths': export_paths,
                'deployment_ready': self.results['deployment_test']['success']
            }
            self.report_generator.add_deployment_results(deployment_info)
    
    def _run_report_generation_step(self):
        """Run final report generation step."""
        if self.logger:
            with self.logger.log_step("report_generation", "Generating comprehensive report"):
                self._execute_report_generation()
        else:
            self._execute_report_generation()
    
    def _execute_report_generation(self):
        """Execute report generation logic."""
        if self.report_generator:
            # Generate comprehensive report
            report_path = self.report_generator.generate_comprehensive_report()
            self.results['final_report_path'] = report_path
            
            if self.logger:
                self.logger.log_file_operation("generate_report", report_path, True)
        
        if self.viz_exporter:
            # Export visualization summary
            viz_summary_path = self.viz_exporter.export_plot_summary()
            self.results['visualization_summary_path'] = viz_summary_path
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of pipeline execution results.
        
        Returns:
            Dictionary with key pipeline metrics and results
        """
        if not self.results:
            return {'status': 'Pipeline not executed'}
        
        summary = {
            'session_id': self.session_id,
            'execution_status': 'completed' if 'final_report_path' in self.results else 'incomplete',
            'dataset_info': {
                'n_samples': len(self.results.get('raw_data', [])),
                'n_features': len(self.results.get('wavelengths', [])),
                'class_distribution': self.results.get('label_statistics', {})
            },
            'model_performance': {},
            'best_model': self.results.get('best_model_name', 'unknown'),
            'biological_relevance_score': 0.0,
            'output_files': {}
        }
        
        # Add model performance
        if 'evaluation_results' in self.results:
            best_model_name = self.results.get('best_model_name')
            if best_model_name and best_model_name in self.results['evaluation_results']:
                summary['model_performance'] = self.results['evaluation_results'][best_model_name]
        
        # Add biological relevance
        if 'biological_relevance' in self.results:
            summary['biological_relevance_score'] = self.results['biological_relevance'].get('relevance_score', 0.0)
        
        # Add output file paths
        output_files = {}
        if 'final_report_path' in self.results:
            output_files['report'] = self.results['final_report_path']
        if 'model_export_paths' in self.results:
            output_files['model'] = self.results['model_export_paths']['model_dir']
        if 'deployment_paths' in self.results:
            output_files['deployment'] = list(self.results['deployment_paths'].values())
        
        summary['output_files'] = output_files
        
        return summary
    
    def save_pipeline_state(self, filepath: Optional[str] = None) -> str:
        """
        Save complete pipeline state to JSON file.
        
        Args:
            filepath: Optional path to save file
            
        Returns:
            Path to saved state file
        """
        if filepath is None:
            filepath = str(self.session_dir / f"pipeline_state_{self.session_id}.json")
        
        # Prepare serializable state
        state = {
            'session_id': self.session_id,
            'config': self.config,
            'pipeline_summary': self.get_pipeline_summary(),
            'execution_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save state
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        if self.logger:
            self.logger.log_file_operation("save_state", filepath, True)
        
        return filepath
    
    @classmethod
    def load_pipeline_state(cls, filepath: str) -> Dict[str, Any]:
        """
        Load pipeline state from JSON file.
        
        Args:
            filepath: Path to state file
            
        Returns:
            Dictionary with pipeline state
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        return state