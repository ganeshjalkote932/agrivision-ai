"""
Model training module for hyperspectral crop disease detection.

This module implements comprehensive model training with MLP, Random Forest, and SVM
classifiers optimized for >95% accuracy on hyperspectral data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import time
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from .base import BaseProcessor
from .config import ModelConfig, HyperparameterConfig


class ModelTrainer(BaseProcessor):
    """
    Comprehensive model training system for hyperspectral crop disease detection.
    
    Implements MLP, Random Forest, and SVM classifiers with hyperparameter optimization,
    cross-validation, and performance monitoring optimized for hardware constraints.
    """
    
    def __init__(self, model_config: ModelConfig, hyperparameter_config: HyperparameterConfig):
        """
        Initialize the model trainer.
        
        Args:
            model_config: ModelConfig instance with model parameters
            hyperparameter_config: HyperparameterConfig for optimization
        """
        super().__init__()
        self.model_config = model_config
        self.hyperparameter_config = hyperparameter_config
        
        # Store trained models
        self.models = {}
        self.best_models = {}
        self.training_results = {}
        self.hyperparameter_results = {}
        
    def train_mlp_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> MLPClassifier:
        """
        Train Multi-Layer Perceptron classifier optimized for hyperspectral data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Trained MLPClassifier
        """
        self.logger.info("Training Multi-Layer Perceptron (MLP) classifier...")
        
        start_time = time.time()
        
        # Configure MLP with hardware optimizations
        mlp = MLPClassifier(
            hidden_layer_sizes=self.model_config.mlp_hidden_layers,
            activation=self.model_config.mlp_activation,
            solver=self.model_config.mlp_solver,
            learning_rate=self.model_config.mlp_learning_rate,
            max_iter=self.model_config.mlp_max_iter,
            early_stopping=self.model_config.mlp_early_stopping,
            validation_fraction=self.model_config.mlp_validation_fraction,
            random_state=self.model_config.random_state,
            n_iter_no_change=10,  # Early stopping patience
            tol=1e-4,  # Convergence tolerance
            warm_start=False,  # Don't reuse previous solution
            batch_size='auto'  # Automatic batch sizing for memory efficiency
        )
        
        # Train the model
        self.logger.info(f"MLP Architecture: {self.model_config.mlp_hidden_layers}")
        self.logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        
        mlp.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate on training set
        train_accuracy = mlp.score(X_train, y_train)
        
        # Evaluate on validation set if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_accuracy = mlp.score(X_val, y_val)
        
        # Store training information
        training_info = {
            'model_type': 'MLP',
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'n_iterations': mlp.n_iter_,
            'convergence': mlp.n_iter_ < self.model_config.mlp_max_iter,
            'architecture': self.model_config.mlp_hidden_layers,
            'n_parameters': self._count_mlp_parameters(mlp, X_train.shape[1])
        }
        
        self.training_results['mlp'] = training_info
        self.models['mlp'] = mlp
        
        self.logger.info(f"MLP training completed in {training_time:.2f}s")
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
        if val_accuracy is not None:
            self.logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        self.logger.info(f"Converged in {mlp.n_iter_} iterations")
        
        return mlp
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> RandomForestClassifier:
        """
        Train Random Forest classifier optimized for hyperspectral data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Trained RandomForestClassifier
        """
        self.logger.info("Training Random Forest classifier...")
        
        start_time = time.time()
        
        # Configure Random Forest with CPU optimization
        rf = RandomForestClassifier(
            n_estimators=self.model_config.rf_n_estimators,
            max_depth=self.model_config.rf_max_depth,
            min_samples_split=self.model_config.rf_min_samples_split,
            min_samples_leaf=self.model_config.rf_min_samples_leaf,
            max_features=self.model_config.rf_max_features,
            bootstrap=self.model_config.rf_bootstrap,
            random_state=self.model_config.random_state,
            n_jobs=self.model_config.n_jobs,  # Use all CPU cores
            oob_score=True,  # Out-of-bag score for validation
            warm_start=False,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.logger.info(f"RF Configuration: {self.model_config.rf_n_estimators} trees")
        self.logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        
        # Train the model
        rf.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate on training set
        train_accuracy = rf.score(X_train, y_train)
        
        # Evaluate on validation set if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_accuracy = rf.score(X_val, y_val)
        
        # Get out-of-bag score
        oob_accuracy = rf.oob_score_
        
        # Store training information
        training_info = {
            'model_type': 'RandomForest',
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'oob_accuracy': oob_accuracy,
            'n_estimators': self.model_config.rf_n_estimators,
            'max_depth': self.model_config.rf_max_depth,
            'feature_importances': rf.feature_importances_.tolist()
        }
        
        self.training_results['random_forest'] = training_info
        self.models['random_forest'] = rf
        
        self.logger.info(f"Random Forest training completed in {training_time:.2f}s")
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
        self.logger.info(f"OOB accuracy: {oob_accuracy:.4f}")
        if val_accuracy is not None:
            self.logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return rf
    
    def train_svm_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> SVC:
        """
        Train Support Vector Machine classifier optimized for hyperspectral data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Trained SVC
        """
        self.logger.info("Training Support Vector Machine (SVM) classifier...")
        
        start_time = time.time()
        
        # Configure SVM with memory optimization
        svm = SVC(
            kernel=self.model_config.svm_kernel,
            C=self.model_config.svm_C,
            gamma=self.model_config.svm_gamma,
            probability=self.model_config.svm_probability,
            random_state=self.model_config.random_state,
            class_weight='balanced',  # Handle class imbalance
            cache_size=1000,  # Increase cache for better performance
            max_iter=10000,  # Increase max iterations
            tol=1e-3  # Convergence tolerance
        )
        
        self.logger.info(f"SVM Configuration: {self.model_config.svm_kernel} kernel, C={self.model_config.svm_C}")
        self.logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        
        # For large datasets, consider using a subset for SVM training
        if X_train.shape[0] > 5000:
            self.logger.info("Large dataset detected, using subset for SVM training...")
            # Use stratified sampling to maintain class balance
            from sklearn.model_selection import StratifiedShuffleSplit
            
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=5000, random_state=42)
            subset_idx, _ = next(splitter.split(X_train, y_train))
            
            X_train_subset = X_train[subset_idx]
            y_train_subset = y_train[subset_idx]
            
            self.logger.info(f"Using subset: {X_train_subset.shape[0]} samples")
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        # Train the model
        svm.fit(X_train_subset, y_train_subset)
        
        training_time = time.time() - start_time
        
        # Evaluate on full training set
        train_accuracy = svm.score(X_train, y_train)
        
        # Evaluate on validation set if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_accuracy = svm.score(X_val, y_val)
        
        # Store training information
        training_info = {
            'model_type': 'SVM',
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'kernel': self.model_config.svm_kernel,
            'C': self.model_config.svm_C,
            'gamma': self.model_config.svm_gamma,
            'n_support_vectors': svm.n_support_.tolist(),
            'support_vector_ratio': svm.n_support_.sum() / len(X_train_subset)
        }
        
        self.training_results['svm'] = training_info
        self.models['svm'] = svm
        
        self.logger.info(f"SVM training completed in {training_time:.2f}s")
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
        if val_accuracy is not None:
            self.logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        self.logger.info(f"Support vectors: {svm.n_support_.sum()}/{len(X_train_subset)} ({training_info['support_vector_ratio']:.2%})")
        
        return svm
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                model_type: str, cv_folds: int = None) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model ('mlp', 'random_forest', 'svm')
            cv_folds: Number of CV folds (default from config)
            
        Returns:
            Dictionary with optimization results
        """
        if cv_folds is None:
            cv_folds = self.hyperparameter_config.cv_folds
            
        self.logger.info(f"Optimizing hyperparameters for {model_type}...")
        
        start_time = time.time()
        
        # Get model and parameter grid
        if model_type == 'mlp':
            model = MLPClassifier(
                random_state=self.model_config.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=500  # Reduced for grid search
            )
            param_grid = self.hyperparameter_config.mlp_param_grid
            
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                random_state=self.model_config.random_state,
                n_jobs=self.hyperparameter_config.n_jobs,
                oob_score=True
            )
            param_grid = self.hyperparameter_config.rf_param_grid
            
        elif model_type == 'svm':
            model = SVC(
                random_state=self.model_config.random_state,
                probability=True,
                cache_size=1000
            )
            param_grid = self.hyperparameter_config.svm_param_grid
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Configure GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.hyperparameter_config.scoring,
            cv=cv_folds,
            n_jobs=self.hyperparameter_config.n_jobs,
            verbose=self.hyperparameter_config.verbose,
            return_train_score=True,
            refit=True
        )
        
        # For SVM with large datasets, use a subset for hyperparameter optimization
        if model_type == 'svm' and X_train.shape[0] > 2000:
            from sklearn.model_selection import StratifiedShuffleSplit
            
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=2000, random_state=42)
            subset_idx, _ = next(splitter.split(X_train, y_train))
            
            X_subset = X_train[subset_idx]
            y_subset = y_train[subset_idx]
            
            self.logger.info(f"Using subset for SVM hyperparameter optimization: {len(X_subset)} samples")
        else:
            X_subset = X_train
            y_subset = y_train
        
        # Perform grid search
        self.logger.info(f"Grid search with {len(param_grid)} parameter combinations...")
        grid_search.fit(X_subset, y_subset)
        
        optimization_time = time.time() - start_time
        
        # Get best model and retrain on full dataset if subset was used
        best_model = grid_search.best_estimator_
        
        if model_type == 'svm' and X_train.shape[0] > 2000:
            self.logger.info("Retraining best SVM model on full dataset...")
            best_model.fit(X_train, y_train)
        
        # Store optimization results
        optimization_results = {
            'model_type': model_type,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'optimization_time': optimization_time,
            'n_combinations_tested': len(grid_search.cv_results_['params']),
            'cv_results': {
                'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                'params': grid_search.cv_results_['params']
            }
        }
        
        self.hyperparameter_results[model_type] = optimization_results
        self.best_models[model_type] = best_model
        
        self.logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f}s")
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return optimization_results
    
    def select_best_model(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[str, Any, Dict[str, float]]:
        """
        Select the best model based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (best_model_name, best_model, performance_scores)
        """
        self.logger.info("Selecting best model based on validation performance...")
        
        model_scores = {}
        
        # Evaluate all trained models
        for model_name, model in self.models.items():
            if model is not None:
                val_accuracy = model.score(X_val, y_val)
                y_pred = model.predict(X_val)
                
                # Calculate additional metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
                
                model_scores[model_name] = {
                    'accuracy': val_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                self.logger.info(f"{model_name}: Accuracy={val_accuracy:.4f}, F1={f1:.4f}")
        
        # Also evaluate hyperparameter-optimized models if available
        for model_name, model in self.best_models.items():
            if model is not None:
                val_accuracy = model.score(X_val, y_val)
                y_pred = model.predict(X_val)
                
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
                
                optimized_name = f"{model_name}_optimized"
                model_scores[optimized_name] = {
                    'accuracy': val_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                self.logger.info(f"{optimized_name}: Accuracy={val_accuracy:.4f}, F1={f1:.4f}")
        
        # Select best model based on F1 score (balanced metric)
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['f1_score'])
        best_score = model_scores[best_model_name]
        
        # Get the actual model object
        if best_model_name.endswith('_optimized'):
            base_name = best_model_name.replace('_optimized', '')
            best_model = self.best_models[base_name]
        else:
            best_model = self.models[best_model_name]
        
        self.logger.info(f"Best model selected: {best_model_name}")
        self.logger.info(f"Best performance: Accuracy={best_score['accuracy']:.4f}, F1={best_score['f1_score']:.4f}")
        
        return best_model_name, best_model, model_scores
    
    def _count_mlp_parameters(self, mlp: MLPClassifier, n_features: int) -> int:
        """Count the number of parameters in an MLP model."""
        n_params = 0
        layer_sizes = [n_features] + list(mlp.hidden_layer_sizes) + [len(mlp.classes_)]
        
        for i in range(len(layer_sizes) - 1):
            # Weights + biases
            n_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        
        return n_params
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of training results.
        
        Returns:
            Dictionary with complete training analysis
        """
        return {
            'training_results': self.training_results,
            'hyperparameter_results': self.hyperparameter_results,
            'models_trained': list(self.models.keys()),
            'optimized_models': list(self.best_models.keys())
        }
    
    def save_models(self, output_dir: str) -> Dict[str, str]:
        """
        Save all trained models to files.
        
        Args:
            output_dir: Directory to save models
            
        Returns:
            Dictionary mapping model names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_models = {}
        
        # Save regular models
        for model_name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
                joblib.dump(model, model_path)
                saved_models[model_name] = model_path
                self.logger.info(f"Saved {model_name} model to: {model_path}")
        
        # Save optimized models
        for model_name, model in self.best_models.items():
            if model is not None:
                model_path = os.path.join(output_dir, f"{model_name}_optimized_model.joblib")
                joblib.dump(model, model_path)
                saved_models[f"{model_name}_optimized"] = model_path
                self.logger.info(f"Saved {model_name}_optimized model to: {model_path}")
        
        # Save training results
        results_path = os.path.join(output_dir, "training_results.joblib")
        joblib.dump(self.get_training_summary(), results_path)
        saved_models['training_results'] = results_path
        
        return saved_models
    
    def process(self, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray) -> Tuple[str, Any, Dict[str, Any]]:
        """
        Main processing method for model training.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (best_model_name, best_model, training_summary)
        """
        # Train all models
        self.train_mlp_classifier(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_svm_classifier(X_train, y_train, X_val, y_val)
        
        # Select best model
        best_model_name, best_model, model_scores = self.select_best_model(X_val, y_val)
        
        # Get training summary
        training_summary = self.get_training_summary()
        training_summary['model_comparison'] = model_scores
        training_summary['best_model'] = best_model_name
        
        return best_model_name, best_model, training_summary