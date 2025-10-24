#!/usr/bin/env python3
"""
CPU Optimization Module for Crop Disease Detection Pipeline
Implements multi-core processing optimizations for Ryzen 7 processor
"""

import multiprocessing as mp
import concurrent.futures
import numpy as np
import pandas as pd
from typing import List, Callable, Any, Optional, Tuple
import psutil
import time
from functools import partial
import warnings
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
import joblib

class CPUOptimizer:
    """CPU optimization utilities for multi-core processing"""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize CPU optimizer
        
        Args:
            max_workers: Maximum number of worker processes (default: auto-detect)
        """
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False)
        
        # Optimize for Ryzen 7 architecture (typically 8 cores, 16 threads)
        if max_workers is None:
            # Use physical cores for CPU-intensive tasks, leave some headroom
            self.max_workers = max(1, min(self.physical_cores - 1, 8))
        else:
            self.max_workers = min(max_workers, self.cpu_count)
        
        print(f"🖥️  CPU Optimizer Initialized")
        print(f"   Physical Cores: {self.physical_cores}")
        print(f"   Logical Cores: {self.cpu_count}")
        print(f"   Max Workers: {self.max_workers}")
        
        # Set optimal thread settings for numpy/sklearn
        self._configure_threading()
    
    def _configure_threading(self):
        """Configure optimal threading for numerical libraries"""
        import os
        
        # Set environment variables for optimal threading
        os.environ['OMP_NUM_THREADS'] = str(self.max_workers)
        os.environ['MKL_NUM_THREADS'] = str(self.max_workers)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.max_workers)
        
        # Configure joblib for sklearn
        joblib.parallel.DEFAULT_N_JOBS = self.max_workers
        
        print(f"   Threading configured for {self.max_workers} threads")
    
    def parallel_apply(self, data: np.ndarray, func: Callable, 
                      chunk_size: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Apply function to data in parallel chunks
        
        Args:
            data: Input data array
            func: Function to apply to each chunk
            chunk_size: Size of each chunk (auto-calculated if None)
            **kwargs: Additional arguments for the function
        
        Returns:
            Processed data array
        """
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.max_workers * 2))
        
        # Split data into chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        print(f"🔄 Processing {len(data)} samples in {len(chunks)} chunks using {self.max_workers} workers")
        
        start_time = time.time()
        
        # Process chunks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create partial function with kwargs
            process_func = partial(func, **kwargs)
            
            # Submit all chunks
            futures = [executor.submit(process_func, chunk) for chunk in chunks]
            
            # Collect results
            results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    results.append((i, result))
                except Exception as e:
                    print(f"❌ Error processing chunk {i}: {e}")
                    raise
        
        # Sort results by original order and concatenate
        results.sort(key=lambda x: x[0])
        processed_data = np.concatenate([result[1] for result in results], axis=0)
        
        duration = time.time() - start_time
        throughput = len(data) / duration
        
        print(f"   ✅ Processed {len(data)} samples in {duration:.2f}s ({throughput:.0f} samples/s)")
        
        return processed_data
    
    def parallel_cross_validation(self, estimator: BaseEstimator, X: np.ndarray, 
                                y: np.ndarray, cv: int = 5, 
                                scoring: str = 'accuracy') -> Tuple[np.ndarray, float, float]:
        """
        Perform parallel cross-validation
        
        Args:
            estimator: Sklearn estimator
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            scoring: Scoring metric
        
        Returns:
            Tuple of (scores, mean_score, std_score)
        """
        print(f"🔄 Running {cv}-fold cross-validation with {self.max_workers} parallel jobs")
        
        start_time = time.time()
        
        # Use joblib backend for sklearn parallelization
        with joblib.parallel_backend('threading', n_jobs=self.max_workers):
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=self.max_workers)
        
        duration = time.time() - start_time
        mean_score = scores.mean()
        std_score = scores.std()
        
        print(f"   ✅ CV completed in {duration:.2f}s: {mean_score:.4f} ± {std_score:.4f}")
        
        return scores, mean_score, std_score
    
    def parallel_hyperparameter_search(self, estimator: BaseEstimator, 
                                     param_grid: dict, X: np.ndarray, 
                                     y: np.ndarray, cv: int = 3) -> dict:
        """
        Perform parallel hyperparameter search
        
        Args:
            estimator: Sklearn estimator
            param_grid: Parameter grid to search
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
        
        Returns:
            Best parameters dictionary
        """
        from sklearn.model_selection import GridSearchCV
        
        print(f"🔍 Hyperparameter search with {self.max_workers} parallel jobs")
        
        # Calculate total combinations
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        
        print(f"   Testing {total_combinations} parameter combinations")
        
        start_time = time.time()
        
        # Use parallel grid search
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=self.max_workers,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        duration = time.time() - start_time
        
        print(f"   ✅ Search completed in {duration:.2f}s")
        print(f"   Best score: {grid_search.best_score_:.4f}")
        print(f"   Best params: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def optimize_batch_size_for_cpu(self, data_shape: Tuple[int, ...], 
                                   target_cpu_utilization: float = 0.8) -> int:
        """
        Calculate optimal batch size for CPU processing
        
        Args:
            data_shape: Shape of input data
            target_cpu_utilization: Target CPU utilization (0.0-1.0)
        
        Returns:
            Optimal batch size
        """
        # Base batch size on available cores and data size
        n_samples = data_shape[0]
        
        # Start with a batch size that gives each core some work
        base_batch_size = max(1, n_samples // (self.max_workers * 4))
        
        # Adjust based on data complexity (number of features)
        if len(data_shape) > 1:
            n_features = np.prod(data_shape[1:])
            # Larger feature spaces need smaller batches to fit in CPU cache
            if n_features > 1000:
                base_batch_size = max(1, base_batch_size // 2)
            elif n_features > 10000:
                base_batch_size = max(1, base_batch_size // 4)
        
        # Ensure batch size is reasonable
        optimal_batch_size = max(32, min(base_batch_size, 1024))
        
        print(f"🎯 Optimal batch size for CPU: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def monitor_cpu_usage(self, duration: float = 10.0) -> dict:
        """
        Monitor CPU usage for a specified duration
        
        Args:
            duration: Monitoring duration in seconds
        
        Returns:
            CPU usage statistics
        """
        print(f"📊 Monitoring CPU usage for {duration}s...")
        
        cpu_percentages = []
        per_cpu_percentages = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_percentages.append(cpu_percent)
            
            # Per-CPU usage
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            per_cpu_percentages.append(per_cpu)
        
        # Calculate statistics
        stats = {
            'average_cpu_usage': np.mean(cpu_percentages),
            'max_cpu_usage': np.max(cpu_percentages),
            'min_cpu_usage': np.min(cpu_percentages),
            'cpu_usage_std': np.std(cpu_percentages),
            'per_cpu_average': np.mean(per_cpu_percentages, axis=0).tolist(),
            'cpu_utilization_efficiency': np.mean(cpu_percentages) / 100.0
        }
        
        print(f"   Average CPU usage: {stats['average_cpu_usage']:.1f}%")
        print(f"   CPU efficiency: {stats['cpu_utilization_efficiency']:.2f}")
        
        return stats

class ParallelPreprocessor:
    """Parallel preprocessing utilities"""
    
    def __init__(self, cpu_optimizer: CPUOptimizer):
        self.cpu_optimizer = cpu_optimizer
    
    def parallel_normalize(self, data: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Normalize data in parallel chunks"""
        
        def normalize_chunk(chunk: np.ndarray, method: str) -> np.ndarray:
            if method == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                return scaler.fit_transform(chunk)
            elif method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                return scaler.fit_transform(chunk)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        return self.cpu_optimizer.parallel_apply(data, normalize_chunk, method=method)
    
    def parallel_outlier_detection(self, data: np.ndarray, 
                                 contamination: float = 0.1) -> np.ndarray:
        """Detect outliers in parallel"""
        
        def detect_outliers_chunk(chunk: np.ndarray, contamination: float) -> np.ndarray:
            from sklearn.ensemble import IsolationForest
            
            detector = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = detector.fit_predict(chunk)
            
            # Convert to boolean mask (True for inliers)
            return outlier_labels == 1
        
        return self.cpu_optimizer.parallel_apply(data, detect_outliers_chunk, 
                                               contamination=contamination)

class ParallelModelTrainer:
    """Parallel model training utilities"""
    
    def __init__(self, cpu_optimizer: CPUOptimizer):
        self.cpu_optimizer = cpu_optimizer
    
    def train_multiple_models_parallel(self, models: List[BaseEstimator], 
                                     X: np.ndarray, y: np.ndarray) -> List[BaseEstimator]:
        """Train multiple models in parallel"""
        
        def train_model(model_data: Tuple[BaseEstimator, np.ndarray, np.ndarray]) -> BaseEstimator:
            model, X_train, y_train = model_data
            return model.fit(X_train, y_train)
        
        print(f"🚀 Training {len(models)} models in parallel")
        
        start_time = time.time()
        
        # Prepare data for each model
        model_data = [(model, X, y) for model in models]
        
        # Train models in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(models), self.cpu_optimizer.max_workers)) as executor:
            trained_models = list(executor.map(train_model, model_data))
        
        duration = time.time() - start_time
        print(f"   ✅ Trained {len(models)} models in {duration:.2f}s")
        
        return trained_models
    
    def parallel_ensemble_training(self, base_estimator: BaseEstimator, 
                                 X: np.ndarray, y: np.ndarray, 
                                 n_estimators: int = 10) -> List[BaseEstimator]:
        """Train ensemble of models with different random states"""
        
        # Create models with different random states
        models = []
        for i in range(n_estimators):
            model = base_estimator.__class__(**base_estimator.get_params())
            if hasattr(model, 'random_state'):
                model.set_params(random_state=i)
            models.append(model)
        
        return self.train_multiple_models_parallel(models, X, y)

def optimize_sklearn_for_cpu(n_jobs: int = -1):
    """Configure sklearn for optimal CPU usage"""
    import os
    
    # Set sklearn to use all available cores
    os.environ['SKLEARN_N_JOBS'] = str(n_jobs)
    
    # Configure BLAS libraries for optimal performance
    os.environ['OPENBLAS_NUM_THREADS'] = str(psutil.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(psutil.cpu_count())
    
    print(f"🔧 Sklearn configured for optimal CPU usage ({n_jobs} jobs)")

if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing CPU Optimizer")
    
    # Initialize optimizer
    cpu_optimizer = CPUOptimizer()
    
    # Test parallel processing
    print("\n📊 Testing parallel data processing...")
    test_data = np.random.random((10000, 131)).astype(np.float32)
    
    def simple_process(chunk):
        # Simulate some processing
        return chunk * 2 + 1
    
    start_time = time.time()
    result = cpu_optimizer.parallel_apply(test_data, simple_process)
    parallel_time = time.time() - start_time
    
    # Compare with sequential processing
    start_time = time.time()
    sequential_result = simple_process(test_data)
    sequential_time = time.time() - start_time
    
    print(f"   Parallel time: {parallel_time:.3f}s")
    print(f"   Sequential time: {sequential_time:.3f}s")
    print(f"   Speedup: {sequential_time/parallel_time:.2f}x")
    print(f"   Results match: {np.allclose(result, sequential_result)}")
    
    # Test CPU monitoring
    print("\n📈 Testing CPU monitoring...")
    stats = cpu_optimizer.monitor_cpu_usage(duration=2.0)
    
    print("✅ CPU optimizer test completed")