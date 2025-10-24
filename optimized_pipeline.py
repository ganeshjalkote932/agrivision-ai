#!/usr/bin/env python3
"""
Optimized Pipeline for Crop Disease Detection with Performance Monitoring
Integrates memory profiling and CPU optimization for Task 12.1
"""

import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

# Import optimization modules
from performance_profiler import MemoryProfiler, MemoryOptimizer, profile_pipeline_function
from cpu_optimizer import CPUOptimizer, ParallelPreprocessor, ParallelModelTrainer, optimize_sklearn_for_cpu

# Import existing pipeline components
from crop_disease_detection.pipeline import CropDiseaseDetectionPipeline
from crop_disease_detection.config import DataConfig, PreprocessingConfig, ModelConfig


class OptimizedCropDiseaseDetectionPipeline(CropDiseaseDetectionPipeline):
    """
    Performance-optimized version of the crop disease detection pipeline.
    
    Integrates memory profiling, CPU optimization, and batch processing
    for efficient execution on resource-constrained hardware.
    """
    
    def __init__(self, config_file: Optional[str] = None,
                 output_dir: str = "results",
                 session_id: Optional[str] = None,
                 enable_logging: bool = True,
                 enable_visualization: bool = True,
                 enable_reporting: bool = True,
                 target_memory_gb: float = 4.0,
                 enable_profiling: bool = True):
        """
        Initialize optimized pipeline with performance monitoring.
        
        Args:
            target_memory_gb: Target memory limit in GB
            enable_profiling: Enable performance profiling
        """
        # Initialize parent pipeline
        super().__init__(config_file, output_dir, session_id, 
                        enable_logging, enable_visualization, enable_reporting)
        
        # Initialize performance optimizers
        self.target_memory_gb = target_memory_gb
        self.enable_profiling = enable_profiling
        
        if self.enable_profiling:
            self.memory_profiler = MemoryProfiler(target_memory_limit_gb=target_memory_gb)
            self.memory_profiler.enable_tracemalloc()
        else:
            self.memory_profiler = None
        
        self.cpu_optimizer = CPUOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        
        # Configure sklearn for optimal CPU usage
        optimize_sklearn_for_cpu(n_jobs=self.cpu_optimizer.max_workers)
        
        # Initialize parallel processors
        self.parallel_preprocessor = ParallelPreprocessor(self.cpu_optimizer)
        self.parallel_trainer = ParallelModelTrainer(self.cpu_optimizer)
        
        if self.logger:
            self.logger.logger.info(f"Optimized pipeline initialized with {target_memory_gb}GB memory limit")
    
    def run_complete_pipeline(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Run optimized complete pipeline with performance monitoring."""
        
        if self.memory_profiler:
            self.memory_profiler.take_snapshot("pipeline_start")
        
        try:
            # Run the complete pipeline with optimization
            results = self._run_optimized_pipeline(dataset_path)
            
            # Generate performance report
            if self.memory_profiler:
                self.memory_profiler.take_snapshot("pipeline_complete")
                performance_dir = self.session_dir / "performance_analysis"
                self.memory_profiler.generate_report(str(performance_dir))
                results['performance_report_dir'] = str(performance_dir)
            
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Optimized pipeline execution failed: {str(e)}")
            raise
        finally:
            # Cleanup
            if self.memory_profiler:
                self.memory_profiler.disable_tracemalloc()
    
    def _run_optimized_pipeline(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Internal optimized pipeline execution."""
        
        if not self.is_initialized:
            self.initialize_components()
        
        # Use provided dataset path or config default
        data_path = dataset_path or self.config['data']['dataset_path']
        
        if self.logger:
            self.logger.logger.info("="*60)
            self.logger.logger.info("STARTING OPTIMIZED PIPELINE EXECUTION")
            self.logger.logger.info("="*60)
        
        # Step 1: Optimized Data Loading
        self._run_optimized_data_loading_step(data_path)
        
        # Step 2: Optimized Label Creation
        self._run_optimized_label_creation_step()
        
        # Step 3: Optimized Preprocessing
        self._run_optimized_preprocessing_step()
        
        # Step 4: Optimized Data Splitting
        self._run_optimized_data_splitting_step()
        
        # Step 5: Optimized Model Training
        self._run_optimized_model_training_step()
        
        # Step 6: Optimized Model Evaluation
        self._run_optimized_model_evaluation_step()
        
        # Step 7: Feature Analysis
        self._run_feature_analysis_step()
        
        # Step 8: Model Export
        self._run_model_export_step()
        
        # Step 9: Report Generation
        self._run_report_generation_step()
        
        if self.logger:
            self.logger.logger.info("="*60)
            self.logger.logger.info("OPTIMIZED PIPELINE EXECUTION COMPLETED")
            self.logger.logger.info("="*60)
        
        return self.results