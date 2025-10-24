#!/usr/bin/env python3
"""
Performance Profiler and Memory Optimizer for Crop Disease Detection Pipeline
Implements Task 12.1: Optimize memory usage and processing speed
"""

import psutil
import time
import gc
import sys
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a specific point in time"""
    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory percentage
    available_mb: float  # Available system memory in MB
    step_name: str
    peak_tracemalloc_mb: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for a processing step"""
    step_name: str
    duration_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    data_size_mb: Optional[float] = None
    throughput_mb_per_sec: Optional[float] = None

class MemoryProfiler:
    """Advanced memory profiler with optimization recommendations"""
    
    def __init__(self, target_memory_limit_gb: float = 4.0):
        """
        Initialize memory profiler
        
        Args:
            target_memory_limit_gb: Target memory limit in GB (default: 4GB for hardware constraint)
        """
        self.target_memory_limit_mb = target_memory_limit_gb * 1024
        self.snapshots: List[MemorySnapshot] = []
        self.metrics: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.tracemalloc_enabled = False
        
        # System info
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.cpu_count = psutil.cpu_count()
        
        print(f"🔧 Performance Profiler Initialized")
        print(f"   Target Memory Limit: {target_memory_limit_gb:.1f} GB")
        print(f"   System Memory: {self.system_memory_gb:.1f} GB")
        print(f"   CPU Cores: {self.cpu_count}")
        print(f"   Memory Headroom: {self.system_memory_gb - target_memory_limit_gb:.1f} GB")
    
    def enable_tracemalloc(self):
        """Enable detailed memory tracing"""
        if not self.tracemalloc_enabled:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            print("📊 Detailed memory tracing enabled")
    
    def disable_tracemalloc(self):
        """Disable detailed memory tracing"""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'rss_mb': memory_info.rss / (1024**2),
            'vms_mb': memory_info.vms / (1024**2),
            'percent': process.memory_percent(),
            'available_mb': system_memory.available / (1024**2),
            'used_mb': system_memory.used / (1024**2),
            'total_mb': system_memory.total / (1024**2)
        }
    
    def take_snapshot(self, step_name: str) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        memory_stats = self.get_memory_usage()
        
        peak_tracemalloc_mb = None
        if self.tracemalloc_enabled:
            current, peak = tracemalloc.get_traced_memory()
            peak_tracemalloc_mb = peak / (1024**2)
        
        snapshot = MemorySnapshot(
            timestamp=time.time() - self.start_time,
            rss_mb=memory_stats['rss_mb'],
            vms_mb=memory_stats['vms_mb'],
            percent=memory_stats['percent'],
            available_mb=memory_stats['available_mb'],
            step_name=step_name,
            peak_tracemalloc_mb=peak_tracemalloc_mb
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    @contextmanager
    def profile_step(self, step_name: str, data_size_mb: Optional[float] = None):
        """Context manager to profile a processing step"""
        print(f"🔍 Profiling: {step_name}")
        
        # Take before snapshot
        before_snapshot = self.take_snapshot(f"{step_name}_start")
        start_time = time.time()
        
        # Start CPU monitoring
        cpu_percent_start = psutil.cpu_percent(interval=None)
        
        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            
            # Take after snapshot
            after_snapshot = self.take_snapshot(f"{step_name}_end")
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Calculate throughput if data size provided
            throughput = None
            if data_size_mb and duration > 0:
                throughput = data_size_mb / duration
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                step_name=step_name,
                duration_seconds=duration,
                memory_before_mb=before_snapshot.rss_mb,
                memory_after_mb=after_snapshot.rss_mb,
                memory_peak_mb=max(before_snapshot.rss_mb, after_snapshot.rss_mb),
                memory_delta_mb=after_snapshot.rss_mb - before_snapshot.rss_mb,
                cpu_percent=cpu_percent,
                data_size_mb=data_size_mb,
                throughput_mb_per_sec=throughput
            )
            
            self.metrics.append(metrics)
            
            # Print step summary
            self._print_step_summary(metrics)
            
            # Check memory warnings
            self._check_memory_warnings(after_snapshot)
    
    def _print_step_summary(self, metrics: PerformanceMetrics):
        """Print summary for a profiled step"""
        print(f"   ⏱️  Duration: {metrics.duration_seconds:.2f}s")
        print(f"   💾 Memory: {metrics.memory_before_mb:.1f} → {metrics.memory_after_mb:.1f} MB "
              f"(Δ{metrics.memory_delta_mb:+.1f} MB)")
        print(f"   🖥️  CPU: {metrics.cpu_percent:.1f}%")
        
        if metrics.throughput_mb_per_sec:
            print(f"   📈 Throughput: {metrics.throughput_mb_per_sec:.1f} MB/s")
        
        # Memory efficiency warning
        if metrics.memory_after_mb > self.target_memory_limit_mb * 0.8:
            print(f"   ⚠️  Memory usage approaching limit ({metrics.memory_after_mb:.1f}/{self.target_memory_limit_mb:.1f} MB)")
    
    def _check_memory_warnings(self, snapshot: MemorySnapshot):
        """Check for memory usage warnings"""
        if snapshot.rss_mb > self.target_memory_limit_mb:
            print(f"🚨 MEMORY LIMIT EXCEEDED: {snapshot.rss_mb:.1f} MB > {self.target_memory_limit_mb:.1f} MB")
        elif snapshot.rss_mb > self.target_memory_limit_mb * 0.9:
            print(f"⚠️  Memory usage critical: {snapshot.rss_mb:.1f} MB (90% of limit)")
        elif snapshot.rss_mb > self.target_memory_limit_mb * 0.8:
            print(f"⚠️  Memory usage high: {snapshot.rss_mb:.1f} MB (80% of limit)")
    
    def force_garbage_collection(self) -> float:
        """Force garbage collection and return memory freed"""
        before = self.get_memory_usage()['rss_mb']
        
        # Multiple GC passes for thorough cleanup
        for _ in range(3):
            gc.collect()
        
        after = self.get_memory_usage()['rss_mb']
        freed_mb = before - after
        
        if freed_mb > 1.0:  # Only report if significant memory freed
            print(f"🗑️  Garbage collection freed {freed_mb:.1f} MB")
        
        return freed_mb
    
    def optimize_pandas_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas DataFrame memory usage"""
        print(f"🔧 Optimizing DataFrame memory usage...")
        
        before_mb = df.memory_usage(deep=True).sum() / (1024**2)
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns (strings)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        after_mb = df.memory_usage(deep=True).sum() / (1024**2)
        saved_mb = before_mb - after_mb
        
        print(f"   Memory: {before_mb:.1f} → {after_mb:.1f} MB (saved {saved_mb:.1f} MB)")
        
        return df
    
    def generate_report(self, output_dir: str = "performance_analysis"):
        """Generate comprehensive performance report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📊 Generating performance report in {output_path}")
        
        # Create visualizations
        self._create_memory_timeline_plot(output_path)
        self._create_performance_summary_plot(output_path)
        self._create_optimization_recommendations(output_path)
        
        # Save raw data
        self._save_raw_data(output_path)
        
        print(f"✅ Performance report generated in {output_path}")
    
    def _create_memory_timeline_plot(self, output_path: Path):
        """Create memory usage timeline plot"""
        if not self.snapshots:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Extract data
        timestamps = [s.timestamp for s in self.snapshots]
        memory_usage = [s.rss_mb for s in self.snapshots]
        step_names = [s.step_name for s in self.snapshots]
        
        # Plot memory usage
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, memory_usage, 'b-', linewidth=2, label='Memory Usage')
        plt.axhline(y=self.target_memory_limit_mb, color='r', linestyle='--', 
                   label=f'Target Limit ({self.target_memory_limit_mb:.0f} MB)')
        plt.axhline(y=self.target_memory_limit_mb * 0.8, color='orange', linestyle='--', 
                   label='80% Warning')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot step durations
        plt.subplot(2, 1, 2)
        if self.metrics:
            step_names = [m.step_name for m in self.metrics]
            durations = [m.duration_seconds for m in self.metrics]
            
            bars = plt.bar(range(len(step_names)), durations, alpha=0.7)
            plt.xlabel('Processing Steps')
            plt.ylabel('Duration (seconds)')
            plt.title('Step Processing Times')
            plt.xticks(range(len(step_names)), step_names, rotation=45, ha='right')
            
            # Color bars by duration (red for slow steps)
            max_duration = max(durations) if durations else 1
            for bar, duration in zip(bars, durations):
                intensity = duration / max_duration
                bar.set_color(plt.cm.Reds(0.3 + 0.7 * intensity))
        
        plt.tight_layout()
        plt.savefig(output_path / 'memory_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_summary_plot(self, output_path: Path):
        """Create performance summary visualization"""
        if not self.metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Memory delta by step
        step_names = [m.step_name for m in self.metrics]
        memory_deltas = [m.memory_delta_mb for m in self.metrics]
        
        axes[0, 0].bar(range(len(step_names)), memory_deltas, 
                      color=['red' if x > 0 else 'green' for x in memory_deltas])
        axes[0, 0].set_title('Memory Change by Step')
        axes[0, 0].set_ylabel('Memory Delta (MB)')
        axes[0, 0].set_xticks(range(len(step_names)))
        axes[0, 0].set_xticklabels(step_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Processing time vs memory usage
        durations = [m.duration_seconds for m in self.metrics]
        peak_memory = [m.memory_peak_mb for m in self.metrics]
        
        scatter = axes[0, 1].scatter(durations, peak_memory, 
                                   c=range(len(durations)), cmap='viridis', s=100)
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].set_ylabel('Peak Memory (MB)')
        axes[0, 1].set_title('Processing Time vs Memory Usage')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add step labels to scatter plot
        for i, name in enumerate(step_names):
            axes[0, 1].annotate(name, (durations[i], peak_memory[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Throughput analysis (if available)
        throughputs = [m.throughput_mb_per_sec for m in self.metrics if m.throughput_mb_per_sec]
        throughput_steps = [m.step_name for m in self.metrics if m.throughput_mb_per_sec]
        
        if throughputs:
            axes[1, 0].bar(range(len(throughput_steps)), throughputs, alpha=0.7)
            axes[1, 0].set_title('Data Processing Throughput')
            axes[1, 0].set_ylabel('Throughput (MB/s)')
            axes[1, 0].set_xticks(range(len(throughput_steps)))
            axes[1, 0].set_xticklabels(throughput_steps, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No throughput data available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Data Processing Throughput')
        
        # CPU usage by step
        cpu_usage = [m.cpu_percent for m in self.metrics]
        
        axes[1, 1].bar(range(len(step_names)), cpu_usage, alpha=0.7, color='orange')
        axes[1, 1].set_title('CPU Usage by Step')
        axes[1, 1].set_ylabel('CPU Usage (%)')
        axes[1, 1].set_xticks(range(len(step_names)))
        axes[1, 1].set_xticklabels(step_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_optimization_recommendations(self, output_path: Path):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze memory usage patterns
        if self.snapshots:
            max_memory = max(s.rss_mb for s in self.snapshots)
            avg_memory = sum(s.rss_mb for s in self.snapshots) / len(self.snapshots)
            
            if max_memory > self.target_memory_limit_mb:
                recommendations.append({
                    'type': 'CRITICAL',
                    'issue': 'Memory limit exceeded',
                    'details': f'Peak memory usage: {max_memory:.1f} MB > {self.target_memory_limit_mb:.1f} MB',
                    'suggestions': [
                        'Implement batch processing for large datasets',
                        'Use memory-mapped files for large data',
                        'Reduce model complexity or feature dimensions',
                        'Implement data streaming instead of loading all at once'
                    ]
                })
            elif max_memory > self.target_memory_limit_mb * 0.9:
                recommendations.append({
                    'type': 'WARNING',
                    'issue': 'Memory usage near limit',
                    'details': f'Peak memory usage: {max_memory:.1f} MB (90% of limit)',
                    'suggestions': [
                        'Monitor memory usage more closely',
                        'Consider implementing memory cleanup between steps',
                        'Optimize data structures for memory efficiency'
                    ]
                })
        
        # Analyze processing times
        if self.metrics:
            slow_steps = [m for m in self.metrics if m.duration_seconds > 10]
            if slow_steps:
                recommendations.append({
                    'type': 'PERFORMANCE',
                    'issue': 'Slow processing steps detected',
                    'details': f'Steps taking >10s: {[s.step_name for s in slow_steps]}',
                    'suggestions': [
                        'Implement parallel processing for CPU-intensive steps',
                        'Use vectorized operations instead of loops',
                        'Consider caching intermediate results',
                        'Optimize algorithms for better time complexity'
                    ]
                })
            
            # Memory leak detection
            memory_increasing_steps = [m for m in self.metrics if m.memory_delta_mb > 100]
            if memory_increasing_steps:
                recommendations.append({
                    'type': 'WARNING',
                    'issue': 'Potential memory leaks detected',
                    'details': f'Steps with large memory increases: {[s.step_name for s in memory_increasing_steps]}',
                    'suggestions': [
                        'Add explicit garbage collection after memory-intensive steps',
                        'Use context managers for resource management',
                        'Check for circular references in data structures',
                        'Implement memory profiling for specific functions'
                    ]
                })
        
        # Hardware-specific recommendations
        recommendations.append({
            'type': 'OPTIMIZATION',
            'issue': 'Hardware-specific optimizations',
            'details': f'System: {self.cpu_count} cores, {self.system_memory_gb:.1f} GB RAM',
            'suggestions': [
                f'Use n_jobs={min(self.cpu_count, 8)} for parallel processing',
                'Enable CPU-specific optimizations (AVX, SSE)',
                'Consider using numba for numerical computations',
                'Implement memory pooling for frequent allocations'
            ]
        })
        
        # Save recommendations
        with open(output_path / 'optimization_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Create readable report
        with open(output_path / 'optimization_report.md', 'w') as f:
            f.write("# Performance Optimization Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## System Information\n")
            f.write(f"- CPU Cores: {self.cpu_count}\n")
            f.write(f"- System Memory: {self.system_memory_gb:.1f} GB\n")
            f.write(f"- Target Memory Limit: {self.target_memory_limit_mb/1024:.1f} GB\n\n")
            
            if self.snapshots:
                max_memory = max(s.rss_mb for s in self.snapshots)
                f.write("## Memory Usage Summary\n")
                f.write(f"- Peak Memory Usage: {max_memory:.1f} MB\n")
                f.write(f"- Memory Efficiency: {(max_memory/self.target_memory_limit_mb)*100:.1f}% of target\n\n")
            
            f.write("## Recommendations\n\n")
            for rec in recommendations:
                f.write(f"### {rec['type']}: {rec['issue']}\n")
                f.write(f"**Details:** {rec['details']}\n\n")
                f.write("**Suggestions:**\n")
                for suggestion in rec['suggestions']:
                    f.write(f"- {suggestion}\n")
                f.write("\n")
    
    def _save_raw_data(self, output_path: Path):
        """Save raw profiling data"""
        # Save snapshots
        snapshots_data = []
        for s in self.snapshots:
            snapshots_data.append({
                'timestamp': s.timestamp,
                'rss_mb': s.rss_mb,
                'vms_mb': s.vms_mb,
                'percent': s.percent,
                'available_mb': s.available_mb,
                'step_name': s.step_name,
                'peak_tracemalloc_mb': s.peak_tracemalloc_mb
            })
        
        with open(output_path / 'memory_snapshots.json', 'w') as f:
            json.dump(snapshots_data, f, indent=2)
        
        # Save metrics
        metrics_data = []
        for m in self.metrics:
            metrics_data.append({
                'step_name': m.step_name,
                'duration_seconds': m.duration_seconds,
                'memory_before_mb': m.memory_before_mb,
                'memory_after_mb': m.memory_after_mb,
                'memory_peak_mb': m.memory_peak_mb,
                'memory_delta_mb': m.memory_delta_mb,
                'cpu_percent': m.cpu_percent,
                'data_size_mb': m.data_size_mb,
                'throughput_mb_per_sec': m.throughput_mb_per_sec
            })
        
        with open(output_path / 'performance_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)

class MemoryOptimizer:
    """Memory optimization utilities for the pipeline"""
    
    @staticmethod
    def optimize_numpy_arrays(arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize numpy arrays for memory efficiency"""
        optimized = []
        
        for arr in arrays:
            # Use appropriate dtype
            if arr.dtype == np.float64:
                # Check if we can use float32 without significant precision loss
                if np.allclose(arr, arr.astype(np.float32), rtol=1e-6):
                    arr = arr.astype(np.float32)
            
            elif arr.dtype == np.int64:
                # Use smallest integer type that can hold the data
                if arr.min() >= np.iinfo(np.int32).min and arr.max() <= np.iinfo(np.int32).max:
                    arr = arr.astype(np.int32)
                elif arr.min() >= np.iinfo(np.int16).min and arr.max() <= np.iinfo(np.int16).max:
                    arr = arr.astype(np.int16)
                elif arr.min() >= np.iinfo(np.int8).min and arr.max() <= np.iinfo(np.int8).max:
                    arr = arr.astype(np.int8)
            
            optimized.append(arr)
        
        return optimized
    
    @staticmethod
    def batch_process_data(data: np.ndarray, batch_size: int, 
                          process_func: Callable, **kwargs) -> np.ndarray:
        """Process data in batches to manage memory usage"""
        n_samples = data.shape[0]
        results = []
        
        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            batch_result = process_func(batch, **kwargs)
            results.append(batch_result)
            
            # Force garbage collection between batches
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        return np.concatenate(results, axis=0)
    
    @staticmethod
    def calculate_optimal_batch_size(data_shape: tuple, target_memory_mb: float = 500) -> int:
        """Calculate optimal batch size based on memory constraints"""
        # Estimate memory per sample (assuming float32)
        bytes_per_sample = np.prod(data_shape[1:]) * 4  # 4 bytes for float32
        target_memory_bytes = target_memory_mb * 1024 * 1024
        
        optimal_batch_size = max(1, int(target_memory_bytes / bytes_per_sample))
        
        # Round to nearest power of 2 for better performance
        optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
        
        return min(optimal_batch_size, data_shape[0])

def profile_pipeline_function(profiler: MemoryProfiler):
    """Decorator to automatically profile pipeline functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Estimate data size if possible
            data_size_mb = None
            for arg in args:
                if isinstance(arg, (np.ndarray, pd.DataFrame)):
                    if isinstance(arg, np.ndarray):
                        data_size_mb = arg.nbytes / (1024**2)
                    else:  # DataFrame
                        data_size_mb = arg.memory_usage(deep=True).sum() / (1024**2)
                    break
            
            with profiler.profile_step(func_name, data_size_mb):
                result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Performance Profiler")
    
    profiler = MemoryProfiler(target_memory_limit_gb=4.0)
    profiler.enable_tracemalloc()
    
    # Simulate some processing steps
    with profiler.profile_step("data_loading", data_size_mb=100):
        # Simulate loading data
        data = np.random.random((10000, 131)).astype(np.float32)
        time.sleep(0.5)
    
    with profiler.profile_step("preprocessing", data_size_mb=50):
        # Simulate preprocessing
        processed_data = data * 2 + 1
        time.sleep(0.3)
    
    with profiler.profile_step("model_training"):
        # Simulate training
        time.sleep(1.0)
        # Simulate memory growth
        temp_data = np.random.random((5000, 131))
    
    # Force cleanup
    del temp_data
    profiler.force_garbage_collection()
    
    # Generate report
    profiler.generate_report("test_performance_analysis")
    
    print("✅ Performance profiler test completed")