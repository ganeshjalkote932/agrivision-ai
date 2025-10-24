#!/usr/bin/env python3
"""
Main execution script for hyperspectral crop disease detection.

This script provides a command-line interface for running the complete pipeline
from data loading to model deployment with comprehensive logging and reporting.
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crop_disease_detection.pipeline import CropDiseaseDetectionPipeline


def create_default_config():
    """Create a default configuration file."""
    default_config = {
        "data": {
            "dataset_path": "GHISACONUS_2008_001_speclib.csv",
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True
        },
        "preprocessing": {
            "scaler_type": "StandardScaler",
            "handle_outliers": True,
            "outlier_method": "isolation_forest"
        },
        "models": {
            "algorithms": ["random_forest", "mlp", "svm"],
            "hyperparameter_optimization": True,
            "cross_validation_folds": 5
        },
        "feature_analysis": {
            "top_n_features": 15,
            "biological_validation": True
        },
        "output": {
            "save_models": True,
            "create_deployment_package": True,
            "generate_report": True
        }
    }
    
    config_path = "pipeline_config.json"
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration created: {config_path}")
    return config_path


def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Hyperspectral Crop Disease Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py

  # Run with custom dataset
  python main.py --dataset my_data.csv

  # Run with custom configuration
  python main.py --config my_config.json

  # Run with specific output directory
  python main.py --output results_2024

  # Create default configuration file
  python main.py --create-config

  # Run in debug mode
  python main.py --log-level DEBUG

  # Quick run (no hyperparameter optimization)
  python main.py --quick
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Path to hyperspectral dataset CSV file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--session-id', '-s',
        type=str,
        help='Custom session ID (default: timestamp)'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file and exit'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick run without hyperparameter optimization'
    )
    
    parser.add_argument(
        '--no-models',
        action='store_true',
        help='Skip model saving (faster execution)'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    args = parser.parse_args()
    
    # Handle create-config option
    if args.create_config:
        create_default_config()
        return True
    
    print("=" * 70)
    print("HYPERSPECTRAL CROP DISEASE DETECTION PIPELINE")
    print("=" * 70)
    
    try:
        # Load or modify configuration
        config_file = args.config
        if args.quick or args.no_models or args.no_report:
            # Create temporary config with modifications
            if config_file and os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                # Use default config
                config = {
                    "data": {
                        "dataset_path": args.dataset or "GHISACONUS_2008_001_speclib.csv",
                        "test_size": 0.2,
                        "random_state": 42,
                        "stratify": True
                    },
                    "preprocessing": {
                        "scaler_type": "StandardScaler",
                        "handle_outliers": True,
                        "outlier_method": "isolation_forest"
                    },
                    "models": {
                        "algorithms": ["random_forest", "mlp"],
                        "hyperparameter_optimization": not args.quick,
                        "cross_validation_folds": 3 if args.quick else 5
                    },
                    "feature_analysis": {
                        "top_n_features": 10 if args.quick else 15,
                        "biological_validation": True
                    },
                    "output": {
                        "save_models": not args.no_models,
                        "create_deployment_package": not args.no_models,
                        "generate_report": not args.no_report
                    }
                }
            
            # Apply command-line modifications
            if args.quick:
                config['models']['hyperparameter_optimization'] = False
                config['models']['cross_validation_folds'] = 3
                config['feature_analysis']['top_n_features'] = 10
            
            if args.no_models:
                config['output']['save_models'] = False
                config['output']['create_deployment_package'] = False
            
            if args.no_report:
                config['output']['generate_report'] = False
            
            # Save temporary config
            temp_config_path = "temp_config.json"
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            config_file = temp_config_path
        
        # Initialize pipeline
        print(f"Initializing pipeline...")
        print(f"  Output directory: {args.output}")
        print(f"  Log level: {args.log_level}")
        print(f"  Configuration: {config_file or 'default'}")
        
        pipeline = CropDiseaseDetectionPipeline(
            config_file=config_file,
            output_dir=args.output,
            session_id=args.session_id,
            enable_logging=True,
            enable_visualization=True,
            enable_reporting=not args.no_report
        )
        
        # Run complete pipeline
        print(f"\nStarting pipeline execution...")
        print(f"Session ID: {pipeline.session_id}")
        
        results = pipeline.run_complete_pipeline(dataset_path=args.dataset)
        
        # Get summary
        summary = pipeline.get_pipeline_summary()
        
        # Display results summary
        print("\n" + "=" * 70)
        if summary.get('execution_status') == 'completed':
            print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        else:
            print("❌ PIPELINE EXECUTION FAILED!")
        print("=" * 70)
        
        print(f"\nSession ID: {summary['session_id']}")
        print(f"Results Directory: {pipeline.session_dir}")
        
        if summary.get('execution_status') == 'completed':
            print(f"\nResults Summary:")
            dataset_info = summary.get('dataset_info', {})
            print(f"  Dataset Samples: {dataset_info.get('n_samples', 'N/A')}")
            print(f"  Dataset Features: {dataset_info.get('n_features', 'N/A')}")
            print(f"  Best Model: {summary.get('best_model', 'N/A')}")
            
            performance = summary.get('model_performance', {})
            best_accuracy = performance.get('accuracy', 0)
            print(f"  Best Accuracy: {best_accuracy:.4f}")
            
            target_accuracy = 0.95
            if best_accuracy >= target_accuracy:
                print(f"  ✅ Target accuracy ({target_accuracy:.1%}) achieved!")
            else:
                print(f"  ⚠️  Target accuracy ({target_accuracy:.1%}) not yet achieved.")
            
            relevance_score = summary.get('biological_relevance_score', 0)
            print(f"  Biological Relevance: {relevance_score:.1%}")
            
            output_files = summary.get('output_files', {})
            if output_files.get('model'):
                print(f"  ✅ Model exported and ready for deployment")
            
            if output_files.get('report'):
                print(f"  📊 Report: {Path(output_files['report']).name}")
        else:
            print(f"\nExecution Status: {summary.get('execution_status', 'Unknown')}")
        
        print(f"\nFor detailed results, check: {pipeline.session_dir}")
        
        # Clean up temporary config
        if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        return results['success']
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline execution interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ PIPELINE EXECUTION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)