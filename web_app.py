#!/usr/bin/env python3
"""
Crop Disease Detection Web Application
Flask-based web interface for hyperspectral crop disease detection
"""

import os
import io
import base64
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import traceback
from datetime import datetime
import tempfile
import zipfile

# Import our detection modules
from crop_disease_detection.disease_detector import DiseaseDetector
from crop_disease_detection.preprocessor import Preprocessor
from crop_disease_detection.config import PreprocessingConfig
from file_processor import HyperspectralFileProcessor
from web_utils import WebVisualizationGenerator, ResultsExporter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crop_disease_detection_2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'web_results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables for model and processor
disease_detector = None
file_processor = None
viz_generator = None
results_exporter = None

def initialize_components():
    """Initialize the disease detection components"""
    global disease_detector, file_processor, viz_generator, results_exporter
    
    try:
        # Initialize file processor
        file_processor = HyperspectralFileProcessor()
        
        # Initialize visualization generator
        viz_generator = WebVisualizationGenerator()
        
        # Initialize results exporter
        results_exporter = ResultsExporter()
        
        # Try to load the trained model
        model_dir = Path("results") / "session_20251023_151852" / "models"
        if model_dir.exists():
            # Find the best model directory
            model_dirs = [d for d in model_dir.iterdir() if d.is_dir() and "best_model" in d.name]
            if model_dirs:
                best_model_dir = model_dirs[0]
                model_path = best_model_dir / f"{best_model_dir.name}_model.joblib"
                preprocessor_path = best_model_dir / f"{best_model_dir.name}_preprocessor.joblib"
                metadata_path = best_model_dir / f"{best_model_dir.name}_metadata.json"
                
                if all(p.exists() for p in [model_path, preprocessor_path, metadata_path]):
                    disease_detector = DiseaseDetector(
                        model_path=str(model_path),
                        preprocessor_path=str(preprocessor_path),
                        metadata_path=str(metadata_path)
                    )
                    print(f"✅ Model loaded from {best_model_dir}")
                    return True
        
        print("⚠️ No trained model found. Please train a model first.")
        return False
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    model_available = disease_detector is not None
    return render_template('index.html', model_available=model_available)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and processing"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Process the file
            result = process_uploaded_file(filepath, unique_filename)
            
            if result['success']:
                return render_template('results.html', 
                                     result=result, 
                                     filename=filename)
            else:
                flash(f'Error processing file: {result["error"]}')
                return redirect(request.url)
                
        except Exception as e:
            flash(f'Error uploading file: {str(e)}')
            return redirect(request.url)
    else:
        flash('Invalid file type. Please upload .npy or .tiff files.')
        return redirect(request.url)

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'.npy', '.tiff', '.tif', '.npz'}
    return Path(filename).suffix.lower() in allowed_extensions

def process_uploaded_file(filepath, unique_filename):
    """Process uploaded hyperspectral file"""
    try:
        if disease_detector is None:
            return {
                'success': False,
                'error': 'Disease detection model not available. Please train a model first.'
            }
        
        # Load and process the file
        spectral_data, metadata = file_processor.load_file(filepath)
        
        if spectral_data is None:
            return {
                'success': False,
                'error': 'Could not load spectral data from file'
            }
        
        # Preprocess the data
        processed_data = file_processor.preprocess_for_prediction(spectral_data)
        
        # Make predictions
        if processed_data.ndim == 1:
            # Single sample
            prediction_result = disease_detector.predict_single_sample(processed_data)
            predictions = [prediction_result]
        else:
            # Multiple samples
            predictions = []
            for i, sample in enumerate(processed_data):
                pred = disease_detector.predict_single_sample(sample)
                pred['sample_index'] = i
                predictions.append(pred)
        
        # Generate visualizations
        visualizations = viz_generator.create_analysis_visualizations(
            spectral_data, processed_data, predictions, metadata
        )
        
        # Create results summary
        results_summary = create_results_summary(predictions, metadata)
        
        # Export results
        results_dir = Path(app.config['RESULTS_FOLDER']) / unique_filename.split('.')[0]
        results_dir.mkdir(exist_ok=True)
        
        export_paths = results_exporter.export_results(
            predictions=predictions,
            visualizations=visualizations,
            metadata=metadata,
            output_dir=str(results_dir)
        )
        
        return {
            'success': True,
            'predictions': predictions,
            'summary': results_summary,
            'visualizations': visualizations,
            'metadata': metadata,
            'export_paths': export_paths,
            'results_dir': str(results_dir)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Processing error: {str(e)}',
            'traceback': traceback.format_exc()
        }

def create_results_summary(predictions, metadata):
    """Create summary of prediction results"""
    if not predictions:
        return {}
    
    # Calculate statistics
    diseased_count = sum(1 for p in predictions if p['prediction'] == 1)
    healthy_count = len(predictions) - diseased_count
    
    # Calculate confidence statistics
    confidences = [p['confidence'] for p in predictions]
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    
    # Disease probability statistics
    disease_probs = [p['disease_probability'] for p in predictions]
    avg_disease_prob = np.mean(disease_probs)
    
    summary = {
        'total_samples': len(predictions),
        'diseased_samples': diseased_count,
        'healthy_samples': healthy_count,
        'disease_percentage': (diseased_count / len(predictions)) * 100,
        'average_confidence': avg_confidence,
        'confidence_range': [min_confidence, max_confidence],
        'average_disease_probability': avg_disease_prob,
        'high_confidence_predictions': sum(1 for c in confidences if c > 0.8),
        'low_confidence_predictions': sum(1 for c in confidences if c < 0.6)
    }
    
    return summary

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            
            # Process file
            result = process_uploaded_file(tmp_file.name, f"api_{file.filename}")
            
            # Clean up
            os.unlink(tmp_file.name)
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'predictions': result['predictions'],
                    'summary': result['summary']
                })
            else:
                return jsonify({'error': result['error']}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_results/<path:results_dir>')
def download_results(results_dir):
    """Download results as ZIP file"""
    try:
        results_path = Path(app.config['RESULTS_FOLDER']) / results_dir
        
        if not results_path.exists():
            flash('Results not found')
            return redirect(url_for('index'))
        
        # Create ZIP file
        zip_path = results_path.parent / f"{results_dir}_results.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in results_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(results_path)
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name=f"{results_dir}_results.zip")
        
    except Exception as e:
        flash(f'Error creating download: {str(e)}')
        return redirect(url_for('index'))

@app.route('/model_info')
def model_info():
    """Display model information"""
    if disease_detector is None:
        return render_template('model_info.html', model_available=False)
    
    try:
        model_metadata = disease_detector.get_model_metadata()
        return render_template('model_info.html', 
                             model_available=True, 
                             metadata=model_metadata)
    except Exception as e:
        return render_template('model_info.html', 
                             model_available=False, 
                             error=str(e))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 100MB.')
    return redirect(url_for('upload_file'))

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    print("🌱 Initializing Crop Disease Detection Web Application...")
    
    # Initialize components
    if initialize_components():
        print("✅ Components initialized successfully")
    else:
        print("⚠️ Some components failed to initialize")
    
    print("🚀 Starting web server...")
    print("📱 Access the application at: http://localhost:5000")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)