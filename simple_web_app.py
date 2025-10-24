#!/usr/bin/env python3
"""
AgriVision AI - Hyperspectral Crop Disease Detection
Advanced AI-powered web application for crop disease analysis
"""

import os
import io
import base64
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crop_disease_detection_2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class SimpleFileProcessor:
    """Simple file processor for hyperspectral data."""
    
    def __init__(self):
        self.supported_extensions = {'.npy', '.npz', '.tiff', '.tif'}
    
    def load_file(self, filepath):
        """Load hyperspectral data."""
        try:
            filepath = Path(filepath)
            extension = filepath.suffix.lower()
            
            if extension == '.npy':
                data = np.load(filepath)
            elif extension == '.npz':
                npz_data = np.load(filepath)
                # Use first array as data
                data = npz_data[npz_data.files[0]]
            elif extension in {'.tiff', '.tif'}:
                try:
                    import tifffile
                    data = tifffile.imread(filepath)
                except ImportError:
                    with Image.open(filepath) as img:
                        data = np.array(img)
            else:
                raise ValueError(f"Unsupported format: {extension}")
            
            # Normalize data
            if data.max() > 1.0:
                if data.max() <= 100.0:
                    data = data / 100.0
                else:
                    data = (data - data.min()) / (data.max() - data.min())
            
            metadata = {
                'file_type': extension[1:],
                'original_shape': data.shape,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                'data_range': [float(data.min()), float(data.max())]
            }
            
            return data, metadata
            
        except Exception as e:
            return None, {'error': str(e)}
    
    def preprocess_for_prediction(self, data):
        """Preprocess data for prediction."""
        if data.ndim == 1:
            return data.reshape(1, -1)
        elif data.ndim == 2:
            return data
        elif data.ndim == 3:
            # Flatten spatial dimensions
            return data.reshape(-1, data.shape[-1])
        else:
            raise ValueError(f"Cannot handle {data.ndim}D data")

class SimpleDiseaseDetector:
    """Simple disease detector with mock predictions."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        
    def predict_batch(self, data):
        """Make predictions on batch data."""
        results = []
        
        # Limit to first 20 samples for display, but analyze all for statistics
        display_limit = min(20, len(data))
        
        for i, sample in enumerate(data):
            # Mock prediction based on data statistics
            mean_val = np.mean(sample)
            std_val = np.std(sample)
            
            # Simple heuristic: if mean is low and std is high, likely diseased
            disease_prob = 1.0 / (1.0 + np.exp(-(std_val - mean_val) * 10))
            prediction = 1 if disease_prob > 0.5 else 0
            confidence = abs(disease_prob - 0.5) * 2
            
            result = {
                'sample_index': i,
                'prediction': prediction,
                'prediction_label': 'Diseased' if prediction == 1 else 'Healthy',
                'disease_probability': disease_prob,
                'confidence': confidence,
                'show_in_table': i < display_limit  # Only show first 100 in detailed table
            }
            
            results.append(result)
        
        return results

# Global instances
file_processor = SimpleFileProcessor()
disease_detector = SimpleDiseaseDetector()

def allowed_file(filename):
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in {'.npy', '.npz', '.tiff', '.tif'}

def create_visualization(data, predictions):
    """Create simple visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spectral plot
    if data.ndim >= 2:
        for i, spectrum in enumerate(data[:5]):  # Plot first 5 spectra
            label = f"Sample {i+1}"
            if i < len(predictions):
                pred_label = "Diseased" if predictions[i]['prediction'] == 1 else "Healthy"
                label += f" ({pred_label})"
            ax1.plot(spectrum, label=label, alpha=0.8)
    else:
        ax1.plot(data, label='Spectral Signature')
    
    ax1.set_xlabel('Spectral Band')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Hyperspectral Signatures')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prediction results
    diseased_count = sum(1 for p in predictions if p['prediction'] == 1)
    healthy_count = len(predictions) - diseased_count
    
    labels = ['Healthy', 'Diseased']
    counts = [healthy_count, diseased_count]
    colors = ['green', 'red']
    
    ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Disease Detection Results')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"

@app.route('/')
def index():
    """Main page."""
    return render_template('simple_index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and processing."""
    if request.method == 'GET':
        return render_template('simple_upload.html')
    
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
            data, metadata = file_processor.load_file(filepath)
            
            if data is None:
                flash(f'Error processing file: {metadata.get("error", "Unknown error")}')
                return redirect(request.url)
            
            # Preprocess for prediction
            processed_data = file_processor.preprocess_for_prediction(data)
            
            # Make predictions
            predictions = disease_detector.predict_batch(processed_data)
            
            # Create visualization
            visualization = create_visualization(processed_data, predictions)
            
            # Create summary
            diseased_count = sum(1 for p in predictions if p['prediction'] == 1)
            summary = {
                'total_samples': len(predictions),
                'healthy_samples': len(predictions) - diseased_count,
                'diseased_samples': diseased_count,
                'disease_percentage': (diseased_count / len(predictions)) * 100,
                'average_confidence': np.mean([p['confidence'] for p in predictions]),
                'samples_shown': min(20, len(predictions)),
                'samples_hidden': max(0, len(predictions) - 20)
            }
            
            # Filter predictions for display (only first 20)
            display_predictions = [p for p in predictions if p.get('show_in_table', True)]
            
            return render_template('simple_results.html',
                                 filename=filename,
                                 metadata=metadata,
                                 predictions=display_predictions,
                                 summary=summary,
                                 visualization=visualization)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)
    else:
        flash('Invalid file type. Please upload .npy, .npz, .tiff, or .tif files.')
        return redirect(request.url)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            
            # Process file
            data, metadata = file_processor.load_file(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            if data is None:
                return jsonify({'error': metadata.get('error', 'Processing failed')}), 500
            
            # Preprocess and predict
            processed_data = file_processor.preprocess_for_prediction(data)
            predictions = disease_detector.predict_batch(processed_data)
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'metadata': metadata
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌱 Starting AgriVision AI - Hyperspectral Analysis Platform...")
    print("📱 Access AgriVision AI at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)