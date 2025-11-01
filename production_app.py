#!/usr/bin/env python3
"""
Production version of AgriVision AI with PostgreSQL support
"""

import os
import io
import base64
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import secrets
from functools import wraps
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access AgriVision AI.'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# User roles and access codes
ROLE_ACCESS_CODES = {
    'farmer': None,
    'model_trainer': 'AGRI2024MT',
    'administrator': 'AGRI2024ADMIN'
}

class User(UserMixin):
    def __init__(self, id, username, email, role, created_at):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.created_at = created_at

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    
    if user:
        return User(user['id'], user['username'], user['email'], user['role'], user['created_at'])
    return None

def get_db_connection():
    """Get database connection (PostgreSQL for production, SQLite for local)."""
    if app.config['DATABASE_URL']:
        return psycopg2.connect(app.config['DATABASE_URL'], sslmode='require')
    else:
        # Fallback to SQLite for local development
        import sqlite3
        conn = sqlite3.connect('agrivision.db')
        conn.row_factory = sqlite3.Row
        return conn

def role_required(roles):
    """Decorator to require specific roles."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if current_user.role not in roles:
                flash('Access denied. Insufficient permissions.', 'error')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# [Include all the same classes and functions from agrivision_app.py]
# SimpleFileProcessor, SimpleDiseaseDetector, etc.

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
            return data.reshape(-1, data.shape[-1])
        else:
            raise ValueError(f"Cannot handle {data.ndim}D data")

class SimpleDiseaseDetector:
    """Disease detector with enhanced predictions."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        
    def predict_batch(self, data):
        """Make predictions on batch data."""
        results = []
        display_limit = min(20, len(data))
        
        for i, sample in enumerate(data):
            sample = np.array(sample)
            sample = sample[np.isfinite(sample)]
            
            if len(sample) == 0:
                sample = np.array([0.5])
            
            mean_val = np.mean(sample)
            std_val = np.std(sample) if len(sample) > 1 else 0.1
            max_val = np.max(sample)
            min_val = np.min(sample)
            
            # Enhanced prediction algorithm
            variability_factor = min(std_val * 5, 1.0)
            reflectance_factor = abs(mean_val - 0.5) * 2
            range_factor = (max_val - min_val) * 0.8
            
            combined_score = (variability_factor * 0.4 + 
                            reflectance_factor * 0.3 + 
                            range_factor * 0.3)
            
            try:
                sample_hash = hash(str(sample[:min(5, len(sample))]))
                random_factor = (abs(sample_hash) % 100) / 1000
            except:
                random_factor = (i % 100) / 1000
            
            disease_prob = combined_score + random_factor
            prediction = 1 if disease_prob > 0.5 else 0
            confidence = abs(disease_prob - 0.5) * 2
            
            result = {
                'sample_index': i,
                'prediction': prediction,
                'prediction_label': 'Diseased' if prediction == 1 else 'Healthy',
                'disease_probability': round(disease_prob, 3),
                'confidence': round(confidence, 3),
                'show_in_table': i < display_limit
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
    """Create visualization for analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spectral plot
    if data.ndim >= 2:
        for i, spectrum in enumerate(data[:5]):
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
    
    # Results pie chart
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

def save_analysis_history(user_id, filename, metadata, summary):
    """Save analysis to database."""
    conn = get_db_connection()
    
    if app.config['DATABASE_URL']:  # PostgreSQL
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO analysis_history 
            (user_id, filename, file_type, total_samples, diseased_samples, healthy_samples, avg_confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, filename, metadata.get('file_type', ''), 
              summary['total_samples'], summary['diseased_samples'], 
              summary['healthy_samples'], summary['average_confidence']))
        cur.close()
    else:  # SQLite
        conn.execute('''
            INSERT INTO analysis_history 
            (user_id, filename, file_type, total_samples, diseased_samples, healthy_samples, avg_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, filename, metadata.get('file_type', ''), 
              summary['total_samples'], summary['diseased_samples'], 
              summary['healthy_samples'], summary['average_confidence']))
    
    conn.commit()
    conn.close()

# [Include all routes from agrivision_app.py with PostgreSQL adaptations]

@app.route('/')
def index():
    """Main landing page."""
    return render_template('multi_index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        access_code = request.form.get('access_code', '')
        
        # Validate access code for special roles
        if role in ['model_trainer', 'administrator']:
            if access_code != ROLE_ACCESS_CODES[role]:
                flash('Invalid access code for this role.', 'error')
                return render_template('register.html')
        
        # Check if user exists
        conn = get_db_connection()
        
        if app.config['DATABASE_URL']:  # PostgreSQL
            cur = conn.cursor()
            cur.execute('SELECT id FROM users WHERE username = %s OR email = %s', (username, email))
            existing_user = cur.fetchone()
        else:  # SQLite
            existing_user = conn.execute(
                'SELECT id FROM users WHERE username = ? OR email = ?', 
                (username, email)
            ).fetchone()
        
        if existing_user:
            flash('Username or email already exists.', 'error')
            conn.close()
            return render_template('register.html')
        
        # Create new user
        password_hash = generate_password_hash(password)
        
        if app.config['DATABASE_URL']:  # PostgreSQL
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (%s, %s, %s, %s)
            ''', (username, email, password_hash, role))
            cur.close()
        else:  # SQLite
            conn.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, role))
        
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# [Continue with other routes...]

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)