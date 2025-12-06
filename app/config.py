"""
Configuration management for the Crop Disease Detection System
"""
import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database configuration
    DB_HOST = os.environ.get('DB_HOST') or 'localhost'
    DB_PORT = int(os.environ.get('DB_PORT') or 3306)
    DB_USER = os.environ.get('DB_USER') or 'root'
    DB_PASSWORD = os.environ.get('DB_PASSWORD') or 'Hello! World'
    DB_NAME = os.environ.get('DB_NAME') or 'crop'
    
    # Database connection pool settings
    DB_POOL_SIZE = 5
    DB_POOL_MAX_OVERFLOW = 10
    DB_POOL_TIMEOUT = 30
    DB_POOL_RECYCLE = 3600
    
    # Model configuration
    BASE_DIR = Path(__file__).parent.parent
    MODEL_PATH = BASE_DIR / 'best_model (2).pth'
    MODEL_DEVICE = 'cuda'  # Will auto-detect GPU availability
    MODEL_BATCH_SIZE = 1
    
    # Session configuration
    SESSION_TIMEOUT = 1800  # 30 minutes for farmers
    ADMIN_SESSION_TIMEOUT = 3600  # 60 minutes for admins
    
    # File upload configuration
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'npy'}  # Only .npy hyperspectral files
    
    # Admin configuration
    ADMIN_SPECIAL_CODE = os.environ.get('ADMIN_SPECIAL_CODE') or 'ADMIN2024SECURE'
    
    # Security configuration
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None  # No time limit for CSRF tokens
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 1800  # 30 minutes
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
        (Config.BASE_DIR / 'models').mkdir(exist_ok=True)
        (Config.BASE_DIR / 'static').mkdir(exist_ok=True)
        (Config.BASE_DIR / 'static' / 'css').mkdir(exist_ok=True)
        (Config.BASE_DIR / 'static' / 'js').mkdir(exist_ok=True)
        
        # Set Flask config if app is provided
        if app:
            app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_SIZE
