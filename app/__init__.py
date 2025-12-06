"""
Crop Disease Detection System - Flask Application Factory
"""
from flask import Flask, session
from app.config import Config


def create_app(config_class=Config):
    """Create and configure the Flask application"""
    app = Flask(__name__, 
                static_folder='../static',
                template_folder='templates')
    app.config.from_object(config_class)
    
    # Initialize configuration
    config_class.init_app(app)
    
    # Register error handlers
    from app.error_handlers import register_error_handlers
    register_error_handlers(app)
    
    # Add security headers to all responses
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses"""
        # Prevent XSS attacks
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self';"
        )
        
        # Prevent MIME type sniffing
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response
    
    # Generate CSRF token for each session
    @app.before_request
    def generate_csrf_token():
        """Generate CSRF token for session if not present"""
        from app.security import SecurityUtils
        
        if 'session_id' in session and 'csrf_token' not in session:
            session['csrf_token'] = SecurityUtils.generate_csrf_token(session['session_id'])
    
    # Make CSRF token available in templates
    @app.context_processor
    def inject_csrf_token():
        """Inject CSRF token into all templates"""
        return dict(csrf_token=session.get('csrf_token', ''))
    
    # Register blueprints
    from app.auth import auth_bp
    from app.farmer import farmer_bp
    from app.admin import admin_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(farmer_bp)
    app.register_blueprint(admin_bp)
    
    # Initialize and start background processing worker
    # Using app context instead of deprecated before_first_request
    with app.app_context():
        try:
            from app.model_service import ModelService
            from app.file_service import FileService
            from app.data_structures.queue import ProcessingQueue
            from app.processing_worker import initialize_worker, start_worker
            
            # Initialize services
            model_service = ModelService()
            file_service = FileService()
            processing_queue = ProcessingQueue()
            
            # Initialize and start worker
            worker = initialize_worker(model_service, file_service, processing_queue)
            if start_worker():
                print("✓ Background processing worker started successfully")
            else:
                print("✗ Failed to start background processing worker")
        except Exception as e:
            print(f"✗ Error starting background worker: {e}")
            import traceback
            traceback.print_exc()
    
    return app
