"""
Error handling utilities and custom error pages.

This module provides:
- Custom error handlers for common HTTP errors
- Error logging with timestamps
- User-friendly error pages
- Error recovery mechanisms
"""
import logging
import traceback
from datetime import datetime
from flask import render_template, jsonify, request
from typing import Tuple, Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ErrorLogger:
    """Utility class for logging errors with context"""
    
    @staticmethod
    def log_error(error: Exception, context: dict = None) -> None:
        """
        Log an error with context information.
        
        Args:
            error: The exception that occurred
            context: Additional context information (user, request, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        error_info = {
            'timestamp': timestamp,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        if context:
            error_info.update(context)
        
        # Log the error
        logger.error(f"Error occurred: {error_info}")
    
    @staticmethod
    def log_warning(message: str, context: dict = None) -> None:
        """
        Log a warning with context information.
        
        Args:
            message: Warning message
            context: Additional context information
        """
        timestamp = datetime.now().isoformat()
        
        warning_info = {
            'timestamp': timestamp,
            'message': message
        }
        
        if context:
            warning_info.update(context)
        
        logger.warning(f"Warning: {warning_info}")
    
    @staticmethod
    def log_info(message: str, context: dict = None) -> None:
        """
        Log an info message with context.
        
        Args:
            message: Info message
            context: Additional context information
        """
        timestamp = datetime.now().isoformat()
        
        info = {
            'timestamp': timestamp,
            'message': message
        }
        
        if context:
            info.update(context)
        
        logger.info(f"Info: {info}")


def register_error_handlers(app):
    """
    Register error handlers for the Flask application.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(400)
    def bad_request(error) -> Tuple[Any, int]:
        """Handle 400 Bad Request errors"""
        ErrorLogger.log_warning(
            "Bad request",
            {
                'url': request.url,
                'method': request.method,
                'error': str(error)
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Bad request',
                'message': 'The request could not be understood or was missing required parameters.'
            }), 400
        
        return render_template('errors/400.html', error=error), 400
    
    @app.errorhandler(401)
    def unauthorized(error) -> Tuple[Any, int]:
        """Handle 401 Unauthorized errors"""
        ErrorLogger.log_warning(
            "Unauthorized access attempt",
            {
                'url': request.url,
                'method': request.method,
                'ip': request.remote_addr
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Authentication is required to access this resource.'
            }), 401
        
        return render_template('errors/401.html', error=error), 401
    
    @app.errorhandler(403)
    def forbidden(error) -> Tuple[Any, int]:
        """Handle 403 Forbidden errors"""
        ErrorLogger.log_warning(
            "Forbidden access attempt",
            {
                'url': request.url,
                'method': request.method,
                'ip': request.remote_addr
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Forbidden',
                'message': 'You do not have permission to access this resource.'
            }), 403
        
        return render_template('errors/403.html', error=error), 403
    
    @app.errorhandler(404)
    def not_found(error) -> Tuple[Any, int]:
        """Handle 404 Not Found errors"""
        ErrorLogger.log_info(
            "Resource not found",
            {
                'url': request.url,
                'method': request.method
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Not found',
                'message': 'The requested resource was not found.'
            }), 404
        
        return render_template('errors/404.html', error=error), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error) -> Tuple[Any, int]:
        """Handle 405 Method Not Allowed errors"""
        ErrorLogger.log_warning(
            "Method not allowed",
            {
                'url': request.url,
                'method': request.method
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Method not allowed',
                'message': 'The HTTP method is not allowed for this resource.'
            }), 405
        
        return render_template('errors/405.html', error=error), 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error) -> Tuple[Any, int]:
        """Handle 413 Request Entity Too Large errors"""
        ErrorLogger.log_warning(
            "Request entity too large",
            {
                'url': request.url,
                'method': request.method
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'File too large',
                'message': 'The uploaded file exceeds the maximum allowed size.'
            }), 413
        
        return render_template('errors/413.html', error=error), 413
    
    @app.errorhandler(500)
    def internal_server_error(error) -> Tuple[Any, int]:
        """Handle 500 Internal Server Error"""
        ErrorLogger.log_error(
            error,
            {
                'url': request.url,
                'method': request.method,
                'ip': request.remote_addr
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'message': 'An unexpected error occurred. Please try again later.'
            }), 500
        
        return render_template('errors/500.html', error=error), 500
    
    @app.errorhandler(503)
    def service_unavailable(error) -> Tuple[Any, int]:
        """Handle 503 Service Unavailable errors"""
        ErrorLogger.log_error(
            error,
            {
                'url': request.url,
                'method': request.method
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Service unavailable',
                'message': 'The service is temporarily unavailable. Please try again later.'
            }), 503
        
        return render_template('errors/503.html', error=error), 503
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error) -> Tuple[Any, int]:
        """Handle any unexpected errors"""
        ErrorLogger.log_error(
            error,
            {
                'url': request.url,
                'method': request.method,
                'ip': request.remote_addr
            }
        )
        
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Unexpected error',
                'message': 'An unexpected error occurred. Please try again later.'
            }), 500
        
        return render_template('errors/500.html', error=error), 500


class DatabaseErrorHandler:
    """Handler for database-related errors with retry logic"""
    
    @staticmethod
    def execute_with_retry(func, max_retries: int = 3, *args, **kwargs):
        """
        Execute a database function with retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func execution
            
        Raises:
            Exception: If all retry attempts fail
        """
        import time
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                ErrorLogger.log_warning(
                    f"Database operation failed (attempt {attempt + 1}/{max_retries})",
                    {'error': str(e)}
                )
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    ErrorLogger.log_error(
                        e,
                        {'message': 'All retry attempts failed'}
                    )
        
        # All retries failed
        raise last_error


def safe_execute(func, default_return=None, log_errors=True):
    """
    Safely execute a function and return a default value on error.
    
    Args:
        func: Function to execute
        default_return: Value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Result of func or default_return on error
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            ErrorLogger.log_error(e)
        return default_return
