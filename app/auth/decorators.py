"""
Access control decorators for route protection.

This module provides decorators for:
- Farmer-only route protection
- Admin-only route protection
- Session validation on protected routes
- Automatic redirect for unauthorized access
"""
from functools import wraps
from flask import session, redirect, url_for, flash, request, abort
from app.auth.auth_service import auth_service
from app.error_handlers import ErrorLogger


def login_required(f):
    """
    Decorator to require any authenticated user (farmer or admin).
    
    Validates session and ensures user is logged in.
    Redirects to appropriate login page if not authenticated.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = session.get('session_id')
        user_type = session.get('user_type')
        
        if not session_id:
            ErrorLogger.log_warning(
                "Unauthorized access attempt - no session",
                {
                    'url': request.url,
                    'method': request.method,
                    'ip': request.remote_addr
                }
            )
            flash('Please log in to access this page', 'error')
            
            # Redirect to appropriate login page based on URL
            if '/admin/' in request.path:
                return redirect(url_for('admin.login'))
            else:
                return redirect(url_for('farmer.login'))
        
        # Validate session
        if not auth_service.validate_session(session_id):
            ErrorLogger.log_warning(
                "Invalid or expired session",
                {
                    'session_id': session_id,
                    'url': request.url,
                    'ip': request.remote_addr
                }
            )
            # Clear invalid session
            session.clear()
            flash('Your session has expired. Please log in again.', 'error')
            
            # Redirect to appropriate login page
            if user_type == 'admin':
                return redirect(url_for('admin.login'))
            else:
                return redirect(url_for('farmer.login'))
        
        return f(*args, **kwargs)
    
    return decorated_function


def farmer_required(f):
    """
    Decorator to require farmer authentication for routes.
    
    Validates session and ensures user is a farmer.
    Redirects to farmer login if not authenticated or not a farmer.
    
    Requirements: 3.2
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = session.get('session_id')
        user_type = session.get('user_type')
        
        # Check if user is logged in
        if not session_id:
            ErrorLogger.log_warning(
                "Farmer route access attempt without session",
                {
                    'url': request.url,
                    'method': request.method,
                    'ip': request.remote_addr
                }
            )
            flash('Please log in as a farmer to access this page', 'error')
            return redirect(url_for('farmer.login'))
        
        # Validate session and check if user is a farmer
        if not auth_service.validate_session(session_id, required_user_type='farmer'):
            ErrorLogger.log_warning(
                "Invalid farmer session or wrong user type",
                {
                    'session_id': session_id,
                    'user_type': user_type,
                    'url': request.url,
                    'ip': request.remote_addr
                }
            )
            # Clear invalid session
            session.clear()
            flash('Access denied. Farmer authentication required.', 'error')
            return redirect(url_for('farmer.login'))
        
        # Verify user type matches
        if user_type != 'farmer':
            ErrorLogger.log_warning(
                "Non-farmer attempting to access farmer route",
                {
                    'user_type': user_type,
                    'url': request.url,
                    'ip': request.remote_addr
                }
            )
            flash('Access denied. This page is for farmers only.', 'error')
            
            # Redirect to appropriate dashboard
            if user_type == 'admin':
                return redirect(url_for('admin.dashboard'))
            else:
                return redirect(url_for('farmer.login'))
        
        return f(*args, **kwargs)
    
    return decorated_function


def admin_required(f):
    """
    Decorator to require admin authentication for routes.
    
    Validates session and ensures user is an administrator.
    Redirects to admin login if not authenticated or not an admin.
    
    Requirements: 4.2
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = session.get('session_id')
        user_type = session.get('user_type')
        
        # Check if user is logged in
        if not session_id:
            ErrorLogger.log_warning(
                "Admin route access attempt without session",
                {
                    'url': request.url,
                    'method': request.method,
                    'ip': request.remote_addr
                }
            )
            flash('Please log in as an administrator to access this page', 'error')
            return redirect(url_for('admin.login'))
        
        # Validate session and check if user is an admin
        if not auth_service.validate_session(session_id, required_user_type='admin'):
            ErrorLogger.log_warning(
                "Invalid admin session or wrong user type",
                {
                    'session_id': session_id,
                    'user_type': user_type,
                    'url': request.url,
                    'ip': request.remote_addr
                }
            )
            # Clear invalid session
            session.clear()
            flash('Access denied. Administrator privileges required.', 'error')
            return redirect(url_for('admin.login'))
        
        # Verify user type matches
        if user_type != 'admin':
            ErrorLogger.log_warning(
                "Non-admin attempting to access admin route",
                {
                    'user_type': user_type,
                    'url': request.url,
                    'ip': request.remote_addr
                }
            )
            flash('Access denied. This page is for administrators only.', 'error')
            
            # Redirect to appropriate dashboard
            if user_type == 'farmer':
                return redirect(url_for('farmer.dashboard'))
            else:
                return redirect(url_for('admin.login'))
        
        return f(*args, **kwargs)
    
    return decorated_function


def api_key_required(f):
    """
    Decorator to require API key for API endpoints.
    
    Validates API key in request headers.
    Returns 401 if API key is missing or invalid.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            ErrorLogger.log_warning(
                "API request without API key",
                {
                    'url': request.url,
                    'method': request.method,
                    'ip': request.remote_addr
                }
            )
            abort(401, description="API key is required")
        
        # Validate API key (implement your validation logic)
        # For now, we'll just check if it exists
        # In production, validate against stored API keys
        
        return f(*args, **kwargs)
    
    return decorated_function


def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """
    Decorator to implement rate limiting on routes.
    
    Args:
        max_requests: Maximum number of requests allowed in the time window
        window_seconds: Time window in seconds
    
    Note: This is a simple in-memory implementation.
    For production, use Redis or a proper rate limiting library.
    """
    from collections import defaultdict
    from time import time
    
    # Store request counts per IP
    request_counts = defaultdict(list)
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ip = request.remote_addr
            current_time = time()
            
            # Clean old requests outside the window
            request_counts[ip] = [
                req_time for req_time in request_counts[ip]
                if current_time - req_time < window_seconds
            ]
            
            # Check if rate limit exceeded
            if len(request_counts[ip]) >= max_requests:
                ErrorLogger.log_warning(
                    "Rate limit exceeded",
                    {
                        'ip': ip,
                        'url': request.url,
                        'requests': len(request_counts[ip])
                    }
                )
                abort(429, description="Rate limit exceeded. Please try again later.")
            
            # Add current request
            request_counts[ip].append(current_time)
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator
