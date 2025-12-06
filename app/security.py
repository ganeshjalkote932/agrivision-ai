"""
Security utilities for input sanitization and validation.

This module provides functions for:
- XSS prevention through HTML escaping
- CSRF token generation and validation
- Filename sanitization to prevent directory traversal
- Input validation and sanitization
"""
import re
import os
import secrets
import hashlib
from typing import Optional, Tuple
from markupsafe import escape
from werkzeug.utils import secure_filename


class SecurityUtils:
    """Utility class for security-related operations"""
    
    # CSRF token storage (in production, use Redis or database)
    _csrf_tokens = {}
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """
        Sanitize HTML input to prevent XSS attacks.
        
        Args:
            text: Input text that may contain HTML
            
        Returns:
            Escaped text safe for HTML rendering
        """
        if not text:
            return ""
        
        # Use markupsafe's escape function to escape HTML special characters
        return str(escape(text))
    
    @staticmethod
    def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize general text input.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length (optional)
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Strip leading/trailing whitespace
        sanitized = text.strip()
        
        # Escape HTML to prevent XSS
        sanitized = SecurityUtils.sanitize_html(sanitized)
        
        # Truncate if max_length specified
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    @staticmethod
    def sanitize_filename(filename: str) -> Tuple[bool, str, str]:
        """
        Sanitize filename to prevent directory traversal attacks.
        
        This function:
        - Uses werkzeug's secure_filename for basic sanitization
        - Removes path separators and parent directory references
        - Validates file extension
        - Ensures filename is not empty after sanitization
        
        Args:
            filename: Original filename from user upload
            
        Returns:
            Tuple of (is_valid, sanitized_filename, error_message)
        """
        if not filename:
            return False, "", "Filename is empty"
        
        # Use werkzeug's secure_filename for basic sanitization
        safe_filename = secure_filename(filename)
        
        if not safe_filename:
            return False, "", "Filename contains only invalid characters"
        
        # Additional checks for directory traversal attempts
        dangerous_patterns = [
            '..',      # Parent directory reference
            '/',       # Unix path separator
            '\\',      # Windows path separator
            ':',       # Drive letter separator (Windows)
            '\0',      # Null byte
        ]
        
        for pattern in dangerous_patterns:
            if pattern in safe_filename:
                return False, "", f"Filename contains invalid pattern: {pattern}"
        
        # Ensure filename has an extension
        if '.' not in safe_filename:
            return False, "", "Filename must have an extension"
        
        # Validate filename length
        if len(safe_filename) > 255:
            return False, "", "Filename is too long (max 255 characters)"
        
        return True, safe_filename, ""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if email format is valid, False otherwise
        """
        if not email:
            return False
        
        # Basic email regex pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        return bool(re.match(email_pattern, email))
    
    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, str]:
        """
        Validate password strength.
        
        Requirements:
        - At least 8 characters long
        - Contains at least one uppercase letter
        - Contains at least one lowercase letter
        - Contains at least one digit
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not password:
            return False, "Password is required"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        return True, ""
    
    @staticmethod
    def generate_csrf_token(session_id: str) -> str:
        """
        Generate a CSRF token for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CSRF token string
        """
        # Generate a random token
        token = secrets.token_urlsafe(32)
        
        # Store token associated with session
        SecurityUtils._csrf_tokens[session_id] = token
        
        return token
    
    @staticmethod
    def validate_csrf_token(session_id: str, token: str) -> bool:
        """
        Validate a CSRF token for a session.
        
        Args:
            session_id: Session identifier
            token: CSRF token to validate
            
        Returns:
            True if token is valid, False otherwise
        """
        if not session_id or not token:
            return False
        
        # Get stored token for session
        stored_token = SecurityUtils._csrf_tokens.get(session_id)
        
        if not stored_token:
            return False
        
        # Compare tokens using constant-time comparison to prevent timing attacks
        return secrets.compare_digest(stored_token, token)
    
    @staticmethod
    def remove_csrf_token(session_id: str) -> None:
        """
        Remove CSRF token for a session (e.g., on logout).
        
        Args:
            session_id: Session identifier
        """
        if session_id in SecurityUtils._csrf_tokens:
            del SecurityUtils._csrf_tokens[session_id]
    
    @staticmethod
    def sanitize_path(path: str, base_dir: str) -> Tuple[bool, str, str]:
        """
        Sanitize and validate a file path to prevent directory traversal.
        
        Args:
            path: Path to sanitize
            base_dir: Base directory that path must be within
            
        Returns:
            Tuple of (is_valid, sanitized_path, error_message)
        """
        if not path:
            return False, "", "Path is empty"
        
        try:
            # Resolve to absolute path
            abs_path = os.path.abspath(path)
            abs_base = os.path.abspath(base_dir)
            
            # Check if path is within base directory
            if not abs_path.startswith(abs_base):
                return False, "", "Path is outside allowed directory"
            
            return True, abs_path, ""
            
        except Exception as e:
            return False, "", f"Invalid path: {str(e)}"
    
    @staticmethod
    def sanitize_sql_like_pattern(pattern: str) -> str:
        """
        Sanitize a pattern for SQL LIKE queries.
        
        Escapes special characters: %, _, [, ]
        
        Args:
            pattern: Search pattern
            
        Returns:
            Escaped pattern safe for SQL LIKE
        """
        if not pattern:
            return ""
        
        # Escape special LIKE characters
        escaped = pattern.replace('\\', '\\\\')  # Escape backslash first
        escaped = escaped.replace('%', '\\%')
        escaped = escaped.replace('_', '\\_')
        escaped = escaped.replace('[', '\\[')
        escaped = escaped.replace(']', '\\]')
        
        return escaped


class CSRFProtection:
    """
    CSRF protection middleware for Flask forms.
    """
    
    @staticmethod
    def generate_token(session_id: str) -> str:
        """Generate CSRF token for a session"""
        return SecurityUtils.generate_csrf_token(session_id)
    
    @staticmethod
    def validate_token(session_id: str, token: str) -> bool:
        """Validate CSRF token for a session"""
        return SecurityUtils.validate_csrf_token(session_id, token)
    
    @staticmethod
    def protect_form(form_data: dict, session_id: str) -> Tuple[bool, str]:
        """
        Validate CSRF token in form data.
        
        Args:
            form_data: Form data dictionary
            session_id: Session identifier
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        token = form_data.get('csrf_token', '')
        
        if not token:
            return False, "CSRF token is missing"
        
        if not CSRFProtection.validate_token(session_id, token):
            return False, "Invalid CSRF token"
        
        return True, ""
