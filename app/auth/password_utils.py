"""
Password hashing and validation utilities
Uses bcrypt for secure password hashing
"""
import re
import bcrypt
from typing import Tuple


class PasswordHasher:
    """
    Utility class for password hashing and verification using bcrypt.
    
    Security features:
    - Uses bcrypt with salt for secure hashing
    - Validates password strength before hashing
    - Prevents timing attacks during verification
    """
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Hashed password as a string
            
        Raises:
            ValueError: If password doesn't meet strength requirements
        """
        # Validate password strength first
        is_valid, error_message = PasswordHasher.validate_password_strength(password)
        if not is_valid:
            raise ValueError(error_message)
        
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password to verify
            hashed_password: Hashed password to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            print(f"Error verifying password: {e}")
            return False
    
    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, str]:
        """
        Validate password strength according to security requirements.
        
        Requirements:
        - Minimum 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is empty string
        """
        if not password:
            return False, "Password cannot be empty"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not email:
        return False, "Email cannot be empty"
    
    # Basic email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return False, "Invalid email format"
    
    return True, ""


def validate_name(name: str) -> Tuple[bool, str]:
    """
    Validate name field.
    
    Args:
        name: Name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not name:
        return False, "Name cannot be empty"
    
    if len(name.strip()) < 2:
        return False, "Name must be at least 2 characters long"
    
    if len(name) > 100:
        return False, "Name must not exceed 100 characters"
    
    return True, ""
