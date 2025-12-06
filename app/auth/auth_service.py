"""
Authentication service for farmers and administrators
Handles registration, login, and logout operations
"""
from typing import Optional, Tuple, Dict, Any
from app.repositories import FarmerRepository, AdminRepository, AdminActionRepository
from app.auth.password_utils import (
    PasswordHasher, validate_email, validate_name
)
from app.auth.session_manager import session_manager, Session
from app.data_structures.hash_table import UserHashTable
from app.data_structures.bst import UserBST
from app.config import Config


class AuthService:
    """
    Authentication service for the Crop Disease Detection System.
    
    Features:
    - Farmer registration and authentication
    - Administrator registration and authentication
    - Session management
    - Hash table for O(1) duplicate email checking
    - BST for sorted administrator storage
    """
    
    def __init__(self):
        """Initialize authentication service with data structures."""
        # Hash table for fast email lookup (duplicate checking)
        self.farmer_hash_table = UserHashTable(size=1000)
        self.admin_hash_table = UserHashTable(size=500)
        
        # BST for sorted administrator storage
        self.admin_bst = UserBST()
        
        # Load existing users into data structures
        self._load_existing_users()
    
    def _load_existing_users(self) -> None:
        """Load existing users from database into data structures."""
        # Load farmers into hash table
        farmers = FarmerRepository.find_all()
        for farmer in farmers:
            self.farmer_hash_table.insert(farmer['email'], farmer)
        
        # Load admins into hash table and BST
        admins = AdminRepository.find_all()
        for admin in admins:
            self.admin_hash_table.insert(admin['email'], admin)
            self.admin_bst.insert(admin['A_Code'], admin)
    
    def register_farmer(self, name: str, email: str, password: str) -> Tuple[bool, str, Optional[int]]:
        """
        Register a new farmer account.
        
        Validation:
        - Name must be valid (2-100 characters)
        - Email must be valid format and unique
        - Password must meet strength requirements
        
        Args:
            name: Farmer's name
            email: Farmer's email
            password: Farmer's password (plain text)
            
        Returns:
            Tuple of (success, message, farmer_code)
            - success: True if registration successful
            - message: Success or error message
            - farmer_code: F_code if successful, None otherwise
        """
        # Validate name
        is_valid, error_msg = validate_name(name)
        if not is_valid:
            return False, error_msg, None
        
        # Validate email format
        is_valid, error_msg = validate_email(email)
        if not is_valid:
            return False, error_msg, None
        
        # Check for duplicate email using hash table (O(1))
        if self.farmer_hash_table.exists(email):
            return False, "Email already registered", None
        
        # Also check database to ensure consistency
        if FarmerRepository.email_exists(email):
            return False, "Email already registered", None
        
        # Validate password strength
        try:
            hashed_password = PasswordHasher.hash_password(password)
        except ValueError as e:
            return False, str(e), None
        
        # Create farmer in database
        farmer_code = FarmerRepository.create(name, email, hashed_password)
        
        if farmer_code is None:
            return False, "Failed to create farmer account", None
        
        # Add to hash table for future lookups
        farmer_data = {
            'F_code': farmer_code,
            'name': name,
            'email': email,
            'password': hashed_password
        }
        self.farmer_hash_table.insert(email, farmer_data)
        
        return True, "Farmer registered successfully", farmer_code
    
    def login_farmer(self, email: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """
        Authenticate a farmer and create a session.
        
        Args:
            email: Farmer's email
            password: Farmer's password (plain text)
            
        Returns:
            Tuple of (success, message, session_id)
            - success: True if login successful
            - message: Success or error message
            - session_id: Session ID if successful, None otherwise
        """
        # Validate inputs
        if not email or not password:
            return False, "Email and password are required", None
        
        # Look up farmer by email
        farmer = FarmerRepository.find_by_email(email)
        
        if farmer is None:
            return False, "Invalid email or password", None
        
        # Check if account is active
        if not farmer.get('is_active', True):
            return False, "Account is deactivated", None
        
        # Verify password
        if not PasswordHasher.verify_password(password, farmer['password']):
            return False, "Invalid email or password", None
        
        # Create session
        session_id = session_manager.create_session(
            user_code=farmer['F_code'],
            user_type='farmer',
            email=farmer['email'],
            name=farmer['name']
        )
        
        return True, "Login successful", session_id
    
    def logout(self, session_id: str) -> Tuple[bool, str]:
        """
        Log out a user by destroying their session.
        
        Args:
            session_id: Session ID to destroy
            
        Returns:
            Tuple of (success, message)
        """
        if session_manager.destroy_session(session_id):
            return True, "Logged out successfully"
        return False, "Session not found"
    
    def register_admin(self, name: str, email: str, password: str, 
                      special_code: str) -> Tuple[bool, str, Optional[int]]:
        """
        Register a new administrator account.
        
        Validation:
        - Name must be valid (2-100 characters)
        - Email must be valid format and unique
        - Password must meet strength requirements
        - Special code must match the configured admin code
        
        Args:
            name: Administrator's name
            email: Administrator's email
            password: Administrator's password (plain text)
            special_code: Special authorization code
            
        Returns:
            Tuple of (success, message, admin_code)
            - success: True if registration successful
            - message: Success or error message
            - admin_code: A_Code if successful, None otherwise
        """
        # Validate name
        is_valid, error_msg = validate_name(name)
        if not is_valid:
            return False, error_msg, None
        
        # Validate email format
        is_valid, error_msg = validate_email(email)
        if not is_valid:
            return False, error_msg, None
        
        # Validate special code
        if special_code != Config.ADMIN_SPECIAL_CODE:
            return False, "Invalid special authorization code", None
        
        # Check for duplicate email using hash table (O(1))
        if self.admin_hash_table.exists(email):
            return False, "Email already registered", None
        
        # Also check database to ensure consistency
        if AdminRepository.email_exists(email):
            return False, "Email already registered", None
        
        # Validate password strength
        try:
            hashed_password = PasswordHasher.hash_password(password)
        except ValueError as e:
            return False, str(e), None
        
        # Create administrator in database
        admin_code = AdminRepository.create(name, email, hashed_password, special_code)
        
        if admin_code is None:
            return False, "Failed to create administrator account", None
        
        # Add to hash table and BST for future lookups
        admin_data = {
            'A_Code': admin_code,
            'name': name,
            'email': email,
            'password': hashed_password,
            'Special_Code': special_code
        }
        self.admin_hash_table.insert(email, admin_data)
        self.admin_bst.insert(admin_code, admin_data)
        
        return True, "Administrator registered successfully", admin_code
    
    def login_admin(self, email: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """
        Authenticate an administrator and create a privileged session.
        
        Security: Error messages do not reveal whether email exists.
        
        Args:
            email: Administrator's email
            password: Administrator's password (plain text)
            
        Returns:
            Tuple of (success, message, session_id)
            - success: True if login successful
            - message: Success or error message (generic for security)
            - session_id: Session ID if successful, None otherwise
        """
        # Validate inputs
        if not email or not password:
            return False, "Invalid credentials", None
        
        # Look up administrator by email
        admin = AdminRepository.find_by_email(email)
        
        if admin is None:
            # Generic error message for security
            return False, "Invalid credentials", None
        
        # Check if account is active
        if not admin.get('is_active', True):
            return False, "Invalid credentials", None
        
        # Verify password
        if not PasswordHasher.verify_password(password, admin['password']):
            return False, "Invalid credentials", None
        
        # Create privileged session
        session_id = session_manager.create_session(
            user_code=admin['A_Code'],
            user_type='admin',
            email=admin['email'],
            name=admin['name']
        )
        
        # Log admin login action
        AdminActionRepository.create(
            a_code=admin['A_Code'],
            action_type='login',
            action_details=f"Administrator {admin['name']} logged in"
        )
        
        return True, "Login successful", session_id
    
    def get_current_user(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current user information from session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with user information if session valid, None otherwise
        """
        session = session_manager.get_session(session_id)
        
        if session is None:
            return None
        
        return {
            'user_code': session.user_code,
            'user_type': session.user_type,
            'email': session.email,
            'name': session.name
        }
    
    def validate_session(self, session_id: str, required_user_type: Optional[str] = None) -> bool:
        """
        Validate a session and optionally check user type.
        
        Args:
            session_id: Session ID to validate
            required_user_type: Required user type ('farmer' or 'admin'), optional
            
        Returns:
            True if session is valid and matches required type, False otherwise
        """
        return session_manager.validate_session(session_id, required_user_type)


# Global authentication service instance
auth_service = AuthService()
