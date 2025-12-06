"""
Authentication blueprint for farmer and admin authentication
"""
from flask import Blueprint

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Import authentication components
from app.auth.auth_service import auth_service, AuthService
from app.auth.session_manager import session_manager, SessionManager, Session
from app.auth.password_utils import PasswordHasher, validate_email, validate_name

from app.auth import routes
