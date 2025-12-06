"""
Farmer blueprint for farmer-specific functionality
"""
from flask import Blueprint

farmer_bp = Blueprint('farmer', __name__, url_prefix='/farmer')

from app.farmer import routes
