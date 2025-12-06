"""
Authentication routes - placeholder for future implementation
"""
from flask import redirect, url_for, render_template
from app.auth import auth_bp

@auth_bp.route('/')
def index():
    """Root route - redirect to farmer login"""
    return redirect(url_for('farmer.login'))

@auth_bp.route('/home')
def home():
    """Home page with links to farmer and admin portals"""
    return render_template('home.html')
