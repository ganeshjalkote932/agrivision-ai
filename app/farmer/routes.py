"""
Farmer routes for file upload and processing
"""
import os
import uuid
from datetime import datetime
from flask import request, jsonify, session, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from app.farmer import farmer_bp
from app.config import Config
from app.file_service import FileService
from app.data_structures.queue import ProcessingQueue
from app.model_service import ImagePreprocessor
from app.auth.auth_service import auth_service
from app.security import SecurityUtils
from app.auth.decorators import farmer_required

# Initialize services
file_service = FileService()
processing_queue = ProcessingQueue()


# ============================================================================
# Authentication Routes
# ============================================================================

@farmer_bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handle farmer registration.
    
    GET: Display registration form
    POST: Process registration with validation
    
    Requirements: 1.1, 1.2, 1.3, 1.4
    """
    if request.method == 'GET':
        # If already logged in, redirect to dashboard
        if 'session_id' in session and auth_service.validate_session(session['session_id'], 'farmer'):
            return redirect(url_for('farmer.dashboard'))
        
        return render_template('farmer_register.html')
    
    # POST request - process registration
    # Sanitize inputs to prevent XSS
    name = SecurityUtils.sanitize_input(request.form.get('name', ''), max_length=100)
    email = SecurityUtils.sanitize_input(request.form.get('email', ''), max_length=100)
    password = request.form.get('password', '')  # Don't sanitize password (preserve special chars)
    confirm_password = request.form.get('confirm_password', '')
    
    # Validate all fields are present
    if not all([name, email, password, confirm_password]):
        flash('All fields are required', 'error')
        return render_template('farmer_register.html', name=name, email=email)
    
    # Validate email format
    if not SecurityUtils.validate_email(email):
        flash('Invalid email format', 'error')
        return render_template('farmer_register.html', name=name, email=email)
    
    # Validate passwords match
    if password != confirm_password:
        flash('Passwords do not match', 'error')
        return render_template('farmer_register.html', name=name, email=email)
    
    # Attempt registration
    success, message, farmer_code = auth_service.register_farmer(name, email, password)
    
    if success:
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('farmer.login'))
    else:
        flash(message, 'error')
        return render_template('farmer_register.html', name=name, email=email)


@farmer_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle farmer login.
    
    GET: Display login form
    POST: Process login with session creation
    
    Requirements: 3.2, 3.3
    """
    if request.method == 'GET':
        # If already logged in, redirect to dashboard
        if 'session_id' in session and auth_service.validate_session(session['session_id'], 'farmer'):
            return redirect(url_for('farmer.dashboard'))
        
        return render_template('farmer_login.html')
    
    # POST request - process login
    # Sanitize email input
    email = SecurityUtils.sanitize_input(request.form.get('email', ''), max_length=100)
    password = request.form.get('password', '')  # Don't sanitize password
    
    # Validate inputs
    if not email or not password:
        flash('Email and password are required', 'error')
        return render_template('farmer_login.html', email=email)
    
    # Validate email format
    if not SecurityUtils.validate_email(email):
        flash('Invalid email format', 'error')
        return render_template('farmer_login.html', email=email)
    
    # Attempt login
    success, message, session_id = auth_service.login_farmer(email, password)
    
    if success:
        # Store session ID in Flask session
        session['session_id'] = session_id
        session['user_type'] = 'farmer'
        
        # Get user info for session
        user_info = auth_service.get_current_user(session_id)
        if user_info:
            session['user_id'] = user_info['user_code']
            session['user_name'] = user_info['name']
        
        flash('Login successful!', 'success')
        return redirect(url_for('farmer.dashboard'))
    else:
        flash(message, 'error')
        return render_template('farmer_login.html', email=email)


@farmer_bp.route('/logout', methods=['GET', 'POST'])
def logout():
    """
    Handle farmer logout.
    
    Destroys session and redirects to login page.
    
    Requirements: 3.3
    """
    # Get session ID
    session_id = session.get('session_id')
    
    if session_id:
        # Destroy session in auth service
        auth_service.logout(session_id)
    
    # Clear Flask session
    session.clear()
    
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('farmer.login'))


# ============================================================================
# Dashboard and Features
# ============================================================================

@farmer_bp.route('/dashboard', methods=['GET'])
@farmer_required
def dashboard():
    """
    Display farmer dashboard with upload form and basic info.
    
    Shows:
    - Farmer name and basic information
    - Upload form for hyperspectral images
    - Link to detection history
    
    Requirements: 3.2
    """
    # Get session info (already validated by decorator)
    session_id = session.get('session_id')
    
    # Get user information
    user_info = auth_service.get_current_user(session_id)
    
    if not user_info:
        flash('Session expired. Please log in again.', 'error')
        return redirect(url_for('farmer.login'))
    
    # Get farmer's recent files for quick stats
    farmer_code = user_info['user_code']
    recent_files = file_service.get_farmer_files(farmer_code, sort_by='upload_timestamp', reverse=True)
    
    # Calculate quick stats
    total_uploads = len(recent_files)
    pending_count = sum(1 for f in recent_files if f.get('processing_status') == 'pending')
    completed_count = sum(1 for f in recent_files if f.get('processing_status') == 'completed')
    
    return render_template('farmer_dashboard.html',
                         user_name=user_info['name'],
                         user_email=user_info['email'],
                         farmer_code=farmer_code,
                         total_uploads=total_uploads,
                         pending_count=pending_count,
                         completed_count=completed_count,
                         allowed_extensions=', '.join(Config.ALLOWED_EXTENSIONS))


@farmer_bp.route('/history', methods=['GET'])
@farmer_required
def history():
    """
    Display farmer's detection history with sorting and search.
    
    Features:
    - Retrieve farmer's files using FileService
    - Sort by timestamp using merge sort (descending)
    - Display filename, date, result, confidence score
    - Pagination for large result sets
    - Prefix search functionality using trie
    
    Requirements: 6.1, 6.2, 6.3, 6.5
    """
    # Get session info (already validated by decorator)
    session_id = session.get('session_id')
    
    # Get user information
    user_info = auth_service.get_current_user(session_id)
    
    if not user_info:
        flash('Session expired. Please log in again.', 'error')
        return redirect(url_for('farmer.login'))
    
    farmer_code = user_info['user_code']
    
    # Get search query parameter and sanitize
    search_query = SecurityUtils.sanitize_input(request.args.get('search', ''), max_length=100)
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Validate pagination parameters
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 20
    
    # Get sort parameters
    sort_by = request.args.get('sort_by', 'upload_timestamp')
    sort_order = request.args.get('sort_order', 'desc')
    reverse = (sort_order == 'desc')
    
    # Get files based on search query
    if search_query:
        # Use trie for prefix search
        files = file_service.search_files(search_query, f_code=farmer_code)
        # Still need to sort the search results
        from app.data_structures.sorting import mergesort
        files = mergesort(files, key=sort_by, reverse=reverse)
    else:
        # Get all farmer's files sorted by timestamp (using merge sort)
        files = file_service.get_farmer_files(farmer_code, sort_by=sort_by, reverse=reverse)
    
    # Calculate pagination
    total_files = len(files)
    total_pages = (total_files + per_page - 1) // per_page  # Ceiling division
    
    # Validate page number
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
    
    # Get files for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_files = files[start_idx:end_idx]
    
    # Format files for display
    formatted_files = []
    for file_data in paginated_files:
        formatted_file = {
            'file_id': file_data.get('file_id'),
            'filename': file_data.get('filename'),
            'upload_date': file_data.get('upload_timestamp'),
            'detection_result': file_data.get('detection_result', 'Pending'),
            'confidence_score': file_data.get('confidence_score'),
            'processing_status': file_data.get('processing_status', 'pending')
        }
        
        # Format confidence score as percentage
        if formatted_file['confidence_score'] is not None:
            formatted_file['confidence_percentage'] = f"{formatted_file['confidence_score'] * 100:.2f}%"
        else:
            formatted_file['confidence_percentage'] = 'N/A'
        
        # Capitalize result for display
        if formatted_file['detection_result']:
            formatted_file['detection_result_display'] = formatted_file['detection_result'].capitalize()
        else:
            formatted_file['detection_result_display'] = 'Pending'
        
        formatted_files.append(formatted_file)
    
    return render_template('farmer_history.html',
                         user_name=user_info['name'],
                         files=formatted_files,
                         total_files=total_files,
                         page=page,
                         per_page=per_page,
                         total_pages=total_pages,
                         search_query=search_query,
                         sort_by=sort_by,
                         sort_order=sort_order)


# ============================================================================
# Helper Functions
# ============================================================================

def allowed_file(filename: str) -> bool:
    """
    Check if file has an allowed extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@farmer_bp.route('/upload', methods=['POST'])
@farmer_required
def upload_file():
    """
    Handle file upload from farmer.
    
    Validates file, saves to disk, processes immediately, and returns results.
    
    Returns:
        JSON response with detection results or error message
    """
    # User is already validated by decorator
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Get farmer code from session
        f_code = session['user_id']
        
        # Sanitize filename to prevent directory traversal
        is_valid, safe_filename, error_msg = SecurityUtils.sanitize_filename(file.filename)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Invalid filename: {error_msg}'
            }), 400
        
        # Generate unique filename to prevent collisions
        original_filename = safe_filename
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{f_code}_{uuid.uuid4().hex}.{file_extension}"
        
        # Create full file path
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        
        # Save file to disk
        file.save(file_path)
        
        # Validate the saved file using ImagePreprocessor
        preprocessor = ImagePreprocessor()
        is_valid, error_msg = preprocessor.validate_file(file_path)
        
        if not is_valid:
            # Delete invalid file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'success': False,
                'error': f'File validation failed: {error_msg}'
            }), 400
        
        # INSTANT PROCESSING: Process the image immediately
        from app.model_service import ModelService
        model_service = ModelService()
        
        # Load model if not already loaded
        if not model_service.is_loaded():
            model_service.load_model()
        
        # Perform inference immediately
        result = model_service.predict(file_path)
        
        # Extract detection result and confidence
        detection_result = result.get('result')  # 'diseased' or 'healthy'
        confidence_score = result.get('confidence')
        
        # Save file metadata to database with results
        upload_timestamp = datetime.now()
        file_id = file_service.save_file_metadata(
            f_code=f_code,
            filename=original_filename,
            file_path=file_path,
            upload_timestamp=upload_timestamp,
            processing_status='completed'
        )
        
        if file_id is None:
            # Delete file if database save failed
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'success': False,
                'error': 'Failed to save file metadata'
            }), 500
        
        # Update file with results
        file_service.update_file_result(
            file_id=file_id,
            detection_result=detection_result,
            confidence_score=confidence_score,
            processing_status='completed'
        )
        
        # Return success with immediate results
        return jsonify({
            'success': True,
            'message': 'File processed successfully',
            'file_id': file_id,
            'filename': original_filename,
            'status': 'completed',
            'result': detection_result,
            'confidence': confidence_score,
            'confidence_percentage': f"{confidence_score * 100:.2f}%"
        }), 200
        
    except Exception as e:
        # Clean up file if it was saved
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500


@farmer_bp.route('/upload/status/<request_id>', methods=['GET'])
@farmer_required
def get_upload_status(request_id: str):
    """
    Get the status of an upload/processing request.
    
    Args:
        request_id: Unique identifier of the request
        
    Returns:
        JSON response with request status
    """
    # User is already validated by decorator
    
    try:
        # Get request from queue
        request_obj = processing_queue.get_request(request_id)
        
        if request_obj is None:
            return jsonify({
                'success': False,
                'error': 'Request not found'
            }), 404
        
        # Verify request belongs to this farmer
        f_code = session['user_id']
        if request_obj.farmer_code != f_code:
            return jsonify({
                'success': False,
                'error': 'Unauthorized'
            }), 403
        
        # Return request status
        return jsonify({
            'success': True,
            'request_id': request_id,
            'status': request_obj.status,
            'filename': request_obj.filename,
            'timestamp': request_obj.timestamp.isoformat(),
            'result': request_obj.result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get status: {str(e)}'
        }), 500
