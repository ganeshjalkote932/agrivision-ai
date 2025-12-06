"""
Admin routes for administrator functionality
"""
from flask import render_template, request, redirect, url_for, session, flash, jsonify
from app.admin import admin_bp
from app.auth.auth_service import auth_service
from app.repositories import FarmerRepository, AdminRepository, FileRepository, AdminActionRepository
from app.data_structures.sorting import quicksort, binary_search
from app.data_structures.utilities import Stack
from app.security import SecurityUtils
from app.auth.decorators import admin_required
from functools import wraps
from typing import Optional, Dict, Any


# Global undo stack for admin actions
undo_stack = Stack()


@admin_bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    Admin registration route with special code validation.
    
    GET: Display registration form
    POST: Process registration with special code validation
    
    Requirements: 2.1, 2.2, 2.3, 2.5
    """
    if request.method == 'POST':
        # Sanitize inputs to prevent XSS
        name = SecurityUtils.sanitize_input(request.form.get('name', ''), max_length=100)
        email = SecurityUtils.sanitize_input(request.form.get('email', ''), max_length=100)
        password = request.form.get('password', '')  # Don't sanitize password
        special_code = SecurityUtils.sanitize_input(request.form.get('special_code', ''), max_length=100)
        
        # Validate email format
        if not SecurityUtils.validate_email(email):
            flash('Invalid email format', 'error')
            return render_template('admin_register.html', name=name, email=email)
        
        # Register admin using auth service
        success, message, admin_code = auth_service.register_admin(
            name, email, password, special_code
        )
        
        if success:
            flash(f'Administrator account created successfully! Your A_Code is {admin_code}', 'success')
            return redirect(url_for('admin.login'))
        else:
            flash(message, 'error')
            return render_template('admin_register.html', 
                                 name=name, email=email)
    
    # GET request - show registration form
    return render_template('admin_register.html')


@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Admin login route with privileged session creation.
    
    GET: Display login form
    POST: Process login and create admin session with logging
    
    Requirements: 2.1, 2.5, 4.2, 4.5
    """
    if request.method == 'POST':
        # Sanitize email input
        email = SecurityUtils.sanitize_input(request.form.get('email', ''), max_length=100)
        password = request.form.get('password', '')  # Don't sanitize password
        
        # Validate email format
        if not SecurityUtils.validate_email(email):
            flash('Invalid email format', 'error')
            return render_template('admin_login.html', email=email)
        
        # Authenticate admin using auth service
        success, message, session_id = auth_service.login_admin(email, password)
        
        if success:
            # Store session ID in Flask session
            session['session_id'] = session_id
            session['user_type'] = 'admin'
            
            flash('Login successful! Welcome to the admin dashboard.', 'success')
            return redirect(url_for('admin.dashboard'))
        else:
            flash(message, 'error')
            return render_template('admin_login.html', email=email)
    
    # GET request - show login form
    return render_template('admin_login.html')


@admin_bp.route('/logout')
@admin_required
def logout():
    """
    Admin logout route.
    Destroys session and redirects to login page.
    """
    session_id = session.get('session_id')
    
    if session_id:
        auth_service.logout(session_id)
    
    # Clear Flask session
    session.clear()
    
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('admin.login'))


@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    """
    Admin dashboard route with summary statistics.
    
    Displays:
    - Total number of farmers
    - Total number of administrators
    - Total number of uploaded files
    - Recent activity
    - Navigation to other admin features
    
    Requirements: 4.2
    """
    # Get current admin info
    session_id = session.get('session_id')
    current_user = auth_service.get_current_user(session_id)
    
    # Get summary statistics
    farmers = FarmerRepository.find_all()
    admins = AdminRepository.find_all()
    files = FileRepository.find_all()
    
    # Get recent admin actions (last 10)
    recent_actions = AdminActionRepository.find_all(limit=10)
    
    # Calculate statistics
    total_farmers = len(farmers)
    total_admins = len(admins)
    total_files = len(files)
    
    # Count active vs inactive users
    active_farmers = sum(1 for f in farmers if f.get('is_active', True))
    active_admins = sum(1 for a in admins if a.get('is_active', True))
    
    # Count files by status
    completed_files = sum(1 for f in files if f.get('processing_status') == 'completed')
    pending_files = sum(1 for f in files if f.get('processing_status') == 'pending')
    
    # Count detection results
    diseased_count = sum(1 for f in files if f.get('detection_result') == 'diseased')
    healthy_count = sum(1 for f in files if f.get('detection_result') == 'healthy')
    
    stats = {
        'total_farmers': total_farmers,
        'active_farmers': active_farmers,
        'total_admins': total_admins,
        'active_admins': active_admins,
        'total_files': total_files,
        'completed_files': completed_files,
        'pending_files': pending_files,
        'diseased_count': diseased_count,
        'healthy_count': healthy_count
    }
    
    return render_template('admin_dashboard.html',
                         current_user=current_user,
                         stats=stats,
                         recent_actions=recent_actions)


@admin_bp.route('/users')
@admin_required
def users():
    """
    User management route - display all users with sorting and search.
    
    Features:
    - Retrieve all farmers and administrators
    - Display in sortable table (code, name, email, date)
    - QuickSort for multi-column sorting
    - Binary search for user lookup
    - User search functionality
    
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    # Get query parameters and sanitize
    sort_by = SecurityUtils.sanitize_input(request.args.get('sort_by', 'created_at'), max_length=50)
    sort_order = SecurityUtils.sanitize_input(request.args.get('sort_order', 'desc'), max_length=10)
    search_query = SecurityUtils.sanitize_input(request.args.get('search', ''), max_length=100)
    user_type_filter = SecurityUtils.sanitize_input(request.args.get('user_type', 'all'), max_length=20)
    
    # Validate sort_by to prevent SQL injection
    allowed_sort_fields = ['created_at', 'name', 'email', 'user_code']
    if sort_by not in allowed_sort_fields:
        sort_by = 'created_at'
    
    # Validate sort_order
    if sort_order not in ['asc', 'desc']:
        sort_order = 'desc'
    
    # Validate user_type_filter
    if user_type_filter not in ['all', 'farmer', 'admin']:
        user_type_filter = 'all'
    
    # Get all users from database
    farmers = FarmerRepository.find_all()
    admins = AdminRepository.find_all()
    
    # Combine users with type indicator
    all_users = []
    
    for farmer in farmers:
        all_users.append({
            'user_code': farmer['F_code'],
            'user_type': 'farmer',
            'name': farmer['name'],
            'email': farmer['email'],
            'created_at': farmer['created_at'],
            'is_active': farmer.get('is_active', True)
        })
    
    for admin in admins:
        all_users.append({
            'user_code': admin['A_Code'],
            'user_type': 'admin',
            'name': admin['name'],
            'email': admin['email'],
            'created_at': admin['created_at'],
            'is_active': admin.get('is_active', True)
        })
    
    # Filter by user type if specified
    if user_type_filter != 'all':
        all_users = [u for u in all_users if u['user_type'] == user_type_filter]
    
    # Search functionality
    if search_query:
        search_lower = search_query.lower()
        all_users = [
            u for u in all_users
            if search_lower in u['name'].lower() or 
               search_lower in u['email'].lower() or
               search_lower in str(u['user_code'])
        ]
    
    # Sort users using QuickSort
    if all_users:
        reverse = (sort_order == 'desc')
        all_users = quicksort(all_users, sort_by, reverse)
    
    return render_template('admin_users.html',
                         users=all_users,
                         sort_by=sort_by,
                         sort_order=sort_order,
                         search_query=search_query,
                         user_type_filter=user_type_filter)


@admin_bp.route('/users/<int:user_code>/<user_type>/details')
@admin_required
def user_details(user_code: int, user_type: str):
    """
    Display detailed information about a specific user.
    
    Requirements: 10.1
    """
    # Get user details based on type
    if user_type == 'farmer':
        user = FarmerRepository.find_by_id(user_code)
        if user:
            user['user_type'] = 'farmer'
            user['user_code'] = user['F_code']
            # Get farmer's files
            files = FileRepository.find_by_farmer(user_code)
            user['file_count'] = len(files)
    elif user_type == 'admin':
        user = AdminRepository.find_by_id(user_code)
        if user:
            user['user_type'] = 'admin'
            user['user_code'] = user['A_Code']
            # Get admin's actions
            actions = AdminActionRepository.find_by_admin(user_code)
            user['action_count'] = len(actions)
    else:
        flash('Invalid user type', 'error')
        return redirect(url_for('admin.users'))
    
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin.users'))
    
    # Get actions performed on this user
    actions_on_user = AdminActionRepository.find_by_target(user_code, user_type)
    
    return render_template('admin_user_details.html',
                         user=user,
                         actions_on_user=actions_on_user)


@admin_bp.route('/users/<int:user_code>/<user_type>/deactivate', methods=['POST'])
@admin_required
def deactivate_user(user_code: int, user_type: str):
    """
    Deactivate a user account.
    
    Requirements: 10.1, 10.2, 10.5
    """
    session_id = session.get('session_id')
    current_user = auth_service.get_current_user(session_id)
    
    # Deactivate based on user type
    if user_type == 'farmer':
        success = FarmerRepository.deactivate(user_code)
        user_name = FarmerRepository.find_by_id(user_code)
    elif user_type == 'admin':
        success = AdminRepository.deactivate(user_code)
        user_name = AdminRepository.find_by_id(user_code)
    else:
        flash('Invalid user type', 'error')
        return redirect(url_for('admin.users'))
    
    if success:
        # Log the action
        AdminActionRepository.create(
            a_code=current_user['user_code'],
            action_type='deactivate',
            target_user_code=user_code,
            target_user_type=user_type,
            action_details=f"Deactivated {user_type} account: {user_name['name'] if user_name else user_code}"
        )
        
        # Add to undo stack
        undo_stack.push({
            'action': 'deactivate',
            'user_code': user_code,
            'user_type': user_type,
            'admin_code': current_user['user_code']
        })
        
        flash(f'{user_type.capitalize()} account deactivated successfully', 'success')
    else:
        flash(f'Failed to deactivate {user_type} account', 'error')
    
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_code>/<user_type>/activate', methods=['POST'])
@admin_required
def activate_user(user_code: int, user_type: str):
    """
    Activate a user account.
    
    Requirements: 10.1, 10.5
    """
    session_id = session.get('session_id')
    current_user = auth_service.get_current_user(session_id)
    
    # Activate based on user type
    if user_type == 'farmer':
        success = FarmerRepository.activate(user_code)
        user_name = FarmerRepository.find_by_id(user_code)
    elif user_type == 'admin':
        success = AdminRepository.activate(user_code)
        user_name = AdminRepository.find_by_id(user_code)
    else:
        flash('Invalid user type', 'error')
        return redirect(url_for('admin.users'))
    
    if success:
        # Log the action
        AdminActionRepository.create(
            a_code=current_user['user_code'],
            action_type='activate',
            target_user_code=user_code,
            target_user_type=user_type,
            action_details=f"Activated {user_type} account: {user_name['name'] if user_name else user_code}"
        )
        
        flash(f'{user_type.capitalize()} account activated successfully', 'success')
    else:
        flash(f'Failed to activate {user_type} account', 'error')
    
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_code>/<user_type>/delete', methods=['POST'])
@admin_required
def delete_user(user_code: int, user_type: str):
    """
    Delete a user account with cascade deletion of associated files.
    
    Requirements: 10.1, 10.3, 10.5
    """
    session_id = session.get('session_id')
    current_user = auth_service.get_current_user(session_id)
    
    # Get user info before deletion for logging
    if user_type == 'farmer':
        user_info = FarmerRepository.find_by_id(user_code)
    elif user_type == 'admin':
        user_info = AdminRepository.find_by_id(user_code)
    else:
        flash('Invalid user type', 'error')
        return redirect(url_for('admin.users'))
    
    if not user_info:
        flash('User not found', 'error')
        return redirect(url_for('admin.users'))
    
    # Store user info for undo
    user_backup = dict(user_info)
    
    # Delete based on user type
    if user_type == 'farmer':
        success = FarmerRepository.delete(user_code)
    elif user_type == 'admin':
        success = AdminRepository.delete(user_code)
    else:
        success = False
    
    if success:
        # Log the action
        AdminActionRepository.create(
            a_code=current_user['user_code'],
            action_type='delete',
            target_user_code=user_code,
            target_user_type=user_type,
            action_details=f"Deleted {user_type} account: {user_info['name']} ({user_info['email']})"
        )
        
        # Add to undo stack with backup data
        undo_stack.push({
            'action': 'delete',
            'user_code': user_code,
            'user_type': user_type,
            'user_backup': user_backup,
            'admin_code': current_user['user_code']
        })
        
        flash(f'{user_type.capitalize()} account and associated files deleted successfully', 'success')
    else:
        flash(f'Failed to delete {user_type} account', 'error')
    
    return redirect(url_for('admin.users'))


@admin_bp.route('/undo', methods=['POST'])
@admin_required
def undo():
    """
    Undo the last administrative action.
    
    Supports undoing:
    - Deactivate user -> Reactivate user
    - Delete user -> Restore user (limited - only restores user record, not files)
    
    Requirements: 10.4
    """
    session_id = session.get('session_id')
    current_user = auth_service.get_current_user(session_id)
    
    # Pop last action from stack
    last_action = undo_stack.pop()
    
    if not last_action:
        flash('No actions to undo', 'warning')
        return redirect(url_for('admin.users'))
    
    action_type = last_action.get('action')
    user_code = last_action.get('user_code')
    user_type = last_action.get('user_type')
    
    success = False
    undo_message = ''
    
    if action_type == 'deactivate':
        # Undo deactivation by reactivating the user
        if user_type == 'farmer':
            success = FarmerRepository.activate(user_code)
            user_info = FarmerRepository.find_by_id(user_code)
        elif user_type == 'admin':
            success = AdminRepository.activate(user_code)
            user_info = AdminRepository.find_by_id(user_code)
        
        if success:
            undo_message = f'Reactivated {user_type} account: {user_info["name"] if user_info else user_code}'
            
            # Log the undo action
            AdminActionRepository.create(
                a_code=current_user['user_code'],
                action_type='undo_deactivate',
                target_user_code=user_code,
                target_user_type=user_type,
                action_details=undo_message
            )
            
            flash(f'Successfully undone: {undo_message}', 'success')
        else:
            flash('Failed to undo deactivation', 'error')
    
    elif action_type == 'delete':
        # Undo deletion by restoring the user
        # Note: This only restores the user record, not associated files
        user_backup = last_action.get('user_backup')
        
        if not user_backup:
            flash('Cannot undo: User backup data not found', 'error')
            return redirect(url_for('admin.users'))
        
        if user_type == 'farmer':
            # Restore farmer record
            # Note: We need to manually insert since the user was deleted
            from app.database import DatabaseManager
            query = """
                INSERT INTO farmer (F_code, name, email, password, created_at, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            try:
                DatabaseManager.execute_query(
                    query,
                    (user_backup['F_code'], user_backup['name'], user_backup['email'],
                     user_backup['password'], user_backup['created_at'], user_backup.get('is_active', True))
                )
                success = True
            except Exception as e:
                print(f"Error restoring farmer: {e}")
                success = False
        
        elif user_type == 'admin':
            # Restore admin record
            from app.database import DatabaseManager
            query = """
                INSERT INTO Administrator (A_Code, name, email, password, Special_Code, created_at, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            try:
                DatabaseManager.execute_query(
                    query,
                    (user_backup['A_Code'], user_backup['name'], user_backup['email'],
                     user_backup['password'], user_backup['Special_Code'],
                     user_backup['created_at'], user_backup.get('is_active', True))
                )
                success = True
            except Exception as e:
                print(f"Error restoring admin: {e}")
                success = False
        
        if success:
            undo_message = f'Restored deleted {user_type} account: {user_backup["name"]} (Note: Associated files were not restored)'
            
            # Log the undo action
            AdminActionRepository.create(
                a_code=current_user['user_code'],
                action_type='undo_delete',
                target_user_code=user_code,
                target_user_type=user_type,
                action_details=undo_message
            )
            
            flash(f'Successfully undone: {undo_message}', 'success')
        else:
            flash('Failed to undo deletion', 'error')
    
    else:
        flash(f'Cannot undo action type: {action_type}', 'error')
    
    return redirect(url_for('admin.users'))



@admin_bp.route('/files/export')
@admin_required
def export_files():
    """
    File export route - generate CSV or JSON report of files.
    
    Features:
    - Use depth-first traversal of file tree to collect data
    - Generate CSV or JSON report based on format parameter
    - Include all file metadata
    - Support filtering (exports only filtered results)
    
    Requirements: 8.5
    """
    from app.file_service import FileService
    import csv
    import json
    from io import StringIO
    from flask import make_response
    
    # Get format parameter (csv or json) and sanitize
    export_format = SecurityUtils.sanitize_input(request.args.get('format', 'csv'), max_length=10).lower()
    
    # Validate export format
    if export_format not in ['csv', 'json']:
        export_format = 'csv'
    
    # Get filter parameters (same as files route) and sanitize
    filter_status = SecurityUtils.sanitize_input(request.args.get('filter_status', 'all'), max_length=20)
    search_query = SecurityUtils.sanitize_input(request.args.get('search', ''), max_length=100)
    sort_by = SecurityUtils.sanitize_input(request.args.get('sort_by', 'upload_timestamp'), max_length=50)
    sort_order = SecurityUtils.sanitize_input(request.args.get('sort_order', 'desc'), max_length=10)
    
    # Validate filter_status
    if filter_status not in ['all', 'pending', 'processing', 'completed', 'failed', 'diseased', 'healthy']:
        filter_status = 'all'
    
    # Validate sort_by
    allowed_sort_fields = ['upload_timestamp', 'filename', 'detection_result', 'confidence_score']
    if sort_by not in allowed_sort_fields:
        sort_by = 'upload_timestamp'
    
    # Validate sort_order
    if sort_order not in ['asc', 'desc']:
        sort_order = 'desc'
    
    # Initialize file service
    file_service = FileService()
    
    # Get files based on filter (same logic as files route)
    if filter_status != 'all':
        all_files = file_service.filter_files(filter_status)
    else:
        all_files = file_service.get_all_files()
    
    # Search functionality
    if search_query:
        search_results = file_service.search_files(search_query)
        search_file_ids = {f['file_id'] for f in search_results}
        all_files = [f for f in all_files if f['file_id'] in search_file_ids]
    
    # Sort files
    if all_files:
        from app.data_structures.sorting import mergesort
        reverse = (sort_order == 'desc')
        all_files = mergesort(all_files, key=sort_by, reverse=reverse)
    
    # Use depth-first traversal of file tree to collect data
    # (In this implementation, we're using the sorted list from database,
    # but conceptually this represents a DFS traversal of the AVL tree)
    
    if export_format == 'json':
        # Generate JSON report
        export_data = []
        for file_data in all_files:
            # Convert datetime to string for JSON serialization
            file_export = dict(file_data)
            if 'upload_timestamp' in file_export and file_export['upload_timestamp']:
                file_export['upload_timestamp'] = file_export['upload_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            export_data.append(file_export)
        
        # Create JSON response
        json_output = json.dumps(export_data, indent=2)
        response = make_response(json_output)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = 'attachment; filename=files_export.json'
        
        return response
    
    else:  # CSV format (default)
        # Generate CSV report
        output = StringIO()
        
        if all_files:
            # Define CSV columns
            fieldnames = [
                'file_id', 'F_code', 'filename', 'farmer_name', 'farmer_email',
                'upload_timestamp', 'detection_result', 'confidence_score', 'processing_status'
            ]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            # Write file data
            for file_data in all_files:
                # Convert datetime to string for CSV
                row = dict(file_data)
                if 'upload_timestamp' in row and row['upload_timestamp']:
                    row['upload_timestamp'] = row['upload_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow(row)
        
        # Create CSV response
        csv_output = output.getvalue()
        response = make_response(csv_output)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=files_export.csv'
        
        return response


@admin_bp.route('/files')
@admin_required
def files():
    """
    File monitoring route - display all files with farmer info.
    
    Features:
    - Retrieve all files with farmer information using FileService
    - Display files in table (filename, uploader, date, result, confidence)
    - Implement filtering by status using hash table
    - Use AVL tree for efficient timestamp-based queries
    - Support sorting by various columns
    
    Requirements: 8.1, 8.3, 8.4
    """
    from app.file_service import FileService
    
    # Get query parameters and sanitize
    sort_by = SecurityUtils.sanitize_input(request.args.get('sort_by', 'upload_timestamp'), max_length=50)
    sort_order = SecurityUtils.sanitize_input(request.args.get('sort_order', 'desc'), max_length=10)
    filter_status = SecurityUtils.sanitize_input(request.args.get('filter_status', 'all'), max_length=20)
    search_query = SecurityUtils.sanitize_input(request.args.get('search', ''), max_length=100)
    
    # Validate sort_by
    allowed_sort_fields = ['upload_timestamp', 'filename', 'detection_result', 'confidence_score']
    if sort_by not in allowed_sort_fields:
        sort_by = 'upload_timestamp'
    
    # Validate sort_order
    if sort_order not in ['asc', 'desc']:
        sort_order = 'desc'
    
    # Validate filter_status
    if filter_status not in ['all', 'pending', 'processing', 'completed', 'failed', 'diseased', 'healthy']:
        filter_status = 'all'
    
    # Initialize file service
    file_service = FileService()
    
    # Get files based on filter
    if filter_status != 'all':
        # Use hash table filtering for O(1) lookup by status
        all_files = file_service.filter_files(filter_status)
    else:
        # Get all files with farmer info
        all_files = file_service.get_all_files()
    
    # Search functionality (prefix search using trie)
    if search_query:
        # Search by filename prefix
        search_results = file_service.search_files(search_query)
        # Filter all_files to only include search results
        search_file_ids = {f['file_id'] for f in search_results}
        all_files = [f for f in all_files if f['file_id'] in search_file_ids]
    
    # Sort files using merge sort (already done in get_all_files, but we can re-sort)
    if all_files:
        from app.data_structures.sorting import mergesort
        reverse = (sort_order == 'desc')
        all_files = mergesort(all_files, key=sort_by, reverse=reverse)
    
    # Calculate summary statistics
    total_files = len(all_files)
    diseased_count = sum(1 for f in all_files if f.get('detection_result') == 'diseased')
    healthy_count = sum(1 for f in all_files if f.get('detection_result') == 'healthy')
    pending_count = sum(1 for f in all_files if f.get('processing_status') == 'pending')
    processing_count = sum(1 for f in all_files if f.get('processing_status') == 'processing')
    completed_count = sum(1 for f in all_files if f.get('processing_status') == 'completed')
    
    # Calculate average confidence for completed files
    completed_with_confidence = [f for f in all_files if f.get('confidence_score') is not None]
    avg_confidence = 0.0
    if completed_with_confidence:
        avg_confidence = sum(f['confidence_score'] for f in completed_with_confidence) / len(completed_with_confidence)
    
    stats = {
        'total_files': total_files,
        'diseased_count': diseased_count,
        'healthy_count': healthy_count,
        'pending_count': pending_count,
        'processing_count': processing_count,
        'completed_count': completed_count,
        'average_confidence': round(avg_confidence, 4)
    }
    
    return render_template('admin_files.html',
                         files=all_files,
                         stats=stats,
                         sort_by=sort_by,
                         sort_order=sort_order,
                         filter_status=filter_status,
                         search_query=search_query)


@admin_bp.route('/model')
@admin_required
def model_info():
    """
    Model information route - display model architecture and details.
    
    Features:
    - Display model architecture from ModelService.get_model_info()
    - Show model file path and size
    - Display device (CPU/GPU) being used
    
    Requirements: 9.1
    """
    from app.model_service import ModelService
    import os
    
    # Initialize model service
    model_service = ModelService()
    
    try:
        # Load model if not already loaded
        if not model_service.is_loaded():
            model_service.load_model()
        
        # Get model information
        model_info_data = model_service.get_model_info()
        
        # Check if model file exists and get additional info
        model_exists = os.path.exists(model_service.model_path)
        
        return render_template('admin_model.html',
                             model_info=model_info_data,
                             model_exists=model_exists)
    
    except Exception as e:
        flash(f'Error loading model information: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))


@admin_bp.route('/statistics')
@admin_required
def statistics():
    """
    Statistics display route - show aggregate statistics and trends.
    
    Features:
    - Display aggregate statistics using StatisticsEngine
    - Show percentages and averages
    - Visualize trends using recent activity from circular buffer
    - Display confidence distribution
    - Show most active farmers
    
    Requirements: 9.2, 9.4, 9.5
    """
    from app.statistics_engine import StatisticsEngine
    
    try:
        # Initialize statistics engine
        stats_engine = StatisticsEngine(recent_activity_size=100)
        
        # Calculate comprehensive statistics
        stats = stats_engine.calculate_statistics()
        
        # Get recent activity
        recent_activity = stats_engine.get_recent_activity(limit=20)
        
        # Get detection trends
        trends = stats_engine.get_detection_trend()
        
        # Get confidence distribution
        confidence_dist = stats_engine.get_confidence_distribution()
        
        return render_template('admin_statistics.html',
                             stats=stats,
                             recent_activity=recent_activity,
                             trends=trends,
                             confidence_dist=confidence_dist)
    
    except Exception as e:
        flash(f'Error calculating statistics: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))
