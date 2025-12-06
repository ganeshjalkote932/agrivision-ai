"""
Repository classes for database operations
Provides CRUD operations for all database entities
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.database import DatabaseManager


class FarmerRepository:
    """Repository for farmer-related database operations"""
    
    @staticmethod
    def create(name: str, email: str, password: str) -> Optional[int]:
        """
        Create a new farmer record
        
        Args:
            name: Farmer's name
            email: Farmer's email (must be unique)
            password: Hashed password
            
        Returns:
            F_code of the created farmer, or None if creation failed
        """
        query = """
            INSERT INTO farmer (name, email, password)
            VALUES (%s, %s, %s)
        """
        try:
            farmer_id = DatabaseManager.execute_query(query, (name, email, password))
            return farmer_id
        except Exception as e:
            print(f"Error creating farmer: {e}")
            return None
    
    @staticmethod
    def find_by_id(f_code: int) -> Optional[Dict[str, Any]]:
        """
        Find a farmer by F_code
        
        Args:
            f_code: Farmer's unique code
            
        Returns:
            Dictionary containing farmer data, or None if not found
        """
        query = """
            SELECT F_code, name, email, password, created_at, is_active
            FROM farmer
            WHERE F_code = %s
        """
        try:
            return DatabaseManager.execute_query(query, (f_code,), fetch_one=True)
        except Exception as e:
            print(f"Error finding farmer by ID: {e}")
            return None
    
    @staticmethod
    def find_by_email(email: str) -> Optional[Dict[str, Any]]:
        """
        Find a farmer by email
        
        Args:
            email: Farmer's email
            
        Returns:
            Dictionary containing farmer data, or None if not found
        """
        query = """
            SELECT F_code, name, email, password, created_at, is_active
            FROM farmer
            WHERE email = %s
        """
        try:
            return DatabaseManager.execute_query(query, (email,), fetch_one=True)
        except Exception as e:
            print(f"Error finding farmer by email: {e}")
            return None
    
    @staticmethod
    def find_all() -> List[Dict[str, Any]]:
        """
        Get all farmers
        
        Returns:
            List of dictionaries containing farmer data
        """
        query = """
            SELECT F_code, name, email, created_at, is_active
            FROM farmer
            ORDER BY created_at DESC
        """
        try:
            return DatabaseManager.execute_query(query, fetch_all=True)
        except Exception as e:
            print(f"Error finding all farmers: {e}")
            return []
    
    @staticmethod
    def update(f_code: int, **kwargs) -> bool:
        """
        Update farmer record
        
        Args:
            f_code: Farmer's unique code
            **kwargs: Fields to update (name, email, password, is_active)
            
        Returns:
            True if update successful, False otherwise
        """
        allowed_fields = {'name', 'email', 'password', 'is_active'}
        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not update_fields:
            return False
        
        set_clause = ', '.join([f"{field} = %s" for field in update_fields.keys()])
        query = f"UPDATE farmer SET {set_clause} WHERE F_code = %s"
        params = list(update_fields.values()) + [f_code]
        
        try:
            DatabaseManager.execute_query(query, params)
            return True
        except Exception as e:
            print(f"Error updating farmer: {e}")
            return False
    
    @staticmethod
    def delete(f_code: int) -> bool:
        """
        Delete a farmer record (cascade deletes associated files)
        
        Args:
            f_code: Farmer's unique code
            
        Returns:
            True if deletion successful, False otherwise
        """
        query = "DELETE FROM farmer WHERE F_code = %s"
        try:
            DatabaseManager.execute_query(query, (f_code,))
            return True
        except Exception as e:
            print(f"Error deleting farmer: {e}")
            return False
    
    @staticmethod
    def deactivate(f_code: int) -> bool:
        """
        Deactivate a farmer account
        
        Args:
            f_code: Farmer's unique code
            
        Returns:
            True if deactivation successful, False otherwise
        """
        return FarmerRepository.update(f_code, is_active=False)
    
    @staticmethod
    def activate(f_code: int) -> bool:
        """
        Activate a farmer account
        
        Args:
            f_code: Farmer's unique code
            
        Returns:
            True if activation successful, False otherwise
        """
        return FarmerRepository.update(f_code, is_active=True)
    
    @staticmethod
    def email_exists(email: str) -> bool:
        """
        Check if an email already exists
        
        Args:
            email: Email to check
            
        Returns:
            True if email exists, False otherwise
        """
        query = "SELECT COUNT(*) as count FROM farmer WHERE email = %s"
        try:
            result = DatabaseManager.execute_query(query, (email,), fetch_one=True)
            return result['count'] > 0 if result else False
        except Exception as e:
            print(f"Error checking email existence: {e}")
            return False


class AdminRepository:
    """Repository for administrator-related database operations"""
    
    @staticmethod
    def create(name: str, email: str, password: str, special_code: str) -> Optional[int]:
        """
        Create a new administrator record
        
        Args:
            name: Administrator's name
            email: Administrator's email (must be unique)
            password: Hashed password
            special_code: Special authorization code
            
        Returns:
            A_Code of the created administrator, or None if creation failed
        """
        query = """
            INSERT INTO Administrator (name, email, password, Special_Code)
            VALUES (%s, %s, %s, %s)
        """
        try:
            admin_id = DatabaseManager.execute_query(query, (name, email, password, special_code))
            return admin_id
        except Exception as e:
            print(f"Error creating administrator: {e}")
            return None
    
    @staticmethod
    def find_by_id(a_code: int) -> Optional[Dict[str, Any]]:
        """
        Find an administrator by A_Code
        
        Args:
            a_code: Administrator's unique code
            
        Returns:
            Dictionary containing administrator data, or None if not found
        """
        query = """
            SELECT A_Code, name, email, password, Special_Code, created_at, is_active
            FROM Administrator
            WHERE A_Code = %s
        """
        try:
            return DatabaseManager.execute_query(query, (a_code,), fetch_one=True)
        except Exception as e:
            print(f"Error finding administrator by ID: {e}")
            return None
    
    @staticmethod
    def find_by_email(email: str) -> Optional[Dict[str, Any]]:
        """
        Find an administrator by email
        
        Args:
            email: Administrator's email
            
        Returns:
            Dictionary containing administrator data, or None if not found
        """
        query = """
            SELECT A_Code, name, email, password, Special_Code, created_at, is_active
            FROM Administrator
            WHERE email = %s
        """
        try:
            return DatabaseManager.execute_query(query, (email,), fetch_one=True)
        except Exception as e:
            print(f"Error finding administrator by email: {e}")
            return None
    
    @staticmethod
    def find_all() -> List[Dict[str, Any]]:
        """
        Get all administrators
        
        Returns:
            List of dictionaries containing administrator data
        """
        query = """
            SELECT A_Code, name, email, created_at, is_active
            FROM Administrator
            ORDER BY created_at DESC
        """
        try:
            return DatabaseManager.execute_query(query, fetch_all=True)
        except Exception as e:
            print(f"Error finding all administrators: {e}")
            return []
    
    @staticmethod
    def update(a_code: int, **kwargs) -> bool:
        """
        Update administrator record
        
        Args:
            a_code: Administrator's unique code
            **kwargs: Fields to update (name, email, password, is_active)
            
        Returns:
            True if update successful, False otherwise
        """
        allowed_fields = {'name', 'email', 'password', 'is_active'}
        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not update_fields:
            return False
        
        set_clause = ', '.join([f"{field} = %s" for field in update_fields.keys()])
        query = f"UPDATE Administrator SET {set_clause} WHERE A_Code = %s"
        params = list(update_fields.values()) + [a_code]
        
        try:
            DatabaseManager.execute_query(query, params)
            return True
        except Exception as e:
            print(f"Error updating administrator: {e}")
            return False
    
    @staticmethod
    def delete(a_code: int) -> bool:
        """
        Delete an administrator record
        
        Args:
            a_code: Administrator's unique code
            
        Returns:
            True if deletion successful, False otherwise
        """
        query = "DELETE FROM Administrator WHERE A_Code = %s"
        try:
            DatabaseManager.execute_query(query, (a_code,))
            return True
        except Exception as e:
            print(f"Error deleting administrator: {e}")
            return False
    
    @staticmethod
    def deactivate(a_code: int) -> bool:
        """
        Deactivate an administrator account
        
        Args:
            a_code: Administrator's unique code
            
        Returns:
            True if deactivation successful, False otherwise
        """
        return AdminRepository.update(a_code, is_active=False)
    
    @staticmethod
    def activate(a_code: int) -> bool:
        """
        Activate an administrator account
        
        Args:
            a_code: Administrator's unique code
            
        Returns:
            True if activation successful, False otherwise
        """
        return AdminRepository.update(a_code, is_active=True)
    
    @staticmethod
    def email_exists(email: str) -> bool:
        """
        Check if an email already exists
        
        Args:
            email: Email to check
            
        Returns:
            True if email exists, False otherwise
        """
        query = "SELECT COUNT(*) as count FROM Administrator WHERE email = %s"
        try:
            result = DatabaseManager.execute_query(query, (email,), fetch_one=True)
            return result['count'] > 0 if result else False
        except Exception as e:
            print(f"Error checking email existence: {e}")
            return False


class FileRepository:
    """Repository for file upload-related database operations"""
    
    @staticmethod
    def create(f_code: int, filename: str, file_path: str, 
               upload_timestamp: datetime, processing_status: str = 'pending') -> Optional[int]:
        """
        Create a new file upload record
        
        Args:
            f_code: Farmer's unique code
            filename: Name of the uploaded file
            file_path: Path where file is stored
            upload_timestamp: Timestamp of upload
            processing_status: Status of processing (default: 'pending')
            
        Returns:
            file_id of the created record, or None if creation failed
        """
        query = """
            INSERT INTO file_uploads 
            (F_code, filename, file_path, upload_timestamp, processing_status)
            VALUES (%s, %s, %s, %s, %s)
        """
        try:
            file_id = DatabaseManager.execute_query(
                query, 
                (f_code, filename, file_path, upload_timestamp, processing_status)
            )
            return file_id
        except Exception as e:
            print(f"Error creating file upload record: {e}")
            return None
    
    @staticmethod
    def find_by_id(file_id: int) -> Optional[Dict[str, Any]]:
        """
        Find a file upload record by file_id
        
        Args:
            file_id: File's unique ID
            
        Returns:
            Dictionary containing file data, or None if not found
        """
        query = """
            SELECT file_id, F_code, filename, file_path, upload_timestamp,
                   detection_result, confidence_score, processing_status
            FROM file_uploads
            WHERE file_id = %s
        """
        try:
            return DatabaseManager.execute_query(query, (file_id,), fetch_one=True)
        except Exception as e:
            print(f"Error finding file by ID: {e}")
            return None
    
    @staticmethod
    def find_by_farmer(f_code: int) -> List[Dict[str, Any]]:
        """
        Get all file uploads for a specific farmer
        
        Args:
            f_code: Farmer's unique code
            
        Returns:
            List of dictionaries containing file data
        """
        query = """
            SELECT file_id, F_code, filename, file_path, upload_timestamp,
                   detection_result, confidence_score, processing_status
            FROM file_uploads
            WHERE F_code = %s
            ORDER BY upload_timestamp DESC
        """
        try:
            return DatabaseManager.execute_query(query, (f_code,), fetch_all=True)
        except Exception as e:
            print(f"Error finding files by farmer: {e}")
            return []
    
    @staticmethod
    def find_all() -> List[Dict[str, Any]]:
        """
        Get all file uploads with farmer information
        
        Returns:
            List of dictionaries containing file and farmer data
        """
        query = """
            SELECT fu.file_id, fu.F_code, fu.filename, fu.file_path, 
                   fu.upload_timestamp, fu.detection_result, fu.confidence_score,
                   fu.processing_status, f.name as farmer_name, f.email as farmer_email
            FROM file_uploads fu
            JOIN farmer f ON fu.F_code = f.F_code
            ORDER BY fu.upload_timestamp DESC
        """
        try:
            return DatabaseManager.execute_query(query, fetch_all=True)
        except Exception as e:
            print(f"Error finding all files: {e}")
            return []
    
    @staticmethod
    def update_result(file_id: int, detection_result: str, 
                     confidence_score: float, processing_status: str = 'completed') -> bool:
        """
        Update file upload with detection results
        
        Args:
            file_id: File's unique ID
            detection_result: Detection result (e.g., 'diseased', 'healthy')
            confidence_score: Confidence score of the prediction
            processing_status: Processing status (default: 'completed')
            
        Returns:
            True if update successful, False otherwise
        """
        query = """
            UPDATE file_uploads
            SET detection_result = %s, confidence_score = %s, processing_status = %s
            WHERE file_id = %s
        """
        try:
            DatabaseManager.execute_query(
                query, 
                (detection_result, confidence_score, processing_status, file_id)
            )
            return True
        except Exception as e:
            print(f"Error updating file result: {e}")
            return False
    
    @staticmethod
    def update_status(file_id: int, processing_status: str) -> bool:
        """
        Update processing status of a file
        
        Args:
            file_id: File's unique ID
            processing_status: New processing status
            
        Returns:
            True if update successful, False otherwise
        """
        query = "UPDATE file_uploads SET processing_status = %s WHERE file_id = %s"
        try:
            DatabaseManager.execute_query(query, (processing_status, file_id))
            return True
        except Exception as e:
            print(f"Error updating file status: {e}")
            return False
    
    @staticmethod
    def delete(file_id: int) -> bool:
        """
        Delete a file upload record
        
        Args:
            file_id: File's unique ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        query = "DELETE FROM file_uploads WHERE file_id = %s"
        try:
            DatabaseManager.execute_query(query, (file_id,))
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    @staticmethod
    def find_by_status(processing_status: str) -> List[Dict[str, Any]]:
        """
        Get all files with a specific processing status
        
        Args:
            processing_status: Status to filter by
            
        Returns:
            List of dictionaries containing file data
        """
        query = """
            SELECT file_id, F_code, filename, file_path, upload_timestamp,
                   detection_result, confidence_score, processing_status
            FROM file_uploads
            WHERE processing_status = %s
            ORDER BY upload_timestamp DESC
        """
        try:
            return DatabaseManager.execute_query(query, (processing_status,), fetch_all=True)
        except Exception as e:
            print(f"Error finding files by status: {e}")
            return []
    
    @staticmethod
    def find_by_result(detection_result: str) -> List[Dict[str, Any]]:
        """
        Get all files with a specific detection result
        
        Args:
            detection_result: Result to filter by (e.g., 'diseased', 'healthy')
            
        Returns:
            List of dictionaries containing file data
        """
        query = """
            SELECT fu.file_id, fu.F_code, fu.filename, fu.file_path, 
                   fu.upload_timestamp, fu.detection_result, fu.confidence_score,
                   fu.processing_status, f.name as farmer_name, f.email as farmer_email
            FROM file_uploads fu
            JOIN farmer f ON fu.F_code = f.F_code
            WHERE fu.detection_result = %s
            ORDER BY fu.upload_timestamp DESC
        """
        try:
            return DatabaseManager.execute_query(query, (detection_result,), fetch_all=True)
        except Exception as e:
            print(f"Error finding files by result: {e}")
            return []


class AdminActionRepository:
    """Repository for admin action logging"""
    
    @staticmethod
    def create(a_code: int, action_type: str, target_user_code: Optional[int] = None,
               target_user_type: Optional[str] = None, action_details: Optional[str] = None) -> Optional[int]:
        """
        Create a new admin action log entry
        
        Args:
            a_code: Administrator's unique code
            action_type: Type of action performed
            target_user_code: Code of the user affected (optional)
            target_user_type: Type of user affected ('farmer' or 'admin', optional)
            action_details: Additional details about the action (optional)
            
        Returns:
            action_id of the created log entry, or None if creation failed
        """
        query = """
            INSERT INTO admin_actions 
            (A_Code, action_type, target_user_code, target_user_type, action_details, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            action_id = DatabaseManager.execute_query(
                query,
                (a_code, action_type, target_user_code, target_user_type, 
                 action_details, datetime.now())
            )
            return action_id
        except Exception as e:
            print(f"Error creating admin action log: {e}")
            return None
    
    @staticmethod
    def find_by_id(action_id: int) -> Optional[Dict[str, Any]]:
        """
        Find an admin action log by action_id
        
        Args:
            action_id: Action's unique ID
            
        Returns:
            Dictionary containing action data, or None if not found
        """
        query = """
            SELECT action_id, A_Code, action_type, target_user_code,
                   target_user_type, action_details, timestamp
            FROM admin_actions
            WHERE action_id = %s
        """
        try:
            return DatabaseManager.execute_query(query, (action_id,), fetch_one=True)
        except Exception as e:
            print(f"Error finding action by ID: {e}")
            return None
    
    @staticmethod
    def find_by_admin(a_code: int) -> List[Dict[str, Any]]:
        """
        Get all actions performed by a specific administrator
        
        Args:
            a_code: Administrator's unique code
            
        Returns:
            List of dictionaries containing action data
        """
        query = """
            SELECT action_id, A_Code, action_type, target_user_code,
                   target_user_type, action_details, timestamp
            FROM admin_actions
            WHERE A_Code = %s
            ORDER BY timestamp DESC
        """
        try:
            return DatabaseManager.execute_query(query, (a_code,), fetch_all=True)
        except Exception as e:
            print(f"Error finding actions by admin: {e}")
            return []
    
    @staticmethod
    def find_all(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all admin actions
        
        Args:
            limit: Maximum number of records to return (optional)
            
        Returns:
            List of dictionaries containing action data
        """
        query = """
            SELECT aa.action_id, aa.A_Code, aa.action_type, aa.target_user_code,
                   aa.target_user_type, aa.action_details, aa.timestamp,
                   a.name as admin_name, a.email as admin_email
            FROM admin_actions aa
            JOIN Administrator a ON aa.A_Code = a.A_Code
            ORDER BY aa.timestamp DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            return DatabaseManager.execute_query(query, fetch_all=True)
        except Exception as e:
            print(f"Error finding all actions: {e}")
            return []
    
    @staticmethod
    def find_by_target(target_user_code: int, target_user_type: str) -> List[Dict[str, Any]]:
        """
        Get all actions performed on a specific user
        
        Args:
            target_user_code: Code of the target user
            target_user_type: Type of the target user ('farmer' or 'admin')
            
        Returns:
            List of dictionaries containing action data
        """
        query = """
            SELECT action_id, A_Code, action_type, target_user_code,
                   target_user_type, action_details, timestamp
            FROM admin_actions
            WHERE target_user_code = %s AND target_user_type = %s
            ORDER BY timestamp DESC
        """
        try:
            return DatabaseManager.execute_query(
                query, 
                (target_user_code, target_user_type), 
                fetch_all=True
            )
        except Exception as e:
            print(f"Error finding actions by target: {e}")
            return []
    
    @staticmethod
    def delete(action_id: int) -> bool:
        """
        Delete an admin action log entry
        
        Args:
            action_id: Action's unique ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        query = "DELETE FROM admin_actions WHERE action_id = %s"
        try:
            DatabaseManager.execute_query(query, (action_id,))
            return True
        except Exception as e:
            print(f"Error deleting action: {e}")
            return False
