"""
Database connection manager with connection pooling
"""
import mysql.connector
from mysql.connector import pooling, Error
from contextlib import contextmanager
import time
from app.config import Config
from app.error_handlers import ErrorLogger, DatabaseErrorHandler

class DatabaseManager:
    """Manages database connections with pooling and retry logic"""
    
    _pool = None
    
    @classmethod
    def initialize_pool(cls):
        """Initialize the connection pool"""
        if cls._pool is None:
            try:
                cls._pool = pooling.MySQLConnectionPool(
                    pool_name="crop_pool",
                    pool_size=Config.DB_POOL_SIZE,
                    pool_reset_session=True,
                    host=Config.DB_HOST,
                    port=Config.DB_PORT,
                    user=Config.DB_USER,
                    password=Config.DB_PASSWORD,
                    database=Config.DB_NAME
                )
                ErrorLogger.log_info("Database connection pool initialized successfully")
            except Error as e:
                ErrorLogger.log_error(e, {'context': 'Database pool initialization'})
                raise
    
    @classmethod
    def get_connection(cls, max_retries=3):
        """Get a connection from the pool with retry logic"""
        if cls._pool is None:
            cls.initialize_pool()
        
        for attempt in range(max_retries):
            try:
                connection = cls._pool.get_connection()
                return connection
            except Error as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    ErrorLogger.log_warning(
                        f"Connection attempt {attempt + 1} failed. Retrying in {wait_time}s...",
                        {'error': str(e)}
                    )
                    time.sleep(wait_time)
                else:
                    ErrorLogger.log_error(
                        e,
                        {'context': f'Failed to get connection after {max_retries} attempts'}
                    )
                    raise
    
    @classmethod
    @contextmanager
    def get_db_cursor(cls, dictionary=True):
        """Context manager for database operations with automatic commit/rollback"""
        connection = None
        cursor = None
        try:
            connection = cls.get_connection()
            cursor = connection.cursor(dictionary=dictionary)
            yield cursor
            connection.commit()
        except Error as e:
            if connection:
                connection.rollback()
                ErrorLogger.log_error(e, {'context': 'Database operation rolled back'})
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    @classmethod
    def execute_query(cls, query, params=None, fetch_one=False, fetch_all=False):
        """Execute a query and return results with error handling"""
        try:
            with cls.get_db_cursor() as cursor:
                cursor.execute(query, params or ())
                
                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                else:
                    return cursor.lastrowid
        except Error as e:
            ErrorLogger.log_error(
                e,
                {
                    'context': 'Query execution failed',
                    'query': query[:100],  # Log first 100 chars of query
                    'params': str(params)[:100] if params else None
                }
            )
            raise
    
    @classmethod
    def create_tables(cls):
        """Create database tables if they don't exist"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS Administrator (
                A_Code INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                Special_Code VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                INDEX idx_email (email)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS farmer (
                F_code INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                INDEX idx_email (email)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS file_uploads (
                file_id INT PRIMARY KEY AUTO_INCREMENT,
                F_code INT NOT NULL,
                filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                upload_timestamp DATETIME NOT NULL,
                detection_result VARCHAR(50),
                confidence_score FLOAT,
                processing_status VARCHAR(50) DEFAULT 'pending',
                FOREIGN KEY (F_code) REFERENCES farmer(F_code) ON DELETE CASCADE,
                INDEX idx_farmer (F_code),
                INDEX idx_timestamp (upload_timestamp)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS admin_actions (
                action_id INT PRIMARY KEY AUTO_INCREMENT,
                A_Code INT NOT NULL,
                action_type VARCHAR(100) NOT NULL,
                target_user_code INT,
                target_user_type VARCHAR(20),
                action_details TEXT,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (A_Code) REFERENCES Administrator(A_Code),
                INDEX idx_admin (A_Code),
                INDEX idx_timestamp (timestamp)
            )
            """
        ]
        
        try:
            with cls.get_db_cursor() as cursor:
                for table_sql in tables:
                    cursor.execute(table_sql)
            
            ErrorLogger.log_info("Database tables created successfully")
        except Error as e:
            ErrorLogger.log_error(e, {'context': 'Table creation failed'})
            raise
