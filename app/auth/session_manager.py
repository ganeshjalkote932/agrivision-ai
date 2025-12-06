"""
Session management with hash-based O(1) lookups
Handles session creation, validation, and expiration
"""
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from app.config import Config


class Session:
    """Represents a user session."""
    
    def __init__(self, session_id: str, user_code: int, user_type: str, 
                 email: str, name: str, created_at: datetime, expires_at: datetime):
        """
        Initialize a session.
        
        Args:
            session_id: Unique session identifier
            user_code: User's code (F_code or A_Code)
            user_type: Type of user ('farmer' or 'admin')
            email: User's email
            name: User's name
            created_at: Session creation timestamp
            expires_at: Session expiration timestamp
        """
        self.session_id = session_id
        self.user_code = user_code
        self.user_type = user_type
        self.email = email
        self.name = name
        self.created_at = created_at
        self.expires_at = expires_at
        self.last_accessed = created_at
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() > self.expires_at
    
    def refresh(self) -> None:
        """Update last accessed time."""
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            'session_id': self.session_id,
            'user_code': self.user_code,
            'user_type': self.user_type,
            'email': self.email,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat()
        }


class SessionManager:
    """
    Manages user sessions with O(1) lookup using hash map.
    
    Features:
    - Fast session lookup by session_id
    - Automatic session expiration
    - Different timeout for farmers and admins
    - Session cleanup for expired sessions
    
    Time Complexity:
    - create_session: O(1)
    - get_session: O(1)
    - validate_session: O(1)
    - destroy_session: O(1)
    - cleanup_expired_sessions: O(n) where n is number of sessions
    
    Space Complexity: O(n) where n is number of active sessions
    """
    
    def __init__(self):
        """Initialize session manager with empty session map."""
        self.sessions: Dict[str, Session] = {}
    
    def create_session(self, user_code: int, user_type: str, 
                      email: str, name: str) -> str:
        """
        Create a new session for a user.
        
        Args:
            user_code: User's code (F_code or A_Code)
            user_type: Type of user ('farmer' or 'admin')
            email: User's email
            name: User's name
            
        Returns:
            Session ID string
        """
        # Generate secure random session ID
        session_id = secrets.token_urlsafe(32)
        
        # Determine session timeout based on user type
        if user_type == 'admin':
            timeout_seconds = Config.ADMIN_SESSION_TIMEOUT
        else:
            timeout_seconds = Config.SESSION_TIMEOUT
        
        # Create session with expiration time
        created_at = datetime.now()
        expires_at = created_at + timedelta(seconds=timeout_seconds)
        
        session = Session(
            session_id=session_id,
            user_code=user_code,
            user_type=user_type,
            email=email,
            name=name,
            created_at=created_at,
            expires_at=expires_at
        )
        
        # Store session in hash map
        self.sessions[session_id] = session
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by session ID.
        
        Args:
            session_id: Session ID to look up
            
        Returns:
            Session object if found and not expired, None otherwise
        """
        if not session_id:
            return None
        
        session = self.sessions.get(session_id)
        
        if session is None:
            return None
        
        # Check if session has expired
        if session.is_expired():
            # Remove expired session
            self.destroy_session(session_id)
            return None
        
        # Refresh last accessed time
        session.refresh()
        
        return session
    
    def validate_session(self, session_id: str, required_user_type: Optional[str] = None) -> bool:
        """
        Validate a session and optionally check user type.
        
        Args:
            session_id: Session ID to validate
            required_user_type: Required user type ('farmer' or 'admin'), optional
            
        Returns:
            True if session is valid and matches required type, False otherwise
        """
        session = self.get_session(session_id)
        
        if session is None:
            return False
        
        # Check user type if specified
        if required_user_type and session.user_type != required_user_type:
            return False
        
        return True
    
    def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session (logout).
        
        Args:
            session_id: Session ID to destroy
            
        Returns:
            True if session was destroyed, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions from the session map.
        
        Returns:
            Number of sessions cleaned up
        """
        expired_session_ids = []
        
        # Find all expired sessions
        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired_session_ids.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_session_ids:
            del self.sessions[session_id]
        
        return len(expired_session_ids)
    
    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.sessions)
    
    def get_user_session(self, user_code: int, user_type: str) -> Optional[Session]:
        """
        Find an active session for a specific user.
        
        Args:
            user_code: User's code
            user_type: User's type
            
        Returns:
            Session object if found, None otherwise
        """
        for session in self.sessions.values():
            if session.user_code == user_code and session.user_type == user_type:
                if not session.is_expired():
                    return session
        return None


# Global session manager instance
session_manager = SessionManager()
