"""
UserHashTable implementation for O(1) user lookups
Uses chaining for collision resolution
"""

from typing import Optional, Dict, Any, List


class UserHashTable:
    """
    Hash table implementation for efficient user lookups by email.
    Uses chaining (linked lists) for collision resolution.
    
    Time Complexity:
    - Insert: O(1) average case
    - Lookup: O(1) average case
    - Delete: O(1) average case
    - Exists: O(1) average case
    
    Space Complexity: O(n) where n is the number of users
    """
    
    def __init__(self, size: int = 1000):
        """
        Initialize hash table with specified size.
        
        Args:
            size: Number of buckets in the hash table
        """
        self.size = size
        self.table: List[List[tuple]] = [[] for _ in range(size)]
        self.count = 0
    
    def _hash(self, email: str) -> int:
        """
        Hash function for email-based indexing.
        
        Args:
            email: Email address to hash
            
        Returns:
            Hash value (index in table)
        """
        return hash(email) % self.size
    
    def insert(self, email: str, user_data: Dict[str, Any]) -> None:
        """
        Insert a user into the hash table.
        If email already exists, update the user data.
        
        Args:
            email: User's email address (key)
            user_data: Dictionary containing user information
        """
        index = self._hash(email)
        bucket = self.table[index]
        
        # Check if email already exists and update if so
        for i, (key, value) in enumerate(bucket):
            if key == email:
                bucket[i] = (email, user_data)
                return
        
        # Email doesn't exist, add new entry
        bucket.append((email, user_data))
        self.count += 1
    
    def lookup(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Look up a user by email.
        
        Args:
            email: Email address to search for
            
        Returns:
            User data dictionary if found, None otherwise
        """
        index = self._hash(email)
        bucket = self.table[index]
        
        for key, value in bucket:
            if key == email:
                return value
        
        return None
    
    def exists(self, email: str) -> bool:
        """
        Check if a user with given email exists.
        
        Args:
            email: Email address to check
            
        Returns:
            True if user exists, False otherwise
        """
        return self.lookup(email) is not None
    
    def delete(self, email: str) -> bool:
        """
        Delete a user from the hash table.
        
        Args:
            email: Email address of user to delete
            
        Returns:
            True if user was deleted, False if not found
        """
        index = self._hash(email)
        bucket = self.table[index]
        
        for i, (key, value) in enumerate(bucket):
            if key == email:
                bucket.pop(i)
                self.count -= 1
                return True
        
        return False
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Get all users from the hash table.
        
        Returns:
            List of all user data dictionaries
        """
        users = []
        for bucket in self.table:
            for _, user_data in bucket:
                users.append(user_data)
        return users
    
    def __len__(self) -> int:
        """Return the number of users in the hash table."""
        return self.count
    
    def __contains__(self, email: str) -> bool:
        """Support 'in' operator for checking email existence."""
        return self.exists(email)
