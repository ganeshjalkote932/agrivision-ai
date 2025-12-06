"""
ProcessingQueue implementation for image processing requests
Uses Python deque for O(1) enqueue/dequeue operations
"""

from collections import deque
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class ProcessRequest:
    """Represents an image processing request."""
    
    def __init__(self, farmer_code: int, image_path: str, filename: str):
        """
        Initialize a processing request.
        
        Args:
            farmer_code: F_code of the farmer who uploaded the image
            image_path: Path to the uploaded image file
            filename: Original filename of the uploaded image
        """
        self.request_id = str(uuid.uuid4())
        self.farmer_code = farmer_code
        self.image_path = image_path
        self.filename = filename
        self.timestamp = datetime.now()
        self.status = 'pending'  # 'pending', 'processing', 'completed', 'failed'
        self.result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary representation."""
        return {
            'request_id': self.request_id,
            'farmer_code': self.farmer_code,
            'image_path': self.image_path,
            'filename': self.filename,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status,
            'result': self.result
        }


class ProcessingQueue:
    """
    FIFO queue for managing image processing requests.
    Uses deque for O(1) enqueue and dequeue operations.
    Includes hash table for tracking in-progress requests.
    
    Time Complexity:
    - Enqueue: O(1)
    - Dequeue: O(1)
    - Get Status: O(1)
    - Update Status: O(1)
    
    Space Complexity: O(n) where n is the number of requests
    """
    
    def __init__(self):
        """Initialize an empty processing queue."""
        self.queue = deque()
        self.in_progress: Dict[str, ProcessRequest] = {}  # Hash table for tracking
        self.completed: Dict[str, ProcessRequest] = {}  # Hash table for completed requests
    
    def enqueue(self, farmer_code: int, image_path: str, filename: str) -> str:
        """
        Add a processing request to the queue.
        
        Args:
            farmer_code: F_code of the farmer
            image_path: Path to the image file
            filename: Original filename
            
        Returns:
            request_id: Unique identifier for tracking the request
        """
        request = ProcessRequest(farmer_code, image_path, filename)
        self.queue.append(request)
        return request.request_id
    
    def dequeue(self) -> Optional[ProcessRequest]:
        """
        Remove and return the next request from the queue.
        Marks the request as 'processing' and adds to in_progress tracker.
        
        Returns:
            ProcessRequest if queue is not empty, None otherwise
        """
        if not self.queue:
            return None
        
        request = self.queue.popleft()
        request.status = 'processing'
        self.in_progress[request.request_id] = request
        return request
    
    def get_status(self, request_id: str) -> Optional[str]:
        """
        Get the status of a request by its ID.
        
        Args:
            request_id: Unique identifier of the request
            
        Returns:
            Status string ('pending', 'processing', 'completed', 'failed') or None if not found
        """
        # Check in-progress requests
        if request_id in self.in_progress:
            return self.in_progress[request_id].status
        
        # Check completed requests
        if request_id in self.completed:
            return self.completed[request_id].status
        
        # Check pending queue
        for request in self.queue:
            if request.request_id == request_id:
                return request.status
        
        return None
    
    def get_request(self, request_id: str) -> Optional[ProcessRequest]:
        """
        Get the full request object by its ID.
        
        Args:
            request_id: Unique identifier of the request
            
        Returns:
            ProcessRequest if found, None otherwise
        """
        # Check in-progress requests
        if request_id in self.in_progress:
            return self.in_progress[request_id]
        
        # Check completed requests
        if request_id in self.completed:
            return self.completed[request_id]
        
        # Check pending queue
        for request in self.queue:
            if request.request_id == request_id:
                return request
        
        return None
    
    def update_status(self, request_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a request.
        
        Args:
            request_id: Unique identifier of the request
            status: New status ('processing', 'completed', 'failed')
            result: Optional result data (for completed requests)
            
        Returns:
            True if update was successful, False if request not found
        """
        if request_id in self.in_progress:
            request = self.in_progress[request_id]
            request.status = status
            
            if result is not None:
                request.result = result
            
            # Move to completed if status is 'completed' or 'failed'
            if status in ['completed', 'failed']:
                self.completed[request_id] = request
                del self.in_progress[request_id]
            
            return True
        
        return False
    
    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        
        Returns:
            True if queue has no pending requests, False otherwise
        """
        return len(self.queue) == 0
    
    def size(self) -> int:
        """
        Get the number of pending requests in the queue.
        
        Returns:
            Number of pending requests
        """
        return len(self.queue)
    
    def is_in_progress(self, request_id: str) -> bool:
        """
        Check if a request is currently being processed.
        
        Args:
            request_id: Unique identifier of the request
            
        Returns:
            True if request is in progress, False otherwise
        """
        return request_id in self.in_progress
    
    def get_all_in_progress(self) -> list[ProcessRequest]:
        """
        Get all requests currently being processed.
        
        Returns:
            List of ProcessRequest objects
        """
        return list(self.in_progress.values())
    
    def get_all_completed(self) -> list[ProcessRequest]:
        """
        Get all completed requests.
        
        Returns:
            List of ProcessRequest objects
        """
        return list(self.completed.values())
    
    def __len__(self) -> int:
        """Return the total number of requests (pending + in-progress)."""
        return len(self.queue) + len(self.in_progress)
