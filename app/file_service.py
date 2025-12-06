"""
FileService for managing file uploads and processing

This module provides the FileService class for handling file metadata,
retrieval, sorting, filtering, and searching operations using various
data structures for optimal performance.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from app.repositories import FileRepository
from app.data_structures.avl_tree import FileAVLTree
from app.data_structures.trie import Trie
from app.data_structures.sorting import mergesort
from app.data_structures.hash_table import UserHashTable


class FileService:
    """
    Service class for managing file uploads and processing.
    
    Uses multiple data structures for efficient operations:
    - AVL Tree for timestamp-based queries
    - Trie for prefix-based filename search
    - Hash Table for status filtering
    - Merge Sort for sorting by timestamp
    """
    
    def __init__(self):
        """Initialize the FileService with data structures."""
        self.file_tree = FileAVLTree()
        self.filename_trie = Trie()
        self.status_filter = UserHashTable(size=100)
        self._initialized = False
    
    def _initialize_structures(self) -> None:
        """
        Initialize data structures with existing file data from database.
        This should be called once when the service is first used.
        """
        if self._initialized:
            return
        
        # Load all files from database
        all_files = FileRepository.find_all()
        
        # Populate data structures
        for file_data in all_files:
            self._add_to_structures(file_data)
        
        self._initialized = True
    
    def _add_to_structures(self, file_data: Dict[str, Any]) -> None:
        """
        Add file data to all internal data structures.
        
        Args:
            file_data: Dictionary containing file information
        """
        # Add to AVL tree (indexed by timestamp)
        if 'upload_timestamp' in file_data and file_data['upload_timestamp']:
            timestamp = file_data['upload_timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            self.file_tree.insert(timestamp, file_data)
        
        # Add to Trie (indexed by filename)
        if 'filename' in file_data and file_data['filename']:
            # Store file_id with filename for lookup
            trie_data = {
                'file_id': file_data.get('file_id'),
                'F_code': file_data.get('F_code')
            }
            self.filename_trie.insert(file_data['filename'], trie_data)
        
        # Add to status filter hash table
        if 'detection_result' in file_data and file_data['detection_result']:
            status = file_data['detection_result']
            # Get existing list for this status or create new one
            existing = self.status_filter.lookup(status)
            if existing is None:
                file_list = [file_data]
            else:
                file_list = existing
                file_list.append(file_data)
            self.status_filter.insert(status, file_list)
    
    def save_file_metadata(self, f_code: int, filename: str, file_path: str,
                          upload_timestamp: datetime, processing_status: str = 'pending') -> Optional[int]:
        """
        Save file metadata to database and update data structures.
        
        Args:
            f_code: Farmer's unique code
            filename: Name of the uploaded file
            file_path: Path where file is stored
            upload_timestamp: Timestamp of upload
            processing_status: Status of processing (default: 'pending')
            
        Returns:
            file_id of the created record, or None if creation failed
        """
        # Ensure structures are initialized
        self._initialize_structures()
        
        # Save to database
        file_id = FileRepository.create(
            f_code=f_code,
            filename=filename,
            file_path=file_path,
            upload_timestamp=upload_timestamp,
            processing_status=processing_status
        )
        
        if file_id is None:
            return None
        
        # Create file data dictionary
        file_data = {
            'file_id': file_id,
            'F_code': f_code,
            'filename': filename,
            'file_path': file_path,
            'upload_timestamp': upload_timestamp,
            'processing_status': processing_status,
            'detection_result': None,
            'confidence_score': None
        }
        
        # Add to data structures
        self._add_to_structures(file_data)
        
        return file_id
    
    def update_file_result(self, file_id: int, detection_result: str,
                          confidence_score: float, processing_status: str = 'completed') -> bool:
        """
        Update file with detection results.
        
        Args:
            file_id: File's unique ID
            detection_result: Detection result (e.g., 'diseased', 'healthy')
            confidence_score: Confidence score of the prediction
            processing_status: Processing status (default: 'completed')
            
        Returns:
            True if update successful, False otherwise
        """
        # Update in database
        success = FileRepository.update_result(
            file_id=file_id,
            detection_result=detection_result,
            confidence_score=confidence_score,
            processing_status=processing_status
        )
        
        if success:
            # Reload structures to reflect changes
            # In a production system, you might want to update structures incrementally
            self._initialized = False
            self._initialize_structures()
        
        return success
    
    def get_farmer_files(self, f_code: int, sort_by: str = 'upload_timestamp',
                        reverse: bool = True) -> List[Dict[str, Any]]:
        """
        Get all files for a specific farmer with merge sort by timestamp.
        
        Args:
            f_code: Farmer's unique code
            sort_by: Field to sort by (default: 'upload_timestamp')
            reverse: If True, sort in descending order (default: True for most recent first)
            
        Returns:
            List of file dictionaries sorted by specified field
        """
        # Ensure structures are initialized
        self._initialize_structures()
        
        # Get files from database
        files = FileRepository.find_by_farmer(f_code)
        
        if not files:
            return []
        
        # Convert datetime objects to comparable format for sorting
        for file_data in files:
            if 'upload_timestamp' in file_data and isinstance(file_data['upload_timestamp'], datetime):
                # Keep as datetime for sorting
                pass
            elif 'upload_timestamp' in file_data and isinstance(file_data['upload_timestamp'], str):
                file_data['upload_timestamp'] = datetime.fromisoformat(file_data['upload_timestamp'])
        
        # Sort using merge sort (stable sort)
        sorted_files = mergesort(files, key=sort_by, reverse=reverse)
        
        return sorted_files
    
    def get_all_files(self, sort_by: str = 'upload_timestamp',
                     reverse: bool = True) -> List[Dict[str, Any]]:
        """
        Get all files with farmer information for admin access.
        
        Args:
            sort_by: Field to sort by (default: 'upload_timestamp')
            reverse: If True, sort in descending order (default: True)
            
        Returns:
            List of file dictionaries with farmer info, sorted by specified field
        """
        # Ensure structures are initialized
        self._initialize_structures()
        
        # Get all files from database (includes farmer info via JOIN)
        files = FileRepository.find_all()
        
        if not files:
            return []
        
        # Convert datetime objects for sorting
        for file_data in files:
            if 'upload_timestamp' in file_data and isinstance(file_data['upload_timestamp'], datetime):
                pass
            elif 'upload_timestamp' in file_data and isinstance(file_data['upload_timestamp'], str):
                file_data['upload_timestamp'] = datetime.fromisoformat(file_data['upload_timestamp'])
        
        # Sort using merge sort
        sorted_files = mergesort(files, key=sort_by, reverse=reverse)
        
        return sorted_files
    
    def filter_files(self, status: str) -> List[Dict[str, Any]]:
        """
        Filter files by detection result status using hash table.
        
        Args:
            status: Detection result to filter by (e.g., 'diseased', 'healthy')
            
        Returns:
            List of file dictionaries with the specified status
        """
        # Ensure structures are initialized
        self._initialize_structures()
        
        # Try to get from hash table first (O(1) lookup)
        cached_files = self.status_filter.lookup(status)
        
        if cached_files is not None:
            return cached_files
        
        # If not in cache, query database
        files = FileRepository.find_by_result(status)
        
        # Update cache
        if files:
            self.status_filter.insert(status, files)
        
        return files
    
    def search_files(self, prefix: str, f_code: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search files by filename prefix using trie.
        
        Args:
            prefix: Filename prefix to search for
            f_code: Optional farmer code to filter results (None for all farmers)
            
        Returns:
            List of file dictionaries matching the prefix
        """
        # Ensure structures are initialized
        self._initialize_structures()
        
        # Search using trie (O(m + k) where m is prefix length, k is results)
        trie_results = self.filename_trie.starts_with(prefix)
        
        if not trie_results:
            return []
        
        # Get full file data from database for matching file_ids
        matching_files = []
        for trie_data in trie_results:
            file_id = trie_data.get('file_id')
            if file_id:
                file_data = FileRepository.find_by_id(file_id)
                if file_data:
                    # Filter by farmer if specified
                    if f_code is None or file_data.get('F_code') == f_code:
                        matching_files.append(file_data)
        
        return matching_files
    
    def get_file_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific file by its ID.
        
        Args:
            file_id: File's unique ID
            
        Returns:
            File dictionary or None if not found
        """
        return FileRepository.find_by_id(file_id)
    
    def delete_file(self, file_id: int) -> bool:
        """
        Delete a file record.
        
        Args:
            file_id: File's unique ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        success = FileRepository.delete(file_id)
        
        if success:
            # Reload structures to reflect changes
            self._initialized = False
            self._initialize_structures()
        
        return success
    
    def get_files_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get files within a specific date range using AVL tree range query.
        
        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            
        Returns:
            List of file dictionaries within the date range
        """
        # Ensure structures are initialized
        self._initialize_structures()
        
        # Use AVL tree range query (O(log n + k) where k is results)
        files = self.file_tree.range_query(start_date, end_date)
        
        return files
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get file statistics.
        
        Returns:
            Dictionary containing statistics about files
        """
        # Ensure structures are initialized
        self._initialize_structures()
        
        all_files = FileRepository.find_all()
        
        total_files = len(all_files)
        diseased_count = sum(1 for f in all_files if f.get('detection_result') == 'diseased')
        healthy_count = sum(1 for f in all_files if f.get('detection_result') == 'healthy')
        pending_count = sum(1 for f in all_files if f.get('processing_status') == 'pending')
        processing_count = sum(1 for f in all_files if f.get('processing_status') == 'processing')
        completed_count = sum(1 for f in all_files if f.get('processing_status') == 'completed')
        
        # Calculate average confidence for completed files
        completed_files = [f for f in all_files if f.get('confidence_score') is not None]
        avg_confidence = 0.0
        if completed_files:
            avg_confidence = sum(f['confidence_score'] for f in completed_files) / len(completed_files)
        
        return {
            'total_files': total_files,
            'diseased_count': diseased_count,
            'healthy_count': healthy_count,
            'pending_count': pending_count,
            'processing_count': processing_count,
            'completed_count': completed_count,
            'average_confidence': round(avg_confidence, 4),
            'disease_rate': round(diseased_count / total_files * 100, 2) if total_files > 0 else 0.0
        }
