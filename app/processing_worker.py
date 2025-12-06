"""
Background processing worker for image disease detection

This module provides a worker that processes images from the queue,
invokes the model for inference, and updates results in the database.
"""

import time
import threading
from typing import Optional
from datetime import datetime
from app.model_service import ModelService
from app.file_service import FileService
from app.data_structures.queue import ProcessingQueue, ProcessRequest
from app.repositories import FileRepository


class ProcessingWorker:
    """
    Background worker for processing image detection requests.
    
    Dequeues requests from the processing queue, invokes the model,
    updates the database with results, and handles errors gracefully.
    """
    
    def __init__(self, model_service: ModelService, file_service: FileService,
                 processing_queue: ProcessingQueue):
        """
        Initialize the processing worker.
        
        Args:
            model_service: ModelService instance for inference
            file_service: FileService instance for file management
            processing_queue: ProcessingQueue instance for request management
        """
        self.model_service = model_service
        self.file_service = file_service
        self.processing_queue = processing_queue
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.poll_interval = 2  # seconds between queue checks
    
    def start(self) -> bool:
        """
        Start the background worker thread.
        
        Returns:
            True if worker started successfully, False otherwise
        """
        if self.running:
            print("Worker is already running")
            return False
        
        # Ensure model is loaded
        if not self.model_service.is_loaded():
            try:
                self.model_service.load_model()
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False
        
        # Start worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        print("Processing worker started")
        return True
    
    def stop(self) -> None:
        """Stop the background worker thread."""
        if not self.running:
            print("Worker is not running")
            return
        
        self.running = False
        
        # Wait for thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
        
        print("Processing worker stopped")
    
    def _worker_loop(self) -> None:
        """
        Main worker loop that processes requests from the queue.
        
        Continuously checks the queue for pending requests,
        processes them, and updates results.
        """
        print("Worker loop started")
        
        while self.running:
            try:
                # Check if queue has pending requests
                if self.processing_queue.is_empty():
                    # Sleep before checking again
                    time.sleep(self.poll_interval)
                    continue
                
                # Dequeue next request
                request = self.processing_queue.dequeue()
                
                if request is None:
                    # No request available (race condition)
                    time.sleep(self.poll_interval)
                    continue
                
                # Process the request
                self._process_request(request)
                
            except Exception as e:
                print(f"Error in worker loop: {e}")
                # Continue processing despite errors
                time.sleep(self.poll_interval)
        
        print("Worker loop ended")
    
    def _process_request(self, request: ProcessRequest) -> None:
        """
        Process a single image detection request.
        
        Args:
            request: ProcessRequest object containing request details
        """
        print(f"Processing request {request.request_id} for farmer {request.farmer_code}")
        
        try:
            # Update status to processing
            self.processing_queue.update_status(request.request_id, 'processing')
            
            # Perform inference using ModelService
            result = self.model_service.predict(request.image_path)
            
            # Extract detection result and confidence
            detection_result = result.get('result')  # 'diseased' or 'healthy'
            confidence_score = result.get('confidence')
            
            # Find file_id from database using filename and farmer code
            farmer_files = FileRepository.find_by_farmer(request.farmer_code)
            file_id = None
            
            for file_data in farmer_files:
                if file_data.get('file_path') == request.image_path:
                    file_id = file_data.get('file_id')
                    break
            
            if file_id is None:
                raise ValueError(f"Could not find file_id for path: {request.image_path}")
            
            # Update file_uploads table with results
            success = self.file_service.update_file_result(
                file_id=file_id,
                detection_result=detection_result,
                confidence_score=confidence_score,
                processing_status='completed'
            )
            
            if not success:
                raise RuntimeError("Failed to update file result in database")
            
            # Update request status in queue
            self.processing_queue.update_status(
                request.request_id,
                'completed',
                result=result
            )
            
            print(f"Request {request.request_id} completed: {detection_result} ({confidence_score})")
            
        except Exception as e:
            # Handle processing errors gracefully
            error_message = str(e)
            print(f"Error processing request {request.request_id}: {error_message}")
            
            # Update status to failed
            self.processing_queue.update_status(
                request.request_id,
                'failed',
                result={'error': error_message}
            )
            
            # Try to update database status if we have file_id
            try:
                farmer_files = FileRepository.find_by_farmer(request.farmer_code)
                for file_data in farmer_files:
                    if file_data.get('file_path') == request.image_path:
                        file_id = file_data.get('file_id')
                        FileRepository.update_status(file_id, 'failed')
                        break
            except Exception as db_error:
                print(f"Failed to update database status: {db_error}")
    
    def process_single_request(self, request_id: str) -> bool:
        """
        Process a single request immediately (for testing or manual processing).
        
        Args:
            request_id: ID of the request to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        # Get request from queue
        request = self.processing_queue.get_request(request_id)
        
        if request is None:
            print(f"Request {request_id} not found")
            return False
        
        # Process the request
        try:
            self._process_request(request)
            return True
        except Exception as e:
            print(f"Failed to process request: {e}")
            return False
    
    def get_status(self) -> dict:
        """
        Get worker status information.
        
        Returns:
            Dictionary containing worker status
        """
        return {
            'running': self.running,
            'model_loaded': self.model_service.is_loaded(),
            'queue_size': self.processing_queue.size(),
            'in_progress_count': len(self.processing_queue.get_all_in_progress()),
            'completed_count': len(self.processing_queue.get_all_completed())
        }


# Global worker instance (will be initialized in app factory)
_worker_instance: Optional[ProcessingWorker] = None


def get_worker() -> Optional[ProcessingWorker]:
    """
    Get the global worker instance.
    
    Returns:
        ProcessingWorker instance or None if not initialized
    """
    return _worker_instance


def initialize_worker(model_service: ModelService, file_service: FileService,
                     processing_queue: ProcessingQueue) -> ProcessingWorker:
    """
    Initialize the global worker instance.
    
    Args:
        model_service: ModelService instance
        file_service: FileService instance
        processing_queue: ProcessingQueue instance
        
    Returns:
        Initialized ProcessingWorker instance
    """
    global _worker_instance
    
    if _worker_instance is not None:
        print("Worker already initialized")
        return _worker_instance
    
    _worker_instance = ProcessingWorker(
        model_service=model_service,
        file_service=file_service,
        processing_queue=processing_queue
    )
    
    return _worker_instance


def start_worker() -> bool:
    """
    Start the global worker instance.
    
    Returns:
        True if worker started successfully, False otherwise
    """
    if _worker_instance is None:
        print("Worker not initialized. Call initialize_worker() first.")
        return False
    
    return _worker_instance.start()


def stop_worker() -> None:
    """Stop the global worker instance."""
    if _worker_instance is not None:
        _worker_instance.stop()
