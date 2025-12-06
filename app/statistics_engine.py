"""
Statistics Engine for Crop Disease Detection System

This module provides the StatisticsEngine class for calculating
aggregate statistics and tracking recent predictions.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from app.repositories import FileRepository
from app.data_structures.utilities import CircularBuffer


class StatisticsEngine:
    """
    Service class for calculating system statistics.
    
    Handles:
    - Total predictions calculation
    - Disease detection rate computation
    - Average confidence score calculation
    - Recent prediction tracking using circular buffer
    
    Requirements: 9.2, 9.3, 9.4
    """
    
    def __init__(self, recent_activity_size: int = 100):
        """
        Initialize the StatisticsEngine.
        
        Args:
            recent_activity_size: Size of circular buffer for recent predictions
        """
        self.recent_activity_buffer = CircularBuffer(recent_activity_size)
        self._last_update = None
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive system statistics.
        
        Computes:
        - Total number of predictions
        - Disease detection rate (diseased / total)
        - Healthy detection rate (healthy / total)
        - Average confidence score
        - Predictions by status (pending, processing, completed)
        
        Returns:
            dict: Dictionary containing all calculated statistics
            
        Requirements: 9.2, 9.4
        """
        # Get all files from database
        all_files = FileRepository.find_all()
        
        # Initialize counters
        total_predictions = len(all_files)
        diseased_count = 0
        healthy_count = 0
        pending_count = 0
        processing_count = 0
        completed_count = 0
        failed_count = 0
        
        # Lists for confidence calculations
        all_confidence_scores = []
        diseased_confidence_scores = []
        healthy_confidence_scores = []
        
        # Process each file
        for file_data in all_files:
            # Count by detection result
            result = file_data.get('detection_result')
            if result == 'diseased':
                diseased_count += 1
            elif result == 'healthy':
                healthy_count += 1
            
            # Count by processing status
            status = file_data.get('processing_status')
            if status == 'pending':
                pending_count += 1
            elif status == 'processing':
                processing_count += 1
            elif status == 'completed':
                completed_count += 1
            elif status == 'failed':
                failed_count += 1
            
            # Collect confidence scores
            confidence = file_data.get('confidence_score')
            if confidence is not None:
                all_confidence_scores.append(confidence)
                
                if result == 'diseased':
                    diseased_confidence_scores.append(confidence)
                elif result == 'healthy':
                    healthy_confidence_scores.append(confidence)
                
                # Add to recent activity buffer
                self.recent_activity_buffer.insert({
                    'file_id': file_data.get('file_id'),
                    'filename': file_data.get('filename'),
                    'farmer_name': file_data.get('farmer_name'),
                    'result': result,
                    'confidence': confidence,
                    'timestamp': file_data.get('upload_timestamp')
                })
        
        # Calculate rates and averages
        disease_detection_rate = 0.0
        healthy_detection_rate = 0.0
        
        if completed_count > 0:
            disease_detection_rate = (diseased_count / completed_count) * 100
            healthy_detection_rate = (healthy_count / completed_count) * 100
        
        # Calculate average confidence scores
        avg_confidence = 0.0
        avg_diseased_confidence = 0.0
        avg_healthy_confidence = 0.0
        
        if all_confidence_scores:
            avg_confidence = sum(all_confidence_scores) / len(all_confidence_scores)
        
        if diseased_confidence_scores:
            avg_diseased_confidence = sum(diseased_confidence_scores) / len(diseased_confidence_scores)
        
        if healthy_confidence_scores:
            avg_healthy_confidence = sum(healthy_confidence_scores) / len(healthy_confidence_scores)
        
        # Update last update timestamp
        self._last_update = datetime.now()
        
        # Return comprehensive statistics
        return {
            'total_predictions': total_predictions,
            'completed_predictions': completed_count,
            'pending_predictions': pending_count,
            'processing_predictions': processing_count,
            'failed_predictions': failed_count,
            'diseased_count': diseased_count,
            'healthy_count': healthy_count,
            'disease_detection_rate': round(disease_detection_rate, 2),
            'healthy_detection_rate': round(healthy_detection_rate, 2),
            'average_confidence': round(avg_confidence, 4),
            'average_diseased_confidence': round(avg_diseased_confidence, 4),
            'average_healthy_confidence': round(avg_healthy_confidence, 4),
            'last_update': self._last_update
        }
    
    def get_recent_activity(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent prediction activity from circular buffer.
        
        Args:
            limit: Maximum number of recent items to return (optional)
            
        Returns:
            List of recent prediction records
            
        Requirements: 9.3
        """
        if limit:
            return self.recent_activity_buffer.get_recent(limit)
        else:
            return self.recent_activity_buffer.get_all()
    
    def get_detection_trend(self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate detection trends over recent activity.
        
        Analyzes recent predictions to identify trends in:
        - Disease detection frequency
        - Average confidence over time
        - Most active farmers
        
        Args:
            days: Number of days to analyze (uses recent buffer)
            
        Returns:
            dict: Dictionary containing trend analysis
            
        Requirements: 9.5
        """
        recent_items = self.recent_activity_buffer.get_all()
        
        if not recent_items:
            return {
                'trend': 'no_data',
                'recent_diseased_count': 0,
                'recent_healthy_count': 0,
                'recent_avg_confidence': 0.0,
                'most_active_farmers': []
            }
        
        # Count recent results
        recent_diseased = sum(1 for item in recent_items if item.get('result') == 'diseased')
        recent_healthy = sum(1 for item in recent_items if item.get('result') == 'healthy')
        
        # Calculate recent average confidence
        recent_confidences = [item.get('confidence', 0) for item in recent_items if item.get('confidence')]
        recent_avg_confidence = sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0.0
        
        # Find most active farmers
        farmer_counts = {}
        for item in recent_items:
            farmer_name = item.get('farmer_name')
            if farmer_name:
                farmer_counts[farmer_name] = farmer_counts.get(farmer_name, 0) + 1
        
        # Sort farmers by activity
        most_active = sorted(farmer_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Determine trend
        total_recent = len(recent_items)
        disease_rate = (recent_diseased / total_recent * 100) if total_recent > 0 else 0
        
        if disease_rate > 60:
            trend = 'high_disease'
        elif disease_rate > 40:
            trend = 'moderate_disease'
        else:
            trend = 'low_disease'
        
        return {
            'trend': trend,
            'recent_diseased_count': recent_diseased,
            'recent_healthy_count': recent_healthy,
            'recent_avg_confidence': round(recent_avg_confidence, 4),
            'recent_disease_rate': round(disease_rate, 2),
            'most_active_farmers': [{'name': name, 'count': count} for name, count in most_active],
            'total_recent_predictions': total_recent
        }
    
    def get_confidence_distribution(self) -> Dict[str, Any]:
        """
        Calculate confidence score distribution.
        
        Groups predictions by confidence ranges:
        - Very High (>= 0.9)
        - High (0.75 - 0.89)
        - Medium (0.5 - 0.74)
        - Low (< 0.5)
        
        Returns:
            dict: Dictionary containing confidence distribution
        """
        all_files = FileRepository.find_all()
        
        # Initialize counters
        very_high = 0  # >= 0.9
        high = 0       # 0.75 - 0.89
        medium = 0     # 0.5 - 0.74
        low = 0        # < 0.5
        
        for file_data in all_files:
            confidence = file_data.get('confidence_score')
            if confidence is not None:
                if confidence >= 0.9:
                    very_high += 1
                elif confidence >= 0.75:
                    high += 1
                elif confidence >= 0.5:
                    medium += 1
                else:
                    low += 1
        
        total = very_high + high + medium + low
        
        return {
            'very_high_count': very_high,
            'high_count': high,
            'medium_count': medium,
            'low_count': low,
            'very_high_percentage': round((very_high / total * 100) if total > 0 else 0, 2),
            'high_percentage': round((high / total * 100) if total > 0 else 0, 2),
            'medium_percentage': round((medium / total * 100) if total > 0 else 0, 2),
            'low_percentage': round((low / total * 100) if total > 0 else 0, 2),
            'total_with_confidence': total
        }
