"""
Utility data structures: Stack, CircularBuffer, and Heap
"""

from typing import Optional, Any, List, Dict
from datetime import datetime


class Stack:
    """
    Stack implementation for admin undo functionality.
    LIFO (Last In First Out) data structure.
    
    Time Complexity:
    - Push: O(1)
    - Pop: O(1)
    - Peek: O(1)
    
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self):
        """Initialize an empty stack."""
        self.items: List[Any] = []
    
    def push(self, item: Any) -> None:
        """
        Push an item onto the stack.
        
        Args:
            item: Item to push
        """
        self.items.append(item)
    
    def pop(self) -> Optional[Any]:
        """
        Pop and return the top item from the stack.
        
        Returns:
            Top item if stack is not empty, None otherwise
        """
        if self.is_empty():
            return None
        return self.items.pop()
    
    def peek(self) -> Optional[Any]:
        """
        Return the top item without removing it.
        
        Returns:
            Top item if stack is not empty, None otherwise
        """
        if self.is_empty():
            return None
        return self.items[-1]
    
    def is_empty(self) -> bool:
        """
        Check if the stack is empty.
        
        Returns:
            True if stack is empty, False otherwise
        """
        return len(self.items) == 0
    
    def size(self) -> int:
        """
        Get the number of items in the stack.
        
        Returns:
            Number of items
        """
        return len(self.items)
    
    def clear(self) -> None:
        """Clear all items from the stack."""
        self.items = []
    
    def __len__(self) -> int:
        """Return the number of items in the stack."""
        return len(self.items)


class CircularBuffer:
    """
    Circular buffer implementation for recent activity tracking.
    Fixed-size buffer that overwrites oldest entries when full.
    
    Time Complexity:
    - Insert: O(1)
    - Get All: O(n)
    
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize a circular buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self.buffer: List[Optional[Any]] = [None] * capacity
        self.head = 0  # Points to next write position
        self.size = 0  # Current number of items
    
    def insert(self, item: Any) -> None:
        """
        Insert an item into the buffer.
        Overwrites oldest item if buffer is full.
        
        Args:
            item: Item to insert
        """
        self.buffer[self.head] = item
        self.head = (self.head + 1) % self.capacity
        
        if self.size < self.capacity:
            self.size += 1
    
    def get_all(self) -> List[Any]:
        """
        Get all items in the buffer in insertion order (oldest to newest).
        
        Returns:
            List of items
        """
        if self.size == 0:
            return []
        
        if self.size < self.capacity:
            # Buffer not full yet, return items from start
            return [item for item in self.buffer[:self.size] if item is not None]
        else:
            # Buffer is full, return items starting from oldest
            return [self.buffer[(self.head + i) % self.capacity] 
                   for i in range(self.capacity)]
    
    def get_recent(self, n: int) -> List[Any]:
        """
        Get the n most recent items.
        
        Args:
            n: Number of recent items to retrieve
            
        Returns:
            List of n most recent items (or all items if n > size)
        """
        n = min(n, self.size)
        if n == 0:
            return []
        
        result = []
        for i in range(n):
            index = (self.head - 1 - i) % self.capacity
            if self.buffer[index] is not None:
                result.append(self.buffer[index])
        
        return result
    
    def is_full(self) -> bool:
        """
        Check if the buffer is full.
        
        Returns:
            True if buffer is full, False otherwise
        """
        return self.size == self.capacity
    
    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.
        
        Returns:
            True if buffer is empty, False otherwise
        """
        return self.size == 0
    
    def clear(self) -> None:
        """Clear all items from the buffer."""
        self.buffer = [None] * self.capacity
        self.head = 0
        self.size = 0
    
    def __len__(self) -> int:
        """Return the number of items in the buffer."""
        return self.size


class MinHeap:
    """
    Min Heap implementation for priority-based operations.
    
    Time Complexity:
    - Insert: O(log n)
    - Extract Min: O(log n)
    - Peek Min: O(1)
    
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self):
        """Initialize an empty min heap."""
        self.heap: List[tuple] = []  # List of (priority, data) tuples
    
    def insert(self, priority: Any, data: Any) -> None:
        """
        Insert an item with given priority.
        
        Args:
            priority: Priority value (lower = higher priority)
            data: Associated data
        """
        self.heap.append((priority, data))
        self._heapify_up(len(self.heap) - 1)
    
    def extract_min(self) -> Optional[tuple]:
        """
        Remove and return the item with minimum priority.
        
        Returns:
            Tuple of (priority, data) if heap is not empty, None otherwise
        """
        if self.is_empty():
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        # Swap root with last element
        min_item = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return min_item
    
    def peek_min(self) -> Optional[tuple]:
        """
        Return the item with minimum priority without removing it.
        
        Returns:
            Tuple of (priority, data) if heap is not empty, None otherwise
        """
        if self.is_empty():
            return None
        return self.heap[0]
    
    def _heapify_up(self, index: int) -> None:
        """
        Restore heap property by moving element up.
        
        Args:
            index: Index of element to heapify up
        """
        parent_index = (index - 1) // 2
        
        if index > 0 and self.heap[index][0] < self.heap[parent_index][0]:
            # Swap with parent
            self.heap[index], self.heap[parent_index] = \
                self.heap[parent_index], self.heap[index]
            self._heapify_up(parent_index)
    
    def _heapify_down(self, index: int) -> None:
        """
        Restore heap property by moving element down.
        
        Args:
            index: Index of element to heapify down
        """
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2
        
        if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
            smallest = left
        
        if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
            smallest = right
        
        if smallest != index:
            # Swap with smallest child
            self.heap[index], self.heap[smallest] = \
                self.heap[smallest], self.heap[index]
            self._heapify_down(smallest)
    
    def is_empty(self) -> bool:
        """
        Check if the heap is empty.
        
        Returns:
            True if heap is empty, False otherwise
        """
        return len(self.heap) == 0
    
    def size(self) -> int:
        """
        Get the number of items in the heap.
        
        Returns:
            Number of items
        """
        return len(self.heap)
    
    def __len__(self) -> int:
        """Return the number of items in the heap."""
        return len(self.heap)


class MaxHeap:
    """
    Max Heap implementation for priority-based operations.
    
    Time Complexity:
    - Insert: O(log n)
    - Extract Max: O(log n)
    - Peek Max: O(1)
    
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self):
        """Initialize an empty max heap."""
        self.heap: List[tuple] = []  # List of (priority, data) tuples
    
    def insert(self, priority: Any, data: Any) -> None:
        """
        Insert an item with given priority.
        
        Args:
            priority: Priority value (higher = higher priority)
            data: Associated data
        """
        self.heap.append((priority, data))
        self._heapify_up(len(self.heap) - 1)
    
    def extract_max(self) -> Optional[tuple]:
        """
        Remove and return the item with maximum priority.
        
        Returns:
            Tuple of (priority, data) if heap is not empty, None otherwise
        """
        if self.is_empty():
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        # Swap root with last element
        max_item = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return max_item
    
    def peek_max(self) -> Optional[tuple]:
        """
        Return the item with maximum priority without removing it.
        
        Returns:
            Tuple of (priority, data) if heap is not empty, None otherwise
        """
        if self.is_empty():
            return None
        return self.heap[0]
    
    def _heapify_up(self, index: int) -> None:
        """
        Restore heap property by moving element up.
        
        Args:
            index: Index of element to heapify up
        """
        parent_index = (index - 1) // 2
        
        if index > 0 and self.heap[index][0] > self.heap[parent_index][0]:
            # Swap with parent
            self.heap[index], self.heap[parent_index] = \
                self.heap[parent_index], self.heap[index]
            self._heapify_up(parent_index)
    
    def _heapify_down(self, index: int) -> None:
        """
        Restore heap property by moving element down.
        
        Args:
            index: Index of element to heapify down
        """
        largest = index
        left = 2 * index + 1
        right = 2 * index + 2
        
        if left < len(self.heap) and self.heap[left][0] > self.heap[largest][0]:
            largest = left
        
        if right < len(self.heap) and self.heap[right][0] > self.heap[largest][0]:
            largest = right
        
        if largest != index:
            # Swap with largest child
            self.heap[index], self.heap[largest] = \
                self.heap[largest], self.heap[index]
            self._heapify_down(largest)
    
    def is_empty(self) -> bool:
        """
        Check if the heap is empty.
        
        Returns:
            True if heap is empty, False otherwise
        """
        return len(self.heap) == 0
    
    def size(self) -> int:
        """
        Get the number of items in the heap.
        
        Returns:
            Number of items
        """
        return len(self.heap)
    
    def __len__(self) -> int:
        """Return the number of items in the heap."""
        return len(self.heap)
