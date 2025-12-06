"""
Custom data structures for efficient operations
This module contains implementations of:
- UserHashTable: O(1) user lookups
- UserBST: Sorted user storage
- ProcessingQueue: FIFO queue for image processing
- FileAVLTree: Timestamp-based file storage
- Trie: Prefix-based filename search
- Sorting algorithms: QuickSort and MergeSort
- Utility structures: Stack, CircularBuffer, Heap
"""

from .hash_table import UserHashTable
from .bst import UserBST, BSTNode
from .queue import ProcessingQueue, ProcessRequest
from .avl_tree import FileAVLTree, AVLNode
from .trie import Trie, TrieNode
from .sorting import (
    quicksort,
    mergesort,
    sort_by_multiple_keys,
    binary_search,
    binary_search_range
)
from .utilities import Stack, CircularBuffer, MinHeap, MaxHeap

__all__ = [
    # Hash Table
    'UserHashTable',
    
    # Binary Search Tree
    'UserBST',
    'BSTNode',
    
    # Queue
    'ProcessingQueue',
    'ProcessRequest',
    
    # AVL Tree
    'FileAVLTree',
    'AVLNode',
    
    # Trie
    'Trie',
    'TrieNode',
    
    # Sorting
    'quicksort',
    'mergesort',
    'sort_by_multiple_keys',
    'binary_search',
    'binary_search_range',
    
    # Utilities
    'Stack',
    'CircularBuffer',
    'MinHeap',
    'MaxHeap',
]
