"""
Sorting algorithms implementation: QuickSort and MergeSort
"""

from typing import List, Dict, Any, Callable, Optional


def quicksort(arr: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    QuickSort implementation for sorting dictionaries by a specified key.
    Used for multi-column user sorting.
    
    Time Complexity: O(n log n) average case, O(n²) worst case
    Space Complexity: O(log n) for recursion stack
    
    Args:
        arr: List of dictionaries to sort
        key: Dictionary key to sort by
        reverse: If True, sort in descending order
        
    Returns:
        Sorted list of dictionaries
    """
    if len(arr) <= 1:
        return arr
    
    return _quicksort_recursive(arr.copy(), key, reverse, 0, len(arr) - 1)


def _quicksort_recursive(arr: List[Dict[str, Any]], key: str, reverse: bool, 
                        low: int, high: int) -> List[Dict[str, Any]]:
    """
    Recursive helper for QuickSort.
    
    Args:
        arr: Array to sort (modified in place)
        key: Dictionary key to sort by
        reverse: Sort order
        low: Starting index
        high: Ending index
        
    Returns:
        Sorted array
    """
    if low < high:
        # Partition the array and get pivot index
        pivot_index = _partition(arr, key, reverse, low, high)
        
        # Recursively sort elements before and after partition
        _quicksort_recursive(arr, key, reverse, low, pivot_index - 1)
        _quicksort_recursive(arr, key, reverse, pivot_index + 1, high)
    
    return arr


def _partition(arr: List[Dict[str, Any]], key: str, reverse: bool, 
              low: int, high: int) -> int:
    """
    Partition function for QuickSort using last element as pivot.
    
    Args:
        arr: Array to partition
        key: Dictionary key to compare
        reverse: Sort order
        low: Starting index
        high: Ending index
        
    Returns:
        Final position of pivot
    """
    pivot = arr[high][key]
    i = low - 1
    
    for j in range(low, high):
        # Compare based on sort order
        if reverse:
            condition = arr[j][key] > pivot
        else:
            condition = arr[j][key] < pivot
        
        if condition:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def mergesort(arr: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    MergeSort implementation for sorting dictionaries by a specified key.
    Used for file history sorting (stable sort).
    
    Time Complexity: O(n log n) in all cases
    Space Complexity: O(n) for temporary arrays
    
    Args:
        arr: List of dictionaries to sort
        key: Dictionary key to sort by
        reverse: If True, sort in descending order
        
    Returns:
        Sorted list of dictionaries
    """
    if len(arr) <= 1:
        return arr
    
    return _mergesort_recursive(arr, key, reverse)


def _mergesort_recursive(arr: List[Dict[str, Any]], key: str, reverse: bool) -> List[Dict[str, Any]]:
    """
    Recursive helper for MergeSort.
    
    Args:
        arr: Array to sort
        key: Dictionary key to sort by
        reverse: Sort order
        
    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return arr
    
    # Divide the array into two halves
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # Recursively sort both halves
    left = _mergesort_recursive(left, key, reverse)
    right = _mergesort_recursive(right, key, reverse)
    
    # Merge the sorted halves
    return _merge(left, right, key, reverse)


def _merge(left: List[Dict[str, Any]], right: List[Dict[str, Any]], 
          key: str, reverse: bool) -> List[Dict[str, Any]]:
    """
    Merge two sorted arrays.
    
    Args:
        left: Left sorted array
        right: Right sorted array
        key: Dictionary key to compare
        reverse: Sort order
        
    Returns:
        Merged sorted array
    """
    result = []
    i = j = 0
    
    # Merge while both arrays have elements
    while i < len(left) and j < len(right):
        if reverse:
            condition = left[i][key] > right[j][key]
        else:
            condition = left[i][key] < right[j][key]
        
        if condition:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements from left array
    while i < len(left):
        result.append(left[i])
        i += 1
    
    # Add remaining elements from right array
    while j < len(right):
        result.append(right[j])
        j += 1
    
    return result


def sort_by_multiple_keys(arr: List[Dict[str, Any]], keys: List[str], 
                         reverse: bool = False) -> List[Dict[str, Any]]:
    """
    Sort by multiple keys in order of priority.
    Uses QuickSort with custom comparison function.
    
    Args:
        arr: List of dictionaries to sort
        keys: List of keys in order of priority
        reverse: If True, sort in descending order
        
    Returns:
        Sorted list of dictionaries
    """
    if len(arr) <= 1:
        return arr
    
    # Create a copy to avoid modifying original
    sorted_arr = arr.copy()
    
    # Sort by each key in reverse order of priority
    # This ensures primary key has final say
    for key in reversed(keys):
        sorted_arr = mergesort(sorted_arr, key, reverse)
    
    return sorted_arr


def binary_search(arr: List[Dict[str, Any]], key: str, value: Any) -> Optional[int]:
    """
    Binary search for finding an element in a sorted array.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Args:
        arr: Sorted list of dictionaries
        key: Dictionary key to search by
        value: Value to search for
        
    Returns:
        Index of element if found, None otherwise
    """
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_value = arr[mid][key]
        
        if mid_value == value:
            return mid
        elif mid_value < value:
            left = mid + 1
        else:
            right = mid - 1
    
    return None


def binary_search_range(arr: List[Dict[str, Any]], key: str, 
                       start_value: Any, end_value: Any) -> List[Dict[str, Any]]:
    """
    Find all elements in a sorted array within a given range.
    
    Args:
        arr: Sorted list of dictionaries
        key: Dictionary key to search by
        start_value: Start of range (inclusive)
        end_value: End of range (inclusive)
        
    Returns:
        List of elements within range
    """
    result = []
    
    for item in arr:
        if start_value <= item[key] <= end_value:
            result.append(item)
        elif item[key] > end_value:
            break  # Array is sorted, no need to continue
    
    return result
