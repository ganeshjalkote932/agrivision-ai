"""
FileAVLTree implementation for timestamp-based file storage
Self-balancing binary search tree with automatic rotations
"""

from typing import Optional, Dict, Any, List
from datetime import datetime


class AVLNode:
    """Node in an AVL tree."""
    
    def __init__(self, timestamp: datetime, file_data: Dict[str, Any]):
        """
        Initialize an AVL node.
        
        Args:
            timestamp: Upload timestamp used as key
            file_data: Dictionary containing file information
        """
        self.timestamp = timestamp
        self.file_data = file_data
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height = 1


class FileAVLTree:
    """
    Self-balancing AVL tree for timestamp-based file storage.
    Maintains files sorted by upload timestamp with automatic balancing.
    
    Time Complexity:
    - Insert: O(log n)
    - Range Query: O(log n + k) where k is number of results
    - Inorder Traversal: O(n)
    - Delete: O(log n)
    
    Space Complexity: O(n) where n is the number of files
    """
    
    def __init__(self):
        """Initialize an empty AVL tree."""
        self.root: Optional[AVLNode] = None
        self.count = 0
    
    def _get_height(self, node: Optional[AVLNode]) -> int:
        """
        Get the height of a node.
        
        Args:
            node: Node to get height of
            
        Returns:
            Height of the node (0 if None)
        """
        if node is None:
            return 0
        return node.height
    
    def _get_balance(self, node: Optional[AVLNode]) -> int:
        """
        Get the balance factor of a node.
        
        Args:
            node: Node to get balance factor of
            
        Returns:
            Balance factor (left height - right height)
        """
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _update_height(self, node: AVLNode) -> None:
        """
        Update the height of a node based on its children.
        
        Args:
            node: Node to update height for
        """
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
    
    def _rotate_right(self, z: AVLNode) -> AVLNode:
        """
        Perform right rotation.
        
        Args:
            z: Root of subtree to rotate
            
        Returns:
            New root after rotation
        """
        y = z.left
        T3 = y.right
        
        # Perform rotation
        y.right = z
        z.left = T3
        
        # Update heights
        self._update_height(z)
        self._update_height(y)
        
        return y
    
    def _rotate_left(self, z: AVLNode) -> AVLNode:
        """
        Perform left rotation.
        
        Args:
            z: Root of subtree to rotate
            
        Returns:
            New root after rotation
        """
        y = z.right
        T2 = y.left
        
        # Perform rotation
        y.left = z
        z.right = T2
        
        # Update heights
        self._update_height(z)
        self._update_height(y)
        
        return y
    
    def insert(self, timestamp: datetime, file_data: Dict[str, Any]) -> None:
        """
        Insert a file into the AVL tree with automatic balancing.
        
        Args:
            timestamp: Upload timestamp (key)
            file_data: Dictionary containing file information
        """
        self.root = self._insert_recursive(self.root, timestamp, file_data)
    
    def _insert_recursive(self, node: Optional[AVLNode], timestamp: datetime, 
                         file_data: Dict[str, Any]) -> AVLNode:
        """
        Recursive helper for insert with balancing.
        
        Args:
            node: Current node in traversal
            timestamp: Timestamp to insert
            file_data: File data to insert
            
        Returns:
            Root of balanced subtree
        """
        # Standard BST insertion
        if node is None:
            self.count += 1
            return AVLNode(timestamp, file_data)
        
        if timestamp < node.timestamp:
            node.left = self._insert_recursive(node.left, timestamp, file_data)
        elif timestamp > node.timestamp:
            node.right = self._insert_recursive(node.right, timestamp, file_data)
        else:
            # Duplicate timestamp - update data
            node.file_data = file_data
            return node
        
        # Update height
        self._update_height(node)
        
        # Get balance factor
        balance = self._get_balance(node)
        
        # Left-Left case
        if balance > 1 and timestamp < node.left.timestamp:
            return self._rotate_right(node)
        
        # Right-Right case
        if balance < -1 and timestamp > node.right.timestamp:
            return self._rotate_left(node)
        
        # Left-Right case
        if balance > 1 and timestamp > node.left.timestamp:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right-Left case
        if balance < -1 and timestamp < node.right.timestamp:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def range_query(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve all files with timestamps in the given range [start, end].
        
        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            
        Returns:
            List of file data dictionaries in sorted order
        """
        result = []
        self._range_query_recursive(self.root, start, end, result)
        return result
    
    def _range_query_recursive(self, node: Optional[AVLNode], start: datetime, 
                               end: datetime, result: List[Dict[str, Any]]) -> None:
        """
        Recursive helper for range query.
        
        Args:
            node: Current node in traversal
            start: Start of time range
            end: End of time range
            result: List to accumulate results
        """
        if node is None:
            return
        
        # If current node is greater than start, search left subtree
        if node.timestamp > start:
            self._range_query_recursive(node.left, start, end, result)
        
        # If current node is in range, add it
        if start <= node.timestamp <= end:
            result.append(node.file_data)
        
        # If current node is less than end, search right subtree
        if node.timestamp < end:
            self._range_query_recursive(node.right, start, end, result)
    
    def inorder_traversal(self) -> List[Dict[str, Any]]:
        """
        Perform inorder traversal to get files in sorted order by timestamp.
        
        Returns:
            List of file data dictionaries in sorted order
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: Optional[AVLNode], result: List[Dict[str, Any]]) -> None:
        """
        Recursive helper for inorder traversal.
        
        Args:
            node: Current node in traversal
            result: List to accumulate results
        """
        if node is not None:
            self._inorder_recursive(node.left, result)
            result.append(node.file_data)
            self._inorder_recursive(node.right, result)
    
    def delete(self, timestamp: datetime) -> bool:
        """
        Delete a file from the AVL tree.
        
        Args:
            timestamp: Timestamp of file to delete
            
        Returns:
            True if file was deleted, False if not found
        """
        if self.root is None:
            return False
        
        self.root, deleted = self._delete_recursive(self.root, timestamp)
        if deleted:
            self.count -= 1
        return deleted
    
    def _delete_recursive(self, node: Optional[AVLNode], 
                         timestamp: datetime) -> tuple[Optional[AVLNode], bool]:
        """
        Recursive helper for delete with balancing.
        
        Args:
            node: Current node in traversal
            timestamp: Timestamp to delete
            
        Returns:
            Tuple of (updated node, whether deletion occurred)
        """
        if node is None:
            return None, False
        
        deleted = False
        
        # Standard BST deletion
        if timestamp < node.timestamp:
            node.left, deleted = self._delete_recursive(node.left, timestamp)
        elif timestamp > node.timestamp:
            node.right, deleted = self._delete_recursive(node.right, timestamp)
        else:
            # Found the node to delete
            deleted = True
            
            # Node with only one child or no child
            if node.left is None:
                return node.right, deleted
            elif node.right is None:
                return node.left, deleted
            
            # Node with two children
            # Get inorder successor (smallest in right subtree)
            successor = self._find_min(node.right)
            node.timestamp = successor.timestamp
            node.file_data = successor.file_data
            node.right, _ = self._delete_recursive(node.right, successor.timestamp)
        
        if node is None:
            return node, deleted
        
        # Update height
        self._update_height(node)
        
        # Get balance factor
        balance = self._get_balance(node)
        
        # Left-Left case
        if balance > 1 and self._get_balance(node.left) >= 0:
            return self._rotate_right(node), deleted
        
        # Left-Right case
        if balance > 1 and self._get_balance(node.left) < 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node), deleted
        
        # Right-Right case
        if balance < -1 and self._get_balance(node.right) <= 0:
            return self._rotate_left(node), deleted
        
        # Right-Left case
        if balance < -1 and self._get_balance(node.right) > 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node), deleted
        
        return node, deleted
    
    def _find_min(self, node: AVLNode) -> AVLNode:
        """
        Find the node with minimum timestamp in a subtree.
        
        Args:
            node: Root of subtree
            
        Returns:
            Node with minimum timestamp
        """
        current = node
        while current.left is not None:
            current = current.left
        return current
    
    def get_height(self) -> int:
        """
        Get the height of the tree.
        
        Returns:
            Height of the tree
        """
        return self._get_height(self.root)
    
    def __len__(self) -> int:
        """Return the number of files in the tree."""
        return self.count
