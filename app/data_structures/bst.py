"""
UserBST (Binary Search Tree) implementation for sorted user storage
"""

from typing import Optional, Dict, Any, List, Callable


class BSTNode:
    """Node in a Binary Search Tree."""
    
    def __init__(self, user_code: int, user_data: Dict[str, Any]):
        """
        Initialize a BST node.
        
        Args:
            user_code: User code (F_code or A_Code) used as key
            user_data: Dictionary containing user information
        """
        self.user_code = user_code
        self.user_data = user_data
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None


class UserBST:
    """
    Binary Search Tree implementation for sorted user storage.
    Maintains users sorted by user code (F_code or A_Code).
    
    Time Complexity:
    - Insert: O(log n) average, O(n) worst case
    - Search: O(log n) average, O(n) worst case
    - Delete: O(log n) average, O(n) worst case
    - Inorder Traversal: O(n)
    
    Space Complexity: O(n) where n is the number of users
    """
    
    def __init__(self):
        """Initialize an empty BST."""
        self.root: Optional[BSTNode] = None
        self.count = 0
    
    def insert(self, user_code: int, user_data: Dict[str, Any]) -> None:
        """
        Insert a user into the BST.
        If user_code already exists, update the user data.
        
        Args:
            user_code: User code (F_code or A_Code)
            user_data: Dictionary containing user information
        """
        if self.root is None:
            self.root = BSTNode(user_code, user_data)
            self.count += 1
        else:
            self._insert_recursive(self.root, user_code, user_data)
    
    def _insert_recursive(self, node: BSTNode, user_code: int, user_data: Dict[str, Any]) -> BSTNode:
        """
        Recursive helper for insert operation.
        
        Args:
            node: Current node in traversal
            user_code: User code to insert
            user_data: User data to insert
            
        Returns:
            The node after insertion
        """
        if user_code < node.user_code:
            if node.left is None:
                node.left = BSTNode(user_code, user_data)
                self.count += 1
            else:
                self._insert_recursive(node.left, user_code, user_data)
        elif user_code > node.user_code:
            if node.right is None:
                node.right = BSTNode(user_code, user_data)
                self.count += 1
            else:
                self._insert_recursive(node.right, user_code, user_data)
        else:
            # User code already exists, update data
            node.user_data = user_data
        
        return node
    
    def search(self, user_code: int) -> Optional[Dict[str, Any]]:
        """
        Search for a user by user code.
        
        Args:
            user_code: User code to search for
            
        Returns:
            User data dictionary if found, None otherwise
        """
        return self._search_recursive(self.root, user_code)
    
    def _search_recursive(self, node: Optional[BSTNode], user_code: int) -> Optional[Dict[str, Any]]:
        """
        Recursive helper for search operation.
        
        Args:
            node: Current node in traversal
            user_code: User code to search for
            
        Returns:
            User data if found, None otherwise
        """
        if node is None:
            return None
        
        if user_code == node.user_code:
            return node.user_data
        elif user_code < node.user_code:
            return self._search_recursive(node.left, user_code)
        else:
            return self._search_recursive(node.right, user_code)
    
    def inorder_traversal(self) -> List[Dict[str, Any]]:
        """
        Perform inorder traversal to get users in sorted order by user code.
        
        Returns:
            List of user data dictionaries in sorted order
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: Optional[BSTNode], result: List[Dict[str, Any]]) -> None:
        """
        Recursive helper for inorder traversal.
        
        Args:
            node: Current node in traversal
            result: List to accumulate results
        """
        if node is not None:
            self._inorder_recursive(node.left, result)
            result.append(node.user_data)
            self._inorder_recursive(node.right, result)
    
    def delete(self, user_code: int) -> bool:
        """
        Delete a user from the BST.
        
        Args:
            user_code: User code to delete
            
        Returns:
            True if user was deleted, False if not found
        """
        if self.root is None:
            return False
        
        self.root, deleted = self._delete_recursive(self.root, user_code)
        if deleted:
            self.count -= 1
        return deleted
    
    def _delete_recursive(self, node: Optional[BSTNode], user_code: int) -> tuple[Optional[BSTNode], bool]:
        """
        Recursive helper for delete operation.
        
        Args:
            node: Current node in traversal
            user_code: User code to delete
            
        Returns:
            Tuple of (updated node, whether deletion occurred)
        """
        if node is None:
            return None, False
        
        deleted = False
        
        if user_code < node.user_code:
            node.left, deleted = self._delete_recursive(node.left, user_code)
        elif user_code > node.user_code:
            node.right, deleted = self._delete_recursive(node.right, user_code)
        else:
            # Found the node to delete
            deleted = True
            
            # Case 1: Node has no children
            if node.left is None and node.right is None:
                return None, deleted
            
            # Case 2: Node has only right child
            if node.left is None:
                return node.right, deleted
            
            # Case 3: Node has only left child
            if node.right is None:
                return node.left, deleted
            
            # Case 4: Node has both children
            # Find the inorder successor (minimum in right subtree)
            successor = self._find_min(node.right)
            node.user_code = successor.user_code
            node.user_data = successor.user_data
            # Delete the successor
            node.right, _ = self._delete_recursive(node.right, successor.user_code)
        
        return node, deleted
    
    def _find_min(self, node: BSTNode) -> BSTNode:
        """
        Find the node with minimum value in a subtree.
        
        Args:
            node: Root of subtree
            
        Returns:
            Node with minimum value
        """
        current = node
        while current.left is not None:
            current = current.left
        return current
    
    def exists(self, user_code: int) -> bool:
        """
        Check if a user with given code exists.
        
        Args:
            user_code: User code to check
            
        Returns:
            True if user exists, False otherwise
        """
        return self.search(user_code) is not None
    
    def __len__(self) -> int:
        """Return the number of users in the BST."""
        return self.count
    
    def __contains__(self, user_code: int) -> bool:
        """Support 'in' operator for checking user code existence."""
        return self.exists(user_code)
