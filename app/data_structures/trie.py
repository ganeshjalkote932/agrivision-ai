"""
Trie implementation for prefix-based filename search
"""

from typing import Optional, Dict, Any, List


class TrieNode:
    """Node in a Trie (prefix tree)."""
    
    def __init__(self):
        """Initialize a Trie node."""
        self.children: Dict[str, 'TrieNode'] = {}  # Hash map of character -> TrieNode
        self.is_end_of_word = False
        self.data: Optional[Dict[str, Any]] = None  # Associated file data


class Trie:
    """
    Trie (prefix tree) implementation for efficient prefix-based filename search.
    
    Time Complexity:
    - Insert: O(m) where m is the length of the word
    - Search: O(m) where m is the length of the word
    - Starts With (prefix search): O(m + k) where m is prefix length, k is number of results
    - Delete: O(m) where m is the length of the word
    
    Space Complexity: O(n * m) where n is number of words, m is average word length
    """
    
    def __init__(self):
        """Initialize an empty Trie."""
        self.root = TrieNode()
        self.count = 0
    
    def insert(self, word: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Insert a word (filename) into the Trie.
        
        Args:
            word: Word to insert (typically a filename)
            data: Optional associated data (file metadata)
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # Mark end of word and store data
        if not node.is_end_of_word:
            self.count += 1
        node.is_end_of_word = True
        node.data = data
    
    def search(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Search for an exact word in the Trie.
        
        Args:
            word: Word to search for
            
        Returns:
            Associated data if word exists, None otherwise
        """
        node = self._find_node(word)
        
        if node is not None and node.is_end_of_word:
            return node.data
        
        return None
    
    def _find_node(self, word: str) -> Optional[TrieNode]:
        """
        Find the node corresponding to a word or prefix.
        
        Args:
            word: Word or prefix to find
            
        Returns:
            TrieNode if found, None otherwise
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node
    
    def starts_with(self, prefix: str) -> List[Dict[str, Any]]:
        """
        Find all words (filenames) that start with the given prefix.
        
        Args:
            prefix: Prefix to search for
            
        Returns:
            List of data dictionaries for all matching words
        """
        results = []
        node = self._find_node(prefix)
        
        if node is None:
            return results
        
        # Collect all words with this prefix
        self._collect_words(node, prefix, results)
        
        return results
    
    def _collect_words(self, node: TrieNode, current_word: str, 
                      results: List[Dict[str, Any]]) -> None:
        """
        Recursively collect all words from a given node.
        
        Args:
            node: Current node in traversal
            current_word: Word built so far
            results: List to accumulate results
        """
        if node.is_end_of_word and node.data is not None:
            # Add the current word and its data
            result_data = node.data.copy()
            result_data['filename'] = current_word
            results.append(result_data)
        
        # Recursively traverse all children
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, results)
    
    def delete(self, word: str) -> bool:
        """
        Delete a word from the Trie.
        
        Args:
            word: Word to delete
            
        Returns:
            True if word was deleted, False if not found
        """
        found, _ = self._delete_recursive(self.root, word, 0)
        if found:
            self.count -= 1
        return found
    
    def _delete_recursive(self, node: Optional[TrieNode], word: str, index: int) -> tuple[bool, bool]:
        """
        Recursive helper for delete operation.
        
        Args:
            node: Current node in traversal
            word: Word to delete
            index: Current character index in word
            
        Returns:
            Tuple of (word_found, should_delete_node)
        """
        if node is None:
            return False, False
        
        # Base case: reached end of word
        if index == len(word):
            if not node.is_end_of_word:
                return False, False
            
            node.is_end_of_word = False
            node.data = None
            
            # If node has no children, it can be deleted
            return True, len(node.children) == 0
        
        # Recursive case
        char = word[index]
        if char not in node.children:
            return False, False
        
        child_node = node.children[char]
        found, should_delete_child = self._delete_recursive(child_node, word, index + 1)
        
        if should_delete_child:
            del node.children[char]
            # Return True for should_delete if current node has no children and is not end of another word
            return found, len(node.children) == 0 and not node.is_end_of_word
        
        return found, False
    
    def exists(self, word: str) -> bool:
        """
        Check if a word exists in the Trie.
        
        Args:
            word: Word to check
            
        Returns:
            True if word exists, False otherwise
        """
        return self.search(word) is not None
    
    def get_all_words(self) -> List[str]:
        """
        Get all words stored in the Trie.
        
        Returns:
            List of all words
        """
        words = []
        self._collect_all_words(self.root, "", words)
        return words
    
    def _collect_all_words(self, node: TrieNode, current_word: str, words: List[str]) -> None:
        """
        Recursively collect all words from the Trie.
        
        Args:
            node: Current node in traversal
            current_word: Word built so far
            words: List to accumulate words
        """
        if node.is_end_of_word:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._collect_all_words(child_node, current_word + char, words)
    
    def __len__(self) -> int:
        """Return the number of words in the Trie."""
        return self.count
    
    def __contains__(self, word: str) -> bool:
        """Support 'in' operator for checking word existence."""
        return self.exists(word)
