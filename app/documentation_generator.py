"""
DocumentationGenerator for creating comprehensive data structure documentation.
Scans codebase for data structure usage and generates DATA_STRUCTURES.md
"""

import os
import ast
import inspect
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class DataStructureInfo:
    """Container for data structure information."""
    
    def __init__(self, name: str, file_path: str):
        """
        Initialize data structure info.
        
        Args:
            name: Name of the data structure class
            file_path: Path to the file containing the class
        """
        self.name = name
        self.file_path = file_path
        self.docstring = ""
        self.time_complexity: Dict[str, str] = {}
        self.space_complexity = ""
        self.methods: List[str] = []
        self.usage_locations: List[str] = []
        self.use_cases: List[str] = []
        self.code_example = ""
        self.rationale = ""
        self.alternatives = ""


class DocumentationGenerator:
    """
    Generates comprehensive documentation for data structures used in the system.
    Scans codebase to extract data structure information and creates DATA_STRUCTURES.md
    """
    
    def __init__(self, project_root: str = "."):
        """
        Initialize the documentation generator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.data_structures: Dict[str, DataStructureInfo] = {}
        
        # Define known data structures and their locations
        self.ds_files = {
            'app/data_structures/hash_table.py': ['UserHashTable'],
            'app/data_structures/bst.py': ['UserBST', 'BSTNode'],
            'app/data_structures/avl_tree.py': ['FileAVLTree', 'AVLNode'],
            'app/data_structures/queue.py': ['ProcessingQueue', 'ProcessRequest'],
            'app/data_structures/trie.py': ['Trie', 'TrieNode'],
            'app/data_structures/sorting.py': ['quicksort', 'mergesort', 'binary_search'],
            'app/data_structures/utilities.py': ['Stack', 'CircularBuffer', 'MinHeap', 'MaxHeap']
        }
    
    def scan_codebase(self) -> None:
        """
        Scan the codebase for data structure definitions and usage.
        Extracts names, locations, and use cases.
        """
        print("Scanning codebase for data structures...")
        
        # Extract data structure information from definition files
        for file_path, ds_names in self.ds_files.items():
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                self._extract_ds_info(full_path, ds_names)
        
        # Find usage locations across the codebase
        self._find_usage_locations()
        
        print(f"Found {len(self.data_structures)} data structures")
    
    def _extract_ds_info(self, file_path: str, ds_names: List[str]) -> None:
        """
        Extract information from a data structure definition file.
        
        Args:
            file_path: Path to the file
            ds_names: Names of data structures in the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            # Extract information for each data structure
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in ds_names:
                    self._extract_class_info(node, file_path, content)
                elif isinstance(node, ast.FunctionDef) and node.name in ds_names:
                    self._extract_function_info(node, file_path, content)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _extract_class_info(self, node: ast.ClassDef, file_path: str, content: str) -> None:
        """
        Extract information from a class definition.
        
        Args:
            node: AST node for the class
            file_path: Path to the file
            content: Full file content
        """
        ds_info = DataStructureInfo(node.name, file_path)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        if docstring:
            ds_info.docstring = docstring
            # Parse complexity information from docstring
            self._parse_complexity(docstring, ds_info)
        
        # Extract method names
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                ds_info.methods.append(item.name)
        
        # Extract code example (the class definition)
        ds_info.code_example = self._extract_code_snippet(content, node.lineno, node.end_lineno)
        
        self.data_structures[node.name] = ds_info
    
    def _extract_function_info(self, node: ast.FunctionDef, file_path: str, content: str) -> None:
        """
        Extract information from a function definition (for sorting algorithms).
        
        Args:
            node: AST node for the function
            file_path: Path to the file
            content: Full file content
        """
        ds_info = DataStructureInfo(node.name, file_path)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        if docstring:
            ds_info.docstring = docstring
            self._parse_complexity(docstring, ds_info)
        
        # Extract code example
        ds_info.code_example = self._extract_code_snippet(content, node.lineno, node.end_lineno)
        
        self.data_structures[node.name] = ds_info
    
    def _parse_complexity(self, docstring: str, ds_info: DataStructureInfo) -> None:
        """
        Parse time and space complexity from docstring.
        
        Args:
            docstring: The docstring to parse
            ds_info: DataStructureInfo object to update
        """
        lines = docstring.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'Time Complexity:' in line:
                current_section = 'time'
            elif 'Space Complexity:' in line:
                current_section = 'space'
                # Extract space complexity
                if ':' in line:
                    ds_info.space_complexity = line.split(':', 1)[1].strip()
            elif current_section == 'time' and '-' in line:
                # Parse operation: complexity pairs
                parts = line.split(':', 1)
                if len(parts) == 2:
                    operation = parts[0].strip('- ').strip()
                    complexity = parts[1].strip()
                    ds_info.time_complexity[operation] = complexity
            elif current_section == 'space' and line and not line.startswith('-'):
                if not ds_info.space_complexity:
                    ds_info.space_complexity = line
    
    def _extract_code_snippet(self, content: str, start_line: int, end_line: Optional[int]) -> str:
        """
        Extract a code snippet from file content.
        
        Args:
            content: Full file content
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            
        Returns:
            Code snippet as string
        """
        lines = content.split('\n')
        if end_line is None:
            end_line = len(lines)
        
        # Get the snippet (convert to 0-indexed)
        snippet_lines = lines[start_line - 1:end_line]
        
        # Limit to first 30 lines for brevity
        if len(snippet_lines) > 30:
            snippet_lines = snippet_lines[:30]
            snippet_lines.append("    # ... (truncated for brevity)")
        
        return '\n'.join(snippet_lines)
    
    def _find_usage_locations(self) -> None:
        """
        Find where each data structure is used in the codebase.
        """
        print("Finding usage locations...")
        
        # Directories to search
        search_dirs = ['app']
        
        for search_dir in search_dirs:
            full_dir = os.path.join(self.project_root, search_dir)
            if not os.path.exists(full_dir):
                continue
            
            for root, dirs, files in os.walk(full_dir):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != '__pycache__']
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        self._scan_file_for_usage(file_path)
    
    def _scan_file_for_usage(self, file_path: str) -> None:
        """
        Scan a file for data structure usage.
        
        Args:
            file_path: Path to the file to scan
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for imports and usage of each data structure
            for ds_name, ds_info in self.data_structures.items():
                if ds_name in content:
                    # Get relative path
                    rel_path = os.path.relpath(file_path, self.project_root)
                    if rel_path not in ds_info.usage_locations and rel_path != ds_info.file_path:
                        ds_info.usage_locations.append(rel_path)
                        
                        # Try to determine use case from file name
                        use_case = self._infer_use_case(rel_path, ds_name)
                        if use_case and use_case not in ds_info.use_cases:
                            ds_info.use_cases.append(use_case)
        
        except Exception as e:
            pass  # Skip files that can't be read
    
    def _infer_use_case(self, file_path: str, ds_name: str) -> Optional[str]:
        """
        Infer the use case from file path and data structure name.
        
        Args:
            file_path: Path to the file using the data structure
            ds_name: Name of the data structure
            
        Returns:
            Inferred use case description
        """
        use_case_map = {
            'UserHashTable': {
                'auth_service.py': 'User authentication and duplicate email checking',
                'file_service.py': 'Fast user lookup for file operations'
            },
            'UserBST': {
                'auth_service.py': 'Maintaining sorted administrator records'
            },
            'FileAVLTree': {
                'file_service.py': 'Timestamp-based file storage and range queries'
            },
            'ProcessingQueue': {
                'farmer/routes.py': 'Queuing image upload requests',
                'processing_worker.py': 'Processing image analysis requests in FIFO order'
            },
            'Trie': {
                'file_service.py': 'Prefix-based filename search and autocomplete'
            },
            'quicksort': {
                'admin/routes.py': 'Multi-column sorting of user lists'
            },
            'mergesort': {
                'farmer/routes.py': 'Sorting file history by timestamp',
                'admin/routes.py': 'Stable sorting of file records',
                'file_service.py': 'Sorting files for display'
            },
            'Stack': {
                'admin/routes.py': 'Undo functionality for administrative actions'
            },
            'CircularBuffer': {
                'statistics_engine.py': 'Tracking recent prediction activity'
            }
        }
        
        # Get the file name
        file_name = os.path.basename(file_path)
        
        if ds_name in use_case_map and file_name in use_case_map[ds_name]:
            return use_case_map[ds_name][file_name]
        
        return None
    
    def collect_complexity_info(self) -> None:
        """
        Collect time and space complexity information for all data structures.
        This information is already extracted from docstrings in scan_codebase.
        """
        print("Complexity information collected from docstrings")
    
    def generate_documentation(self) -> str:
        """
        Generate the complete DATA_STRUCTURES.md documentation.
        
        Returns:
            Complete documentation as markdown string
        """
        print("Generating documentation...")
        
        doc = []
        
        # Add header
        doc.append("# Data Structures Documentation")
        doc.append("")
        doc.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        doc.append("")
        
        # Add executive summary
        doc.append(self._generate_executive_summary())
        doc.append("")
        
        # Add data structure mapping table
        doc.append(self._generate_mapping_table())
        doc.append("")
        
        # Add detailed catalog
        doc.append(self._generate_detailed_catalog())
        doc.append("")
        
        # Add performance analysis
        doc.append(self._generate_performance_analysis())
        doc.append("")
        
        return '\n'.join(doc)
    
    def _generate_executive_summary(self) -> str:
        """Generate the executive summary section."""
        summary = []
        summary.append("## Executive Summary")
        summary.append("")
        summary.append("The Crop Disease Detection System leverages classic data structures throughout its ")
        summary.append("implementation to achieve optimal performance for searching, sorting, and data management ")
        summary.append("operations. This document provides a comprehensive catalog of all data structures used, ")
        summary.append("their implementations, use cases, and performance characteristics.")
        summary.append("")
        summary.append("### Overview of Data Structures Used")
        summary.append("")
        
        # Group data structures by category
        categories = {
            'Hash-Based Structures': ['UserHashTable'],
            'Tree Structures': ['UserBST', 'FileAVLTree', 'Trie'],
            'Linear Structures': ['ProcessingQueue', 'Stack', 'CircularBuffer'],
            'Heap Structures': ['MinHeap', 'MaxHeap'],
            'Algorithms': ['quicksort', 'mergesort', 'binary_search']
        }
        
        for category, ds_names in categories.items():
            summary.append(f"**{category}:**")
            for ds_name in ds_names:
                if ds_name in self.data_structures:
                    ds_info = self.data_structures[ds_name]
                    # Get first line of docstring as brief description
                    brief = ds_info.docstring.split('\n')[0] if ds_info.docstring else ds_name
                    summary.append(f"- `{ds_name}`: {brief}")
            summary.append("")
        
        summary.append("### Design Philosophy")
        summary.append("")
        summary.append("The system emphasizes:")
        summary.append("- **Performance**: Using appropriate data structures for O(1) or O(log n) operations")
        summary.append("- **Scalability**: Structures that maintain performance as data grows")
        summary.append("- **Clarity**: Clean implementations that demonstrate understanding")
        summary.append("- **Correctness**: Well-tested structures with clear complexity guarantees")
        summary.append("")
        
        return '\n'.join(summary)
    
    def _generate_mapping_table(self) -> str:
        """Generate the data structure mapping table."""
        table = []
        table.append("## Data Structure Mapping")
        table.append("")
        table.append("| Component | Data Structure | Primary Operations | Time Complexity |")
        table.append("|-----------|---------------|-------------------|-----------------|")
        
        # Define mappings
        mappings = [
            ("User Lookup", "UserHashTable", "Insert, Search, Delete", "O(1) average"),
            ("Admin Storage", "UserBST", "Insert, Inorder Traversal", "O(log n) average"),
            ("File Timeline", "FileAVLTree", "Insert, Range Query", "O(log n)"),
            ("Image Processing", "ProcessingQueue", "Enqueue, Dequeue", "O(1)"),
            ("Filename Search", "Trie", "Insert, Prefix Search", "O(m) where m=length"),
            ("User Search", "binary_search", "Search in sorted array", "O(log n)"),
            ("File Sorting", "mergesort", "Sort", "O(n log n)"),
            ("User Sorting", "quicksort", "Sort", "O(n log n) average"),
            ("Recent Activity", "CircularBuffer", "Insert, Read", "O(1)"),
            ("Admin Undo", "Stack", "Push, Pop", "O(1)"),
            ("Priority Operations", "MinHeap/MaxHeap", "Insert, Extract", "O(log n)")
        ]
        
        for component, ds, operations, complexity in mappings:
            table.append(f"| {component} | {ds} | {operations} | {complexity} |")
        
        table.append("")
        return '\n'.join(table)
    
    def _generate_detailed_catalog(self) -> str:
        """Generate the detailed data structure catalog."""
        catalog = []
        catalog.append("## Detailed Data Structure Catalog")
        catalog.append("")
        
        # Order data structures logically
        ordered_ds = [
            'UserHashTable', 'UserBST', 'FileAVLTree', 'ProcessingQueue',
            'Trie', 'quicksort', 'mergesort', 'binary_search',
            'Stack', 'CircularBuffer', 'MinHeap', 'MaxHeap'
        ]
        
        for ds_name in ordered_ds:
            if ds_name in self.data_structures:
                ds_info = self.data_structures[ds_name]
                catalog.append(self._generate_ds_section(ds_info))
                catalog.append("")
        
        return '\n'.join(catalog)
    
    def _generate_ds_section(self, ds_info: DataStructureInfo) -> str:
        """
        Generate documentation section for a single data structure.
        
        Args:
            ds_info: DataStructureInfo object
            
        Returns:
            Markdown section as string
        """
        section = []
        section.append(f"### {ds_info.name}")
        section.append("")
        
        # Location
        section.append(f"**Location:** `{ds_info.file_path}`")
        section.append("")
        
        # Description
        if ds_info.docstring:
            # Get first paragraph of docstring
            desc_lines = []
            for line in ds_info.docstring.split('\n'):
                line = line.strip()
                if line and not line.startswith('Time') and not line.startswith('Space'):
                    desc_lines.append(line)
                elif desc_lines and (line.startswith('Time') or line.startswith('Space')):
                    break
            
            if desc_lines:
                section.append("**Description:**")
                section.append(' '.join(desc_lines))
                section.append("")
        
        # Use cases
        if ds_info.use_cases:
            section.append("**Use Cases:**")
            for use_case in ds_info.use_cases:
                section.append(f"- {use_case}")
            section.append("")
        
        # Time complexity
        if ds_info.time_complexity:
            section.append("**Time Complexity:**")
            for operation, complexity in ds_info.time_complexity.items():
                section.append(f"- {operation}: {complexity}")
            section.append("")
        
        # Space complexity
        if ds_info.space_complexity:
            section.append(f"**Space Complexity:** {ds_info.space_complexity}")
            section.append("")
        
        # Usage locations
        if ds_info.usage_locations:
            section.append("**Used In:**")
            for location in ds_info.usage_locations[:5]:  # Limit to 5
                section.append(f"- `{location}`")
            section.append("")
        
        # Code example
        usage_example = self._get_usage_example(ds_info.name)
        if usage_example:
            section.append("**Usage Example:**")
            section.append("```python")
            section.append(usage_example)
            section.append("```")
            section.append("")
        
        # Rationale
        rationale = self._get_rationale(ds_info.name)
        if rationale:
            section.append("**Rationale:**")
            section.append(rationale)
            section.append("")
        
        # Alternatives and comparisons
        alternatives = self._get_alternatives(ds_info.name)
        if alternatives:
            section.append("**Alternatives Considered:**")
            section.append(alternatives)
            section.append("")
        
        return '\n'.join(section)
    
    def _get_usage_example(self, ds_name: str) -> Optional[str]:
        """
        Get a usage example for a data structure.
        
        Args:
            ds_name: Name of the data structure
            
        Returns:
            Usage example as string
        """
        examples = {
            'UserHashTable': """# Create hash table for user lookups
user_table = UserHashTable(size=1000)

# Insert a user
user_data = {'name': 'John Doe', 'email': 'john@example.com', 'F_code': 12345}
user_table.insert('john@example.com', user_data)

# Look up a user by email (O(1) average)
user = user_table.lookup('john@example.com')

# Check if user exists
if user_table.exists('john@example.com'):
    print("User found!")""",
            
            'UserBST': """# Create BST for sorted admin storage
admin_bst = UserBST()

# Insert admins (automatically sorted by A_Code)
admin_bst.insert(1001, {'name': 'Admin One', 'email': 'admin1@example.com'})
admin_bst.insert(1003, {'name': 'Admin Three', 'email': 'admin3@example.com'})
admin_bst.insert(1002, {'name': 'Admin Two', 'email': 'admin2@example.com'})

# Get all admins in sorted order (O(n))
sorted_admins = admin_bst.inorder_traversal()""",
            
            'FileAVLTree': """# Create AVL tree for timestamp-based file storage
file_tree = FileAVLTree()

# Insert files (automatically balanced)
from datetime import datetime
file_tree.insert(datetime.now(), {'filename': 'crop1.jpg', 'result': 'healthy'})

# Range query for files in time period (O(log n + k))
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
files_in_range = file_tree.range_query(start_date, end_date)""",
            
            'ProcessingQueue': """# Create queue for image processing
queue = ProcessingQueue()

# Enqueue processing request (O(1))
request_id = queue.enqueue(
    farmer_code=12345,
    image_path='/uploads/crop.jpg',
    filename='crop.jpg'
)

# Dequeue next request for processing (O(1))
request = queue.dequeue()

# Update status after processing
queue.update_status(request_id, 'completed', result={'disease': 'healthy'})""",
            
            'Trie': """# Create trie for filename search
trie = Trie()

# Insert filenames
trie.insert('crop_image_001.jpg', {'file_id': 1})
trie.insert('crop_image_002.jpg', {'file_id': 2})
trie.insert('crop_scan_001.jpg', {'file_id': 3})

# Prefix search (O(m + k) where m=prefix length, k=results)
results = trie.starts_with('crop_image')  # Returns first 2 files""",
            
            'quicksort': """# Sort users by multiple columns
from app.data_structures.sorting import quicksort

users = [
    {'name': 'Alice', 'email': 'alice@example.com', 'date': '2024-01-15'},
    {'name': 'Bob', 'email': 'bob@example.com', 'date': '2024-01-10'},
]

# Sort by name (O(n log n) average)
sorted_users = quicksort(users, key='name', reverse=False)""",
            
            'mergesort': """# Sort file history (stable sort)
from app.data_structures.sorting import mergesort

files = [
    {'filename': 'crop1.jpg', 'timestamp': '2024-01-15 10:00:00'},
    {'filename': 'crop2.jpg', 'timestamp': '2024-01-15 09:00:00'},
]

# Sort by timestamp descending (O(n log n))
sorted_files = mergesort(files, key='timestamp', reverse=True)""",
            
            'Stack': """# Create stack for undo functionality
from app.data_structures.utilities import Stack

undo_stack = Stack()

# Push actions (O(1))
undo_stack.push({'action': 'delete_user', 'user_id': 123})
undo_stack.push({'action': 'deactivate_user', 'user_id': 456})

# Pop last action to undo (O(1))
last_action = undo_stack.pop()""",
            
            'CircularBuffer': """# Create circular buffer for recent activity
from app.data_structures.utilities import CircularBuffer

recent_activity = CircularBuffer(capacity=100)

# Insert activity (O(1), overwrites oldest when full)
recent_activity.insert({'action': 'file_upload', 'timestamp': '2024-01-15'})

# Get recent items
last_10 = recent_activity.get_recent(10)""",
            
            'MinHeap': """# Create min heap for priority operations
from app.data_structures.utilities import MinHeap

heap = MinHeap()

# Insert with priority (O(log n))
heap.insert(priority=5, data={'task': 'process_image_1'})
heap.insert(priority=2, data={'task': 'process_image_2'})

# Extract minimum priority item (O(log n))
min_item = heap.extract_min()  # Returns priority=2 item""",
            
            'binary_search': """# Binary search in sorted array
from app.data_structures.sorting import binary_search

sorted_users = [
    {'user_code': 1001, 'name': 'Alice'},
    {'user_code': 1002, 'name': 'Bob'},
    {'user_code': 1003, 'name': 'Charlie'},
]

# Search for user (O(log n))
index = binary_search(sorted_users, key='user_code', value=1002)"""
        }
        
        return examples.get(ds_name)
    
    def _get_rationale(self, ds_name: str) -> Optional[str]:
        """
        Get the rationale for choosing a data structure.
        
        Args:
            ds_name: Name of the data structure
            
        Returns:
            Rationale explanation as string
        """
        rationales = {
            'UserHashTable': """Hash tables were chosen for user lookups because:
- Email-based lookups are the most common operation in authentication
- O(1) average-case lookup time is critical for login performance
- Chaining handles collisions gracefully without performance degradation
- Simple implementation with predictable behavior""",
            
            'UserBST': """Binary Search Tree was chosen for administrator storage because:
- Admins need to be retrieved in sorted order by A_Code
- Inorder traversal provides sorted output in O(n) time
- Insert and search operations are O(log n) on average
- Simpler than AVL tree when perfect balance isn't critical (small admin count)""",
            
            'FileAVLTree': """AVL Tree was chosen for file storage because:
- Files must be efficiently queried by timestamp ranges
- Self-balancing ensures O(log n) operations even with skewed insertions
- Range queries are efficient: O(log n + k) where k is result count
- Maintains sorted order for chronological file display
- Better than BST because file uploads may be bursty (unbalanced insertions)""",
            
            'ProcessingQueue': """Queue (using deque) was chosen for image processing because:
- FIFO ordering ensures fair processing (first uploaded, first processed)
- O(1) enqueue and dequeue operations minimize overhead
- Python's deque is optimized for both ends
- Hash table tracking prevents duplicate processing""",
            
            'Trie': """Trie was chosen for filename search because:
- Prefix search is O(m) where m is prefix length, independent of total files
- Enables autocomplete functionality efficiently
- All matching filenames share common prefix storage (space efficient)
- Better than linear search O(n*m) or binary search O(log n * m)""",
            
            'quicksort': """QuickSort was chosen for user sorting because:
- O(n log n) average case with good cache locality
- In-place sorting minimizes memory usage
- Excellent performance on random data (typical user lists)
- Flexible for multi-column sorting with custom comparators""",
            
            'mergesort': """MergeSort was chosen for file history sorting because:
- Stable sort preserves relative order of equal elements
- Guaranteed O(n log n) worst-case performance
- Predictable behavior for time-critical operations
- Better than QuickSort when stability is required""",
            
            'Stack': """Stack was chosen for undo functionality because:
- LIFO ordering matches undo semantics (undo most recent first)
- O(1) push and pop operations
- Simple and intuitive implementation
- Natural fit for reversible action tracking""",
            
            'CircularBuffer': """Circular Buffer was chosen for recent activity because:
- Fixed memory footprint regardless of total activity
- O(1) insertion with automatic oldest-item eviction
- Efficient for "last N items" queries
- No memory allocation/deallocation overhead""",
            
            'MinHeap': """Min Heap was chosen for priority operations because:
- O(log n) insertion and extraction of minimum
- Efficient for priority-based task scheduling
- Better than sorted array (O(n) insertion) or unsorted array (O(n) extraction)
- Compact array-based representation""",
            
            'binary_search': """Binary Search was chosen for sorted array lookups because:
- O(log n) search time in sorted data
- No additional space required
- Simple implementation with predictable performance
- Ideal for infrequent updates, frequent searches"""
        }
        
        return rationales.get(ds_name)
    
    def _get_alternatives(self, ds_name: str) -> Optional[str]:
        """
        Get alternatives considered and performance comparisons.
        
        Args:
            ds_name: Name of the data structure
            
        Returns:
            Alternatives explanation as string
        """
        alternatives = {
            'UserHashTable': """**Alternative: Python Dictionary**
- Pros: Built-in, well-tested, similar O(1) performance
- Cons: Less educational value, less control over collision handling
- Decision: Custom implementation demonstrates understanding and allows optimization

**Alternative: Database Index**
- Pros: Persistent, handles large datasets
- Cons: Network/disk I/O overhead, slower than in-memory
- Decision: Hash table for hot data, database for persistence""",
            
            'UserBST': """**Alternative: AVL Tree**
- Pros: Guaranteed O(log n) with perfect balance
- Cons: More complex rotations, overhead for small datasets
- Decision: BST sufficient for small admin count, simpler implementation

**Alternative: Sorted Array**
- Pros: Simple, good cache locality
- Cons: O(n) insertion time
- Decision: BST better for frequent insertions""",
            
            'FileAVLTree': """**Alternative: Regular BST**
- Pros: Simpler implementation
- Cons: Can degrade to O(n) with sequential insertions
- Decision: AVL chosen because file uploads are time-sequential (would create unbalanced BST)

**Alternative: B-Tree**
- Pros: Better for disk-based storage
- Cons: More complex, unnecessary for in-memory
- Decision: AVL tree optimal for in-memory timestamp indexing""",
            
            'ProcessingQueue': """**Alternative: Priority Queue (Heap)**
- Pros: Can prioritize urgent requests
- Cons: More complex, FIFO fairness lost
- Decision: Simple queue ensures fair processing order

**Alternative: Database Queue**
- Pros: Persistent across restarts
- Cons: Slower, unnecessary complexity
- Decision: In-memory queue sufficient for current scale""",
            
            'Trie': """**Alternative: Hash Table with Prefix Scanning**
- Pros: Simpler implementation
- Cons: O(n) to find all matches, no prefix sharing
- Decision: Trie provides O(m + k) prefix search

**Alternative: Suffix Tree**
- Pros: Supports substring search
- Cons: More complex, higher memory usage
- Decision: Trie sufficient for prefix-only search""",
            
            'quicksort': """**Alternative: MergeSort**
- Pros: Stable, guaranteed O(n log n)
- Cons: O(n) extra space, slower in practice
- Decision: QuickSort for speed when stability not needed

**Alternative: Built-in sort()**
- Pros: Highly optimized (Timsort)
- Cons: Less educational value
- Decision: Custom implementation demonstrates algorithm knowledge""",
            
            'mergesort': """**Alternative: QuickSort**
- Pros: Faster average case, in-place
- Cons: Unstable, O(n²) worst case
- Decision: MergeSort chosen for stability (preserve upload order for same timestamp)

**Alternative: Timsort**
- Pros: Adaptive, excellent on partially sorted data
- Cons: More complex implementation
- Decision: MergeSort simpler and sufficient""",
            
            'Stack': """**Alternative: Array with Index**
- Pros: Simpler, direct access
- Cons: Manual index management, more error-prone
- Decision: Stack abstraction clearer and safer

**Alternative: Linked List**
- Pros: Dynamic size
- Cons: Pointer overhead, worse cache locality
- Decision: Array-based stack faster for typical undo depth""",
            
            'CircularBuffer': """**Alternative: Queue with Size Limit**
- Pros: Simpler logic
- Cons: Requires dequeue when full (extra operation)
- Decision: Circular buffer more efficient with automatic overwrite

**Alternative: Linked List with Tail Pointer**
- Pros: Dynamic size
- Cons: Memory allocation overhead, pointer chasing
- Decision: Fixed-size array faster and more predictable""",
            
            'MinHeap': """**Alternative: Sorted Array**
- Pros: Simple, O(1) minimum access
- Cons: O(n) insertion time
- Decision: Heap provides O(log n) insertion

**Alternative: Unsorted Array**
- Pros: O(1) insertion
- Cons: O(n) to find minimum
- Decision: Heap balances both operations at O(log n)"""
        }
        
        return alternatives.get(ds_name)
    
    def _generate_performance_analysis(self) -> str:
        """Generate the performance analysis section."""
        analysis = []
        analysis.append("## Performance Analysis")
        analysis.append("")
        analysis.append("### Complexity Summary")
        analysis.append("")
        analysis.append("The system achieves excellent performance through careful data structure selection:")
        analysis.append("")
        analysis.append("**O(1) Operations:**")
        analysis.append("- User lookup by email (UserHashTable)")
        analysis.append("- Session management (hash-based)")
        analysis.append("- Queue operations (ProcessingQueue)")
        analysis.append("- Stack operations (admin undo)")
        analysis.append("- Circular buffer operations (recent activity)")
        analysis.append("")
        analysis.append("**O(log n) Operations:**")
        analysis.append("- File insertion and range queries (FileAVLTree with auto-balancing)")
        analysis.append("- Admin record access (UserBST)")
        analysis.append("- Binary search in sorted arrays")
        analysis.append("- Heap operations (priority queues)")
        analysis.append("")
        analysis.append("**O(n log n) Operations:**")
        analysis.append("- Sorting user lists (QuickSort)")
        analysis.append("- Sorting file history (MergeSort - stable)")
        analysis.append("")
        analysis.append("**O(m) Operations (where m = string length):**")
        analysis.append("- Filename prefix search (Trie)")
        analysis.append("")
        analysis.append("### Scalability Considerations")
        analysis.append("")
        analysis.append("The data structure choices ensure the system scales well:")
        analysis.append("")
        analysis.append("- **Hash tables** provide constant-time user lookups regardless of user count")
        analysis.append("- **AVL trees** maintain O(log n) performance even with millions of files")
        analysis.append("- **Tries** enable fast prefix search without scanning entire filename lists")
        analysis.append("- **Queues** ensure fair processing order with minimal overhead")
        analysis.append("")
        
        return '\n'.join(analysis)
