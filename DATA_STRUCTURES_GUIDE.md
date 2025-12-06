# Data Structures Guide

## Overview
This Crop Disease Detection System uses custom-implemented data structures to optimize performance for specific operations. Each structure is chosen based on its time complexity advantages for particular use cases.

---

## 1. Hash Table (UserHashTable)

**Purpose:** Fast user lookups by email address

**Implementation:** `app/data_structures/hash_table.py`

**Key Features:**
- Uses chaining (linked lists) for collision resolution
- Email-based hashing for quick user authentication
- Dynamic bucket allocation

**Time Complexity:**
- Insert: O(1) average
- Lookup: O(1) average
- Delete: O(1) average

**Use Cases:**
- User authentication (checking if email exists)
- Quick user profile retrieval during login
- Session management

---

## 2. Binary Search Tree (UserBST)

**Purpose:** Maintain users in sorted order by user code

**Implementation:** `app/data_structures/bst.py`

**Key Features:**
- Stores users sorted by F_code (farmers) or A_Code (admins)
- Supports efficient range queries
- Inorder traversal provides sorted user lists

**Time Complexity:**
- Insert: O(log n) average, O(n) worst
- Search: O(log n) average, O(n) worst
- Delete: O(log n) average, O(n) worst

**Use Cases:**
- Displaying users in sorted order
- Finding users within a code range
- Admin user management interface

---

## 3. AVL Tree (FileAVLTree)

**Purpose:** Self-balancing tree for timestamp-based file storage

**Implementation:** `app/data_structures/avl_tree.py`

**Key Features:**
- Automatically balances after insertions/deletions
- Maintains files sorted by upload timestamp
- Efficient range queries for time-based filtering
- Guaranteed O(log n) operations

**Time Complexity:**
- Insert: O(log n)
- Range Query: O(log n + k) where k = results
- Delete: O(log n)

**Use Cases:**
- Retrieving files uploaded within a date range
- Displaying recent uploads chronologically
- Admin file monitoring dashboard
- Historical data analysis

---

## 4. Queue (ProcessingQueue)

**Purpose:** FIFO queue for managing image processing requests

**Implementation:** `app/data_structures/queue.py`

**Key Features:**
- Uses Python deque for O(1) operations
- Tracks request status (pending, processing, completed, failed)
- Hash table integration for quick status lookups
- Unique request ID generation

**Time Complexity:**
- Enqueue: O(1)
- Dequeue: O(1)
- Status Check: O(1)

**Use Cases:**
- Managing crop disease detection requests
- Processing images in order of submission
- Tracking processing status for farmers
- Background job management

---

## 5. Trie (Prefix Tree)

**Purpose:** Efficient prefix-based filename search

**Implementation:** `app/data_structures/trie.py`

**Key Features:**
- Character-by-character tree structure
- Fast autocomplete functionality
- Prefix matching for search suggestions

**Time Complexity:**
- Insert: O(m) where m = word length
- Search: O(m)
- Prefix Search: O(m + k) where k = results

**Use Cases:**
- Filename autocomplete in search
- Finding files with similar names
- Quick filename validation
- Search suggestions for farmers

---

## 6. Sorting Algorithms

**Purpose:** Custom sorting for multi-criteria data

**Implementation:** `app/data_structures/sorting.py`

**Algorithms:**
- **QuickSort:** O(n log n) average, in-place sorting
- **MergeSort:** O(n log n) guaranteed, stable sorting
- **Binary Search:** O(log n) for sorted data

**Use Cases:**
- Sorting files by multiple criteria (timestamp, confidence, result)
- Sorting users by name, email, or registration date
- Efficient searching in sorted datasets

---

## 7. Utility Structures

**Implementation:** `app/data_structures/utilities.py`

### Stack
- LIFO operations
- Used for undo/redo functionality
- Navigation history

### Circular Buffer
- Fixed-size buffer with wraparound
- Recent activity logging
- Performance metrics tracking

### Min/Max Heap
- Priority queue operations
- Finding top N results
- Scheduling tasks by priority

---

## Performance Benefits

### Why Custom Data Structures?

1. **Optimized for Specific Use Cases**
   - Each structure is tailored to the application's needs
   - No overhead from unused features in generic libraries

2. **Predictable Performance**
   - Known time complexities for all operations
   - No hidden costs from abstraction layers

3. **Memory Efficiency**
   - Minimal memory footprint
   - Direct control over data layout

4. **Educational Value**
   - Demonstrates understanding of algorithms
   - Shows ability to implement complex structures

---

## Integration with Database

The data structures work alongside the MySQL database:

- **Database:** Persistent storage, ACID transactions
- **Data Structures:** In-memory caching, fast lookups, temporary processing

**Workflow:**
1. Data loaded from database into data structures on startup
2. Fast operations performed in-memory
3. Changes persisted back to database
4. Data structures refreshed periodically

---

## Summary

| Structure | Primary Use | Key Advantage |
|-----------|-------------|---------------|
| Hash Table | User lookups | O(1) email search |
| BST | Sorted users | Ordered traversal |
| AVL Tree | File timestamps | Balanced range queries |
| Queue | Processing requests | FIFO ordering |
| Trie | Filename search | Prefix matching |
| Sorting | Multi-criteria ordering | Flexible sorting |

These data structures form the backbone of the application's performance optimization strategy, ensuring fast response times even with large datasets.
