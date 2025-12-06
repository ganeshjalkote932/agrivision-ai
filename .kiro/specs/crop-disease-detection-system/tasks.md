# Implementation Plan

- [x] 1. Set up project structure and dependencies





  - Create Flask application structure with blueprints for auth, farmer, and admin modules
  - Install required dependencies: Flask, PyTorch, MySQL connector, bcrypt, Pillow, Hypothesis
  - Configure database connection with connection pooling
  - Set up configuration management for database, model, and application settings
  - Create directory structure for uploads, models, and static files
  - _Requirements: All_

- [x] 2. Implement core data structures





  - [x] 2.1 Implement UserHashTable class for O(1) user lookups


    - Create hash table with chaining for collision resolution
    - Implement insert, lookup, exists, and delete methods
    - Add hash function for email-based indexing
    - _Requirements: 1.5_
  
  - [x] 2.2 Implement UserBST class for sorted user storage


    - Create binary search tree node structure
    - Implement insert, search, and inorder traversal methods
    - Add delete method with proper node replacement
    - _Requirements: 2.4_


  
  - [x] 2.3 Implement ProcessingQueue class for image processing

    - Create queue using Python deque for O(1) operations
    - Implement enqueue, dequeue, and status tracking methods

    - Add hash table for in-progress request tracking
    - _Requirements: 5.2, 11.1, 11.2, 11.4_
  
  - [x] 2.4 Implement FileAVLTree class for timestamp-based file storage

    - Create AVL tree node with height tracking
    - Implement insert with automatic balancing (rotations)
    - Add range query method for time-based file retrieval
    - Implement inorder traversal for sorted file listing
    - _Requirements: 8.2_
  
  - [x] 2.5 Implement Trie class for prefix-based filename search


    - Create trie node with children hash map
    - Implement insert method for adding filenames
    - Add starts_with method for prefix search
    - _Requirements: 6.5_
  
  - [x] 2.6 Implement sorting algorithms (QuickSort and MergeSort)


    - Create QuickSort implementation for multi-column user sorting
    - Create MergeSort implementation for file history sorting
    - Add comparison functions for different sort keys
    - _Requirements: 6.2, 7.4_
  
  - [x] 2.7 Implement utility data structures (Stack, CircularBuffer, Heap)


    - Create Stack for admin undo functionality
    - Create CircularBuffer for recent activity tracking
    - Create MinHeap/MaxHeap for priority-based sorting
    - _Requirements: 9.3, 10.4_
  
  - [ ]* 2.8 Write property tests for data structure correctness
    - **Property 2: Unique farmer code generation**
    - **Property 6: Unique admin code generation**
    - **Property 37: Request ordering (FIFO)**
    - **Property 39: Duplicate processing prevention**
    - **Validates: Requirements 1.2, 2.2, 11.1, 11.2, 11.4**

- [x] 3. Implement database layer





  - [x] 3.1 Create database connection manager

    - Implement connection pooling with retry logic
    - Add error handling for connection failures
    - Create context manager for transaction handling
    - _Requirements: All database operations_
  
  - [x] 3.2 Create database schema and tables

    - Execute SQL to create Administrator table (if not exists)
    - Execute SQL to create farmer table (if not exists)
    - Create file_uploads table with foreign key to farmer
    - Create admin_actions log table
    - Add indexes on email, F_code, A_Code columns
    - _Requirements: 1.2, 2.2, 5.5, 10.5_
  
  - [x] 3.3 Implement database repository classes


    - Create FarmerRepository with CRUD operations
    - Create AdminRepository with CRUD operations
    - Create FileRepository with CRUD operations
    - Create AdminActionRepository for logging
    - Use parameterized queries to prevent SQL injection
    - _Requirements: 1.2, 2.2, 5.5, 7.1, 8.1, 10.5_
  
  - [ ]* 3.4 Write unit tests for database operations
    - Test CRUD operations for all repositories
    - Test foreign key constraints and cascade deletion
    - Test transaction rollback on errors
    - _Requirements: 1.2, 2.2, 5.5, 10.3_

- [-] 4. Implement authentication and session management



  - [x] 4.1 Create password hashing utilities


    - Implement password hashing using bcrypt
    - Add password strength validation
    - Create password verification function
    - _Requirements: 1.1, 2.1_
  
  - [x] 4.2 Implement SessionManager class


    - Create session hash map for O(1) session lookup
    - Implement create_session, get_session, validate_session methods
    - Add session expiration checking
    - Implement cleanup for expired sessions
    - _Requirements: 3.2, 3.4, 3.5, 4.2_
  
  - [x] 4.3 Implement AuthService for farmer authentication


    - Create register_farmer method with validation and hash table duplicate checking
    - Implement login_farmer with credential verification
    - Add logout functionality
    - Generate unique F_code using hash-based approach
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3_
  
  - [x] 4.4 Implement AuthService for admin authentication


    - Create register_admin method with special code validation
    - Implement login_admin with credential verification and logging
    - Generate unique A_Code
    - Store admin in BST for sorted access
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 4.2, 4.3, 4.5_
  
  - [ ]* 4.5 Write property tests for authentication
    - **Property 1: Registration input validation**
    - **Property 3: Duplicate email rejection**
    - **Property 7: Invalid special code rejection**
    - **Property 9: Valid farmer login creates session**
    - **Property 10: Invalid farmer credentials rejection**
    - **Property 12: Valid admin login creates privileged session**
    - **Property 13: Admin login error message security**
    - **Validates: Requirements 1.1, 1.3, 2.3, 3.2, 3.3, 4.2, 4.3**

- [x] 5. Implement model loading and inference





  - [x] 5.1 Create ModelService class


    - Load PyTorch model from best_model (2).pth at startup
    - Implement model caching in memory
    - Add error handling for model loading failures
    - Detect and use GPU if available, otherwise CPU
    - _Requirements: 5.3_
  
  - [x] 5.2 Implement image preprocessing

    - Create ImagePreprocessor for hyperspectral image handling
    - Implement image validation (format, size, dimensions)
    - Add image transformation for model input
    - _Requirements: 5.1_
  
  - [x] 5.3 Implement inference method


    - Create predict method that takes image path
    - Return binary result (diseased/healthy) with confidence score
    - Add error handling for inference failures
    - Format output as dictionary with result and confidence
    - _Requirements: 5.3, 5.4_
  
  - [x] 5.4 Implement get_model_info method

    - Extract and return model architecture details
    - Include model parameters, layers, and configuration
    - _Requirements: 9.1_
  
  - [ ]* 5.5 Write property tests for model inference
    - **Property 15: Image file validation**
    - **Property 17: Inference output format**
    - **Validates: Requirements 5.1, 5.4**

- [x] 6. Implement file upload and processing workflow




  - [x] 6.1 Create FileService class


    - Implement save_file_metadata method to store upload info
    - Create get_farmer_files method with merge sort by timestamp
    - Implement get_all_files for admin access
    - Add filter_files method using hash table for status filtering
    - Implement search_files using trie for prefix search
    - _Requirements: 5.5, 6.1, 6.2, 6.5, 8.1, 8.4_
  
  - [x] 6.2 Implement file upload handler


    - Create upload endpoint that validates file
    - Save file to disk with unique filename
    - Add request to ProcessingQueue
    - Return request_id for status tracking
    - _Requirements: 5.1, 5.2_
  
  - [x] 6.3 Implement background processing worker


    - Create worker that dequeues processing requests
    - Invoke ModelService.predict for each request
    - Update file_uploads table with results
    - Update request status in queue
    - Handle processing errors gracefully
    - _Requirements: 5.3, 5.4, 11.2, 11.3, 11.5_
  
  - [x] 6.4 Integrate FileAVLTree for efficient file management


    - Insert file metadata into AVL tree on upload
    - Use tree for range queries by timestamp
    - Maintain tree balance after insertions
    - _Requirements: 8.2_
  
  - [ ]* 6.5 Write property tests for file processing
    - **Property 16: Valid file queuing**
    - **Property 18: Upload metadata storage**
    - **Property 38: Request processing**
    - **Property 40: Status update after processing**
    - **Validates: Requirements 5.2, 5.5, 11.3, 11.5**

- [x] 7. Implement farmer dashboard and features




  - [x] 7.1 Create farmer authentication routes


    - Implement /farmer/register route with validation
    - Create /farmer/login route with session creation
    - Add /farmer/logout route
    - Implement redirect logic after successful operations
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.2, 3.3_
  
  - [x] 7.2 Create farmer dashboard route


    - Implement /farmer/dashboard with session validation
    - Display farmer name and basic info
    - Show upload form for hyperspectral images
    - Add link to detection history
    - _Requirements: 3.2_
  
  - [x] 7.3 Create file upload route


    - Implement /farmer/upload endpoint
    - Validate uploaded file
    - Add to processing queue
    - Return upload confirmation with request_id
    - _Requirements: 5.1, 5.2_
  
  - [x] 7.4 Create detection history route


    - Implement /farmer/history endpoint
    - Retrieve farmer's files using FileService
    - Sort by timestamp using merge sort (descending)
    - Display filename, date, result, confidence score
    - Implement pagination for large result sets
    - Add prefix search functionality using trie
    - _Requirements: 6.1, 6.2, 6.3, 6.5_
  
  - [ ]* 7.5 Write property tests for farmer features
    - **Property 4: Successful farmer registration redirect**
    - **Property 11: Session expiration redirect**
    - **Property 19: Farmer history isolation**
    - **Property 20: History timestamp sorting**
    - **Property 21: History record completeness**
    - **Property 22: Filename prefix search**
    - **Validates: Requirements 1.4, 3.5, 6.1, 6.2, 6.3, 6.5**

- [x] 8. Implement admin user management features




  - [x] 8.1 Create admin authentication routes


    - Implement /admin/register route with special code validation
    - Create /admin/login route with privileged session creation
    - Add admin session logging
    - Implement redirect logic
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 4.2, 4.5_
  
  - [x] 8.2 Create admin dashboard route


    - Implement /admin/dashboard with admin session validation
    - Display summary statistics (total users, total files, recent activity)
    - Show navigation to user management, file monitoring, model info
    - _Requirements: 4.2_
  
  - [x] 8.3 Create user management route


    - Implement /admin/users endpoint
    - Retrieve all users using UserService
    - Display users in sortable table (code, name, email, date)
    - Implement quicksort for multi-column sorting
    - Add binary search for user lookup
    - Implement user search functionality
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [x] 8.4 Create user management action routes


    - Implement /admin/users/<id>/deactivate endpoint
    - Create /admin/users/<id>/delete endpoint with cascade deletion
    - Add /admin/users/<id>/details endpoint
    - Implement action logging for all operations
    - Add actions to undo stack
    - _Requirements: 10.1, 10.2, 10.3, 10.5_
  
  - [x] 8.5 Implement undo functionality


    - Create /admin/undo endpoint
    - Pop last action from stack
    - Reverse the action (reactivate user, restore deleted user)
    - Update action log
    - _Requirements: 10.4_
  
  - [ ]* 8.6 Write property tests for user management
    - **Property 8: Successful admin registration redirect**
    - **Property 14: Admin session logging**
    - **Property 23: Complete user retrieval**
    - **Property 24: User sorting correctness**
    - **Property 25: User display completeness**
    - **Property 26: Multi-column sorting correctness**
    - **Property 33: Account deactivation**
    - **Property 34: Cascade deletion**
    - **Property 35: Undo functionality**
    - **Property 36: Action logging**
    - **Validates: Requirements 2.5, 4.5, 7.1, 7.2, 7.3, 7.4, 10.2, 10.3, 10.4, 10.5**

- [x] 9. Implement admin file monitoring features






  - [x] 9.1 Create file monitoring route

    - Implement /admin/files endpoint
    - Retrieve all files with farmer info using FileService
    - Display files in table (filename, uploader, date, result, confidence)
    - Implement filtering by status using hash table
    - Use AVL tree for efficient timestamp-based queries
    - _Requirements: 8.1, 8.3, 8.4_
  

  - [x] 9.2 Create file export route

    - Implement /admin/files/export endpoint
    - Use depth-first traversal of file tree to collect data
    - Generate CSV or JSON report
    - Include all file metadata
    - _Requirements: 8.5_
  
  - [ ]* 9.3 Write property tests for file monitoring
    - **Property 27: Complete file retrieval with farmer info**
    - **Property 28: File display completeness**
    - **Property 29: File status filtering**
    - **Property 30: File export completeness**
    - **Validates: Requirements 8.1, 8.3, 8.4, 8.5**

- [x] 10. Implement admin model info and statistics





  - [x] 10.1 Create model info route


    - Implement /admin/model endpoint
    - Display model architecture from ModelService.get_model_info()
    - Show model file path and size
    - Display device (CPU/GPU) being used
    - _Requirements: 9.1_
  
  - [x] 10.2 Create statistics calculation service


    - Implement StatisticsEngine class
    - Calculate total predictions from file_uploads table
    - Compute disease detection rate (diseased / total)
    - Calculate average confidence scores
    - Use circular buffer for recent prediction tracking
    - _Requirements: 9.2, 9.3, 9.4_
  
  - [x] 10.3 Create statistics display route


    - Implement /admin/statistics endpoint
    - Display aggregate statistics using StatisticsEngine
    - Show percentages and averages
    - Visualize trends using recent activity from circular buffer
    - _Requirements: 9.2, 9.4, 9.5_
  
  - [ ]* 10.4 Write property tests for statistics
    - **Property 31: Statistics calculation correctness**
    - **Property 32: Percentage and average correctness**
    - **Validates: Requirements 9.2, 9.4**

- [x] 11. Implement frontend templates and styling





  - [x] 11.1 Create base template and navigation


    - Create base.html with common layout
    - Add navigation bar with role-based menu items
    - Include CSS for styling (clean, professional design)
    - Add JavaScript for interactive elements
    - _Requirements: All UI requirements_
  
  - [x] 11.2 Create farmer templates


    - Create farmer_register.html with form validation
    - Create farmer_login.html
    - Create farmer_dashboard.html with upload form
    - Create farmer_history.html with sortable table and search
    - _Requirements: 1.1, 1.4, 3.2, 5.1, 6.1, 6.3_
  

  - [x] 11.3 Create admin templates

    - Create admin_register.html with special code field
    - Create admin_login.html
    - Create admin_dashboard.html with summary cards
    - Create admin_users.html with sortable table and actions
    - Create admin_files.html with filtering and export
    - Create admin_model.html with model info and statistics
    - _Requirements: 2.1, 2.5, 4.2, 7.3, 8.3, 9.1_
  
  - [x] 11.4 Add client-side validation and interactivity


    - Implement JavaScript form validation
    - Add AJAX for file upload with progress indicator
    - Implement real-time search filtering
    - Add sortable table headers with click handlers
    - _Requirements: 1.1, 2.1, 5.1, 6.5, 7.4_

- [x] 12. Implement data structure documentation generator




  - [x] 12.1 Create DocumentationGenerator class


    - Implement method to scan codebase for data structure usage
    - Extract data structure names, locations, and use cases
    - Collect time and space complexity information
    - _Requirements: 12.1, 12.2_
  
  - [x] 12.2 Implement documentation formatting


    - Create markdown formatter for DATA_STRUCTURES.md
    - Include executive summary section
    - Generate detailed catalog with code examples
    - Create data structure mapping table
    - Add performance analysis section
    - _Requirements: 12.2, 12.3, 12.4, 12.5_
  
  - [x] 12.3 Add code example extraction


    - Extract relevant code snippets for each data structure
    - Include usage examples from actual implementation
    - Add comments explaining the code
    - _Requirements: 12.3_
  
  - [x] 12.4 Add rationale and comparison documentation


    - Document why each data structure was chosen
    - Include performance comparisons where applicable
    - Explain trade-offs between alternatives
    - _Requirements: 12.4, 12.5_
  
  - [x] 12.5 Generate final documentation


    - Run DocumentationGenerator to create DATA_STRUCTURES.md
    - Verify all data structures are documented
    - Ensure completeness of all required sections
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [ ]* 12.6 Write property tests for documentation
    - **Property 41: Documentation completeness**
    - **Property 42: Code example inclusion**
    - **Property 43: Rationale documentation**
    - **Property 44: Performance comparison documentation**
    - **Validates: Requirements 12.2, 12.3, 12.4, 12.5**

- [x] 13. Implement security and error handling





  - [x] 13.1 Add input sanitization


    - Implement XSS prevention for all user inputs
    - Add CSRF token validation for forms
    - Sanitize filenames to prevent directory traversal
    - _Requirements: All input handling_
  
  - [x] 13.2 Implement comprehensive error handling


    - Add try-catch blocks for all database operations
    - Implement error logging with timestamps
    - Create user-friendly error pages
    - Add error recovery mechanisms
    - _Requirements: All operations_
  
  - [x] 13.3 Add access control middleware


    - Create decorator for farmer-only routes
    - Create decorator for admin-only routes
    - Implement session validation on all protected routes
    - Add automatic redirect for unauthorized access
    - _Requirements: 3.2, 4.2_

- [x] 14. Testing and quality assurance




  - [x] 14.1 Run all property-based tests

    - Execute all Hypothesis tests with 100+ iterations
    - Verify all 44 correctness properties pass
    - Fix any failing tests
    - _Requirements: All_
  
  - [ ]* 14.2 Run unit tests
    - Execute all unit tests for data structures
    - Run database operation tests
    - Test authentication and session management
    - _Requirements: All_
  
  - [ ]* 14.3 Perform integration testing
    - Test complete farmer workflow (register → login → upload → view history)
    - Test complete admin workflow (login → manage users → view files → view stats)
    - Test concurrent uploads with multiple farmers
    - Verify cascade deletion works correctly
    - _Requirements: All_
  
  - [x] 14.4 Checkpoint - Ensure all tests pass


    - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Deployment preparation and final verification
  - [ ] 15.1 Create requirements.txt
    - List all Python dependencies with versions
    - Include Flask, PyTorch, MySQL connector, bcrypt, Pillow, Hypothesis
    - _Requirements: All_
  
  - [ ] 15.2 Create setup and installation guide
    - Write README.md with installation instructions
    - Document database setup steps
    - Include configuration instructions
    - Add troubleshooting section
    - _Requirements: All_
  
  - [ ] 15.3 Verify model integration
    - Test model loading with best_model (2).pth
    - Verify inference works with sample hyperspectral images
    - Check GPU/CPU detection and usage
    - _Requirements: 5.3, 5.4_
  
  - [ ] 15.4 Verify database connectivity
    - Test connection to MySQL on localhost:3306
    - Verify credentials (root / "Hello! World")
    - Ensure all tables are created correctly
    - Test CRUD operations on all tables
    - _Requirements: All database operations_
  
  - [ ] 15.5 Final checkpoint - Complete system verification
    - Ensure all tests pass, ask the user if questions arise.
    - Verify DATA_STRUCTURES.md is generated and complete
    - Confirm all features are working end-to-end
    - Validate that all 12 requirements are fully implemented
