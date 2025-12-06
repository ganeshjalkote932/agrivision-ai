# Requirements Document

## Introduction

The Crop Disease Detection System is a multi-level web application that enables farmers to detect crop diseases using hyperspectral image analysis through a trained machine learning model, while providing administrators with comprehensive user management, file tracking, and system oversight capabilities. The system emphasizes the use of classic data structures (searching, sorting, trees, hash tables, queues, stacks) throughout its implementation to efficiently manage users, files, and processing workflows.

## Glossary

- **System**: The Crop Disease Detection Web Application
- **Administrator**: A privileged user with access to user management, file monitoring, and system configuration
- **Farmer**: A standard user who can upload hyperspectral images for disease detection
- **Hyperspectral Image**: A multi-band image file used for crop disease analysis
- **Disease Detection Model**: The pre-trained PyTorch model (best_model.pth) that analyzes images
- **Database**: MySQL database named "crop" running on localhost:3306
- **Administrator Table**: Database table storing administrator credentials and special codes
- **Farmer Table**: Database table storing farmer credentials
- **Data Structure**: Classic computer science structures (arrays, linked lists, trees, hash tables, stacks, queues, heaps) used for efficient data management
- **Detection Result**: Binary classification output indicating diseased or healthy crop status

## Requirements

### Requirement 1

**User Story:** As a farmer, I want to register an account with my credentials, so that I can access the disease detection system.

#### Acceptance Criteria

1. WHEN a farmer submits registration information (name, email, password), THE System SHALL validate the input fields for completeness and format correctness
2. WHEN registration data is valid, THE System SHALL generate a unique F_code using a hash-based data structure and store the farmer record in the Farmer table
3. WHEN a farmer attempts to register with an existing email, THE System SHALL reject the registration and display an appropriate error message
4. WHEN registration is successful, THE System SHALL redirect the farmer to the login page
5. THE System SHALL use a hash table data structure to check for duplicate email addresses in O(1) average time complexity

### Requirement 2

**User Story:** As an administrator, I want to register an account with a special code, so that I can access administrative functions securely.

#### Acceptance Criteria

1. WHEN an administrator submits registration information (name, email, password, special code), THE System SHALL validate all input fields including the special code format
2. WHEN the special code matches the predefined administrator authorization code, THE System SHALL generate a unique A_Code and store the administrator record in the Administrator table
3. WHEN an invalid special code is provided, THE System SHALL reject the registration and display an authentication error
4. THE System SHALL use a binary search tree to maintain sorted administrator records by A_Code for efficient retrieval
5. WHEN registration is successful, THE System SHALL redirect the administrator to the admin login page

### Requirement 3

**User Story:** As a farmer, I want to log in to my account, so that I can access the disease detection features.

#### Acceptance Criteria

1. WHEN a farmer enters email and password credentials, THE System SHALL retrieve the farmer record using a hash table lookup by email
2. WHEN credentials match the stored record, THE System SHALL create a session and redirect to the farmer dashboard
3. WHEN credentials do not match, THE System SHALL display an error message and maintain the current state
4. THE System SHALL use a hash-based session management data structure to track active farmer sessions
5. WHEN a farmer session expires, THE System SHALL redirect to the login page

### Requirement 4

**User Story:** As an administrator, I want to log in to my account, so that I can access administrative functions.

#### Acceptance Criteria

1. WHEN an administrator enters email and password credentials, THE System SHALL retrieve the administrator record using indexed database lookup
2. WHEN credentials match the stored record, THE System SHALL create an administrator session with elevated privileges
3. WHEN credentials do not match, THE System SHALL display an error message without revealing whether the email exists
4. THE System SHALL use a priority queue data structure to manage administrator session priorities
5. WHEN an administrator session is created, THE System SHALL log the access event with timestamp

### Requirement 5

**User Story:** As a farmer, I want to upload a hyperspectral image for disease detection, so that I can determine if my crops are diseased.

#### Acceptance Criteria

1. WHEN a farmer selects a hyperspectral image file, THE System SHALL validate the file format and size constraints
2. WHEN the file is valid, THE System SHALL add the processing request to a queue data structure for sequential processing
3. WHEN the model processes the image, THE System SHALL load the best_model.pth file and perform inference
4. WHEN inference completes, THE System SHALL return a binary result (diseased or healthy) with confidence score
5. THE System SHALL store the upload metadata (filename, timestamp, F_code, result) in a sorted data structure for retrieval

### Requirement 6

**User Story:** As a farmer, I want to view my disease detection history, so that I can track my previous submissions and results.

#### Acceptance Criteria

1. WHEN a farmer accesses the history page, THE System SHALL retrieve all detection records associated with the farmer's F_code
2. THE System SHALL use a merge sort algorithm to sort detection records by timestamp in descending order
3. WHEN displaying results, THE System SHALL show filename, upload date, detection result, and confidence score
4. THE System SHALL use pagination with a binary search tree to efficiently navigate large result sets
5. WHEN a farmer searches their history, THE System SHALL use a trie data structure for prefix-based filename search

### Requirement 7

**User Story:** As an administrator, I want to view all registered users, so that I can monitor system usage and manage accounts.

#### Acceptance Criteria

1. WHEN an administrator accesses the user management page, THE System SHALL retrieve all farmer and administrator records from the database
2. THE System SHALL use a heap data structure to sort users by registration date or activity level
3. WHEN displaying users, THE System SHALL show user code, name, email, and registration date in a sortable table
4. THE System SHALL implement quick sort algorithm to enable sorting by any column (name, email, date)
5. WHEN an administrator searches for a user, THE System SHALL use binary search on sorted user lists for O(log n) lookup

### Requirement 8

**User Story:** As an administrator, I want to view all uploaded files and their detection results, so that I can monitor system activity and model performance.

#### Acceptance Criteria

1. WHEN an administrator accesses the file monitoring page, THE System SHALL retrieve all file upload records with associated farmer information
2. THE System SHALL use a balanced AVL tree to maintain files sorted by upload timestamp for efficient range queries
3. WHEN displaying files, THE System SHALL show filename, uploader name, upload date, detection result, and confidence score
4. THE System SHALL implement filtering using a hash table to quickly find files by status (diseased/healthy)
5. WHEN an administrator exports data, THE System SHALL use depth-first traversal of the file tree structure to generate reports

### Requirement 9

**User Story:** As an administrator, I want to view model information and statistics, so that I can understand system performance and accuracy.

#### Acceptance Criteria

1. WHEN an administrator accesses the model info page, THE System SHALL display model architecture details, training metrics, and deployment information
2. THE System SHALL calculate and display aggregate statistics (total predictions, accuracy rate, disease detection rate) using accumulated data structures
3. THE System SHALL use a circular buffer data structure to maintain recent prediction history for performance trending
4. WHEN displaying statistics, THE System SHALL compute percentages and averages from stored detection results
5. THE System SHALL visualize detection trends using time-series data organized in a linked list structure

### Requirement 10

**User Story:** As an administrator, I want to manage user accounts (activate, deactivate, delete), so that I can maintain system security and data integrity.

#### Acceptance Criteria

1. WHEN an administrator selects a user account, THE System SHALL display account details and available management actions
2. WHEN an administrator deactivates an account, THE System SHALL update the user record and add the user to a blocked list using a hash set
3. WHEN an administrator deletes an account, THE System SHALL remove the user record and cascade delete associated files using graph traversal
4. THE System SHALL use a stack data structure to implement undo functionality for recent administrative actions
5. WHEN account changes are made, THE System SHALL log the action with administrator ID and timestamp in an append-only log structure

### Requirement 11

**User Story:** As a system, I want to efficiently process multiple concurrent image uploads, so that farmers experience minimal wait times.

#### Acceptance Criteria

1. WHEN multiple farmers upload images simultaneously, THE System SHALL add requests to a priority queue based on upload timestamp
2. THE System SHALL process queue items in FIFO order using a queue data structure
3. WHEN processing capacity is available, THE System SHALL dequeue the next request and invoke the disease detection model
4. THE System SHALL maintain a hash table of in-progress requests to prevent duplicate processing
5. WHEN processing completes, THE System SHALL update the request status and notify the farmer through the session

### Requirement 12

**User Story:** As a developer, I want comprehensive documentation of data structure usage, so that the implementation choices and their benefits are clearly understood.

#### Acceptance Criteria

1. THE System SHALL generate a documentation file listing all data structures used throughout the application
2. WHEN documenting each data structure, THE System SHALL specify the use case, time complexity, and space complexity
3. THE System SHALL include code examples showing how each data structure is implemented and utilized
4. THE System SHALL explain the rationale for choosing specific data structures for specific operations
5. THE System SHALL document performance comparisons between different data structure choices where applicable
