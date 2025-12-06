# Design Document

## Overview

The Crop Disease Detection System is a Flask-based web application that provides role-based access for Administrators and Farmers. The system integrates a pre-trained PyTorch model for hyperspectral image analysis and emphasizes the use of classic data structures throughout the implementation to achieve optimal performance for searching, sorting, and data management operations.

The architecture follows a three-tier model:
1. **Presentation Layer**: HTML/CSS/JavaScript frontend with role-specific dashboards
2. **Application Layer**: Flask backend with data structure implementations for efficient operations
3. **Data Layer**: MySQL database with in-memory data structure caching

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Browser (Client)                     │
│              (Farmer Dashboard / Admin Dashboard)            │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Web Server                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Routes     │  │  Auth Layer  │  │  Session Mgr │      │
│  │  (Endpoints) │  │  (Hash Table)│  │  (Hash Map)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Data Structure Layer                         │   │
│  │  • UserHashTable  • FileAVLTree  • ProcessQueue     │   │
│  │  • SearchTrie     • SortingEngine • CacheHeap       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Business Logic Layer                         │   │
│  │  • UserService  • FileService  • ModelService       │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────┬───────────────────┘
                        │                 │
                        ▼                 ▼
            ┌───────────────────┐  ┌──────────────────┐
            │  MySQL Database   │  │  PyTorch Model   │
            │  (crop database)  │  │ (best_model.pth) │
            └───────────────────┘  └──────────────────┘
```

### Technology Stack

- **Backend**: Python 3.8+ with Flask framework
- **Frontend**: HTML5, CSS3, JavaScript (vanilla or lightweight framework)
- **Database**: MySQL 8.0+ (localhost:3306, user: root, password: "Hello! World")
- **ML Framework**: PyTorch for model inference
- **Image Processing**: PIL/Pillow for image handling
- **Session Management**: Flask-Session with server-side storage
- **Data Structures**: Custom implementations in Python

## Components and Interfaces

### 1. Authentication Module

**Purpose**: Handle user registration and login for both Farmers and Administrators

**Components**:
- `AuthService`: Manages authentication logic
- `UserHashTable`: Custom hash table for O(1) user lookup by email
- `PasswordHasher`: Secure password hashing using bcrypt

**Interfaces**:
```python
class AuthService:
    def register_farmer(name: str, email: str, password: str) -> Result[int, Error]
    def register_admin(name: str, email: str, password: str, special_code: str) -> Result[int, Error]
    def login_farmer(email: str, password: str) -> Result[Session, Error]
    def login_admin(email: str, password: str) -> Result[Session, Error]
    def logout(session_id: str) -> bool

class UserHashTable:
    def insert(email: str, user_data: dict) -> None
    def lookup(email: str) -> Optional[dict]
    def exists(email: str) -> bool
    def delete(email: str) -> bool
```

### 2. User Management Module (Admin)

**Purpose**: Enable administrators to view, search, sort, and manage user accounts

**Components**:
- `UserService`: Business logic for user operations
- `UserBST`: Binary Search Tree for sorted user storage by A_Code/F_code
- `QuickSortEngine`: Implements quicksort for multi-column sorting
- `BinarySearchUtil`: Efficient user search in sorted lists

**Interfaces**:
```python
class UserService:
    def get_all_users() -> List[dict]
    def search_user(query: str) -> List[dict]
    def sort_users(field: str, order: str) -> List[dict]
    def deactivate_user(user_code: int, user_type: str) -> bool
    def delete_user(user_code: int, user_type: str) -> bool
    def get_user_statistics() -> dict

class UserBST:
    def insert(user_code: int, user_data: dict) -> None
    def search(user_code: int) -> Optional[dict]
    def inorder_traversal() -> List[dict]
    def delete(user_code: int) -> bool
```

### 3. Disease Detection Module

**Purpose**: Process hyperspectral images using the trained model

**Components**:
- `ModelService`: Loads and manages the PyTorch model
- `ProcessingQueue`: FIFO queue for image processing requests
- `ImagePreprocessor`: Prepares images for model input
- `ResultCache`: LRU cache using heap for recent predictions

**Interfaces**:
```python
class ModelService:
    def load_model(model_path: str) -> None
    def predict(image_path: str) -> Result[dict, Error]
    def get_model_info() -> dict

class ProcessingQueue:
    def enqueue(request: ProcessRequest) -> str  # Returns request_id
    def dequeue() -> Optional[ProcessRequest]
    def get_status(request_id: str) -> str
    def is_empty() -> bool
    def size() -> int

class ProcessRequest:
    request_id: str
    farmer_code: int
    image_path: str
    timestamp: datetime
    status: str  # 'pending', 'processing', 'completed', 'failed'
```

### 4. File Management Module

**Purpose**: Track and manage uploaded files with efficient retrieval

**Components**:
- `FileService`: Business logic for file operations
- `FileAVLTree`: Self-balancing AVL tree for timestamp-based file storage
- `FileHashTable`: Hash table for quick file lookup by filename
- `MergeSortEngine`: Implements merge sort for file history

**Interfaces**:
```python
class FileService:
    def save_file_metadata(filename: str, farmer_code: int, result: dict) -> int
    def get_farmer_files(farmer_code: int) -> List[dict]
    def get_all_files() -> List[dict]
    def filter_files(status: str) -> List[dict]
    def search_files(query: str) -> List[dict]
    def delete_file(file_id: int) -> bool

class FileAVLTree:
    def insert(timestamp: datetime, file_data: dict) -> None
    def range_query(start: datetime, end: datetime) -> List[dict]
    def inorder_traversal() -> List[dict]
    def delete(timestamp: datetime) -> bool
    def get_height() -> int
```

### 5. Search Module

**Purpose**: Provide efficient search capabilities across users and files

**Components**:
- `SearchService`: Unified search interface
- `Trie`: Prefix tree for autocomplete and prefix search
- `SearchIndex`: Inverted index for full-text search

**Interfaces**:
```python
class SearchService:
    def search_users(query: str) -> List[dict]
    def search_files(query: str) -> List[dict]
    def autocomplete(prefix: str) -> List[str]

class Trie:
    def insert(word: str, data: dict) -> None
    def search(word: str) -> Optional[dict]
    def starts_with(prefix: str) -> List[dict]
    def delete(word: str) -> bool
```

### 6. Admin Dashboard Module

**Purpose**: Provide comprehensive administrative interface

**Components**:
- `DashboardService`: Aggregates data for admin views
- `StatisticsEngine`: Computes metrics using efficient data structures
- `CircularBuffer`: Maintains recent activity log
- `ActionStack`: Stack for undo functionality

**Interfaces**:
```python
class DashboardService:
    def get_dashboard_data() -> dict
    def get_user_statistics() -> dict
    def get_file_statistics() -> dict
    def get_model_statistics() -> dict
    def get_recent_activity(limit: int) -> List[dict]

class ActionStack:
    def push(action: Action) -> None
    def pop() -> Optional[Action]
    def peek() -> Optional[Action]
    def undo_last_action() -> bool
```

### 7. Session Management Module

**Purpose**: Manage user sessions with role-based access control

**Components**:
- `SessionManager`: Handles session lifecycle
- `SessionHashMap`: Hash map for O(1) session lookup
- `PriorityQueue`: Manages admin session priorities

**Interfaces**:
```python
class SessionManager:
    def create_session(user_code: int, user_type: str) -> str
    def get_session(session_id: str) -> Optional[dict]
    def validate_session(session_id: str) -> bool
    def destroy_session(session_id: str) -> bool
    def cleanup_expired_sessions() -> int
```

## Data Models

### Database Schema

**Administrator Table**:
```sql
CREATE TABLE Administrator (
    A_Code INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    Special_Code VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

**Farmer Table**:
```sql
CREATE TABLE farmer (
    F_code INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

**File Upload Table** (New):
```sql
CREATE TABLE file_uploads (
    file_id INT PRIMARY KEY AUTO_INCREMENT,
    F_code INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    upload_timestamp DATETIME NOT NULL,
    detection_result VARCHAR(50),
    confidence_score FLOAT,
    processing_status VARCHAR(50),
    FOREIGN KEY (F_code) REFERENCES farmer(F_code) ON DELETE CASCADE
);
```

**Admin Actions Log Table** (New):
```sql
CREATE TABLE admin_actions (
    action_id INT PRIMARY KEY AUTO_INCREMENT,
    A_Code INT NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    target_user_code INT,
    target_user_type VARCHAR(20),
    action_details TEXT,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (A_Code) REFERENCES Administrator(A_Code)
);
```

### In-Memory Data Structures

**User Hash Table Structure**:
```python
class UserHashTable:
    def __init__(self, size=1000):
        self.size = size
        self.table = [[] for _ in range(size)]  # Chaining for collision resolution
    
    def _hash(self, email: str) -> int:
        return hash(email) % self.size
```

**File AVL Tree Node**:
```python
class AVLNode:
    def __init__(self, timestamp, file_data):
        self.timestamp = timestamp
        self.file_data = file_data
        self.left = None
        self.right = None
        self.height = 1
```

**Processing Queue Structure**:
```python
class ProcessingQueue:
    def __init__(self):
        self.queue = deque()  # Using collections.deque for O(1) operations
        self.in_progress = {}  # Hash table for tracking
```

**Trie Node Structure**:
```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Hash map of character -> TrieNode
        self.is_end_of_word = False
        self.data = None  # Associated data
```

## Co
rrectness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Registration and Authentication Properties

**Property 1: Registration input validation**
*For any* farmer registration attempt with input fields (name, email, password), the validation function should accept the registration if and only if all fields are non-empty, email matches valid format, and password meets strength requirements.
**Validates: Requirements 1.1**

**Property 2: Unique farmer code generation**
*For any* set of farmer registrations, all generated F_codes should be unique (no two farmers should have the same F_code).
**Validates: Requirements 1.2**

**Property 3: Duplicate email rejection**
*For any* farmer registration attempt with an email that already exists in the system, the registration should be rejected with an appropriate error message.
**Validates: Requirements 1.3**

**Property 4: Successful farmer registration redirect**
*For any* valid farmer registration, the system should redirect to the login page upon successful completion.
**Validates: Requirements 1.4**

**Property 5: Admin registration validation**
*For any* administrator registration attempt, the validation should accept if and only if all fields are valid and the special code matches the predefined authorization code.
**Validates: Requirements 2.1**

**Property 6: Unique admin code generation**
*For any* set of administrator registrations with valid special codes, all generated A_Codes should be unique.
**Validates: Requirements 2.2**

**Property 7: Invalid special code rejection**
*For any* administrator registration attempt with an invalid special code, the registration should be rejected with an authentication error.
**Validates: Requirements 2.3**

**Property 8: Successful admin registration redirect**
*For any* valid administrator registration, the system should redirect to the admin login page upon successful completion.
**Validates: Requirements 2.5**

**Property 9: Valid farmer login creates session**
*For any* registered farmer with correct email and password credentials, login should create a valid session and redirect to the farmer dashboard.
**Validates: Requirements 3.2**

**Property 10: Invalid farmer credentials rejection**
*For any* login attempt with incorrect email or password, the system should reject the login, display an error message, and maintain the current state without creating a session.
**Validates: Requirements 3.3**

**Property 11: Session expiration redirect**
*For any* expired farmer session, attempts to access protected pages should redirect to the login page.
**Validates: Requirements 3.5**

**Property 12: Valid admin login creates privileged session**
*For any* registered administrator with correct credentials, login should create a session with elevated privileges.
**Validates: Requirements 4.2**

**Property 13: Admin login error message security**
*For any* invalid administrator login attempt, the error message should not reveal whether the email exists in the system.
**Validates: Requirements 4.3**

**Property 14: Admin session logging**
*For any* successful administrator login, the system should create a log entry with the A_Code and timestamp.
**Validates: Requirements 4.5**

### Disease Detection Properties

**Property 15: Image file validation**
*For any* uploaded file, the validation should accept if and only if the file format is a valid image format and the file size is within the specified constraints.
**Validates: Requirements 5.1**

**Property 16: Valid file queuing**
*For any* valid image file upload, the system should add a processing request to the queue with the correct farmer F_code and timestamp.
**Validates: Requirements 5.2**

**Property 17: Inference output format**
*For any* completed image inference, the result should contain a binary classification (diseased or healthy) and a confidence score between 0 and 1.
**Validates: Requirements 5.4**

**Property 18: Upload metadata storage**
*For any* processed image, the stored metadata should include filename, timestamp, F_code, detection result, and confidence score, and retrieving it should return the same values.
**Validates: Requirements 5.5**

### File and History Management Properties

**Property 19: Farmer history isolation**
*For any* farmer accessing their detection history, the returned records should contain only uploads associated with that farmer's F_code and no other farmer's uploads.
**Validates: Requirements 6.1**

**Property 20: History timestamp sorting**
*For any* farmer's detection history, the records should be sorted by timestamp in descending order (most recent first).
**Validates: Requirements 6.2**

**Property 21: History record completeness**
*For any* detection record in a farmer's history, the display should include filename, upload date, detection result, and confidence score.
**Validates: Requirements 6.3**

**Property 22: Filename prefix search**
*For any* prefix string and farmer's file history, the search should return all and only those files whose filenames start with the given prefix.
**Validates: Requirements 6.5**

### Admin User Management Properties

**Property 23: Complete user retrieval**
*For any* administrator accessing the user management page, the system should retrieve all farmer and administrator records from the database with no omissions.
**Validates: Requirements 7.1**

**Property 24: User sorting correctness**
*For any* user list sorted by registration date, the users should appear in chronological order (earliest to latest or vice versa based on sort direction).
**Validates: Requirements 7.2**

**Property 25: User display completeness**
*For any* user record displayed in the admin interface, the display should include user code, name, email, and registration date.
**Validates: Requirements 7.3**

**Property 26: Multi-column sorting correctness**
*For any* user list sorted by a specific column (name, email, or date), the resulting list should be correctly ordered according to that column's values.
**Validates: Requirements 7.4**

### Admin File Monitoring Properties

**Property 27: Complete file retrieval with farmer info**
*For any* administrator accessing the file monitoring page, the system should retrieve all file upload records with associated farmer information (name, email) correctly joined.
**Validates: Requirements 8.1**

**Property 28: File display completeness**
*For any* file record displayed in the admin interface, the display should include filename, uploader name, upload date, detection result, and confidence score.
**Validates: Requirements 8.3**

**Property 29: File status filtering**
*For any* filter by status (diseased or healthy), the returned files should contain only files with the matching detection result.
**Validates: Requirements 8.4**

**Property 30: File export completeness**
*For any* data export operation, the generated report should contain all file records with complete information and no omissions.
**Validates: Requirements 8.5**

### Statistics and Model Info Properties

**Property 31: Statistics calculation correctness**
*For any* set of detection results, the calculated aggregate statistics (total predictions, accuracy rate, disease detection rate) should match the mathematical computation from the raw data.
**Validates: Requirements 9.2**

**Property 32: Percentage and average correctness**
*For any* set of detection results, the computed percentages and averages should be mathematically correct (sum of percentages = 100%, average = sum/count).
**Validates: Requirements 9.4**

### Admin Account Management Properties

**Property 33: Account deactivation**
*For any* user account deactivated by an administrator, the user record should have is_active set to false and the user should be unable to log in.
**Validates: Requirements 10.2**

**Property 34: Cascade deletion**
*For any* user account deleted by an administrator, all associated file upload records should also be deleted from the system.
**Validates: Requirements 10.3**

**Property 35: Undo functionality**
*For any* administrative action that is undone, the system state should be restored to the state immediately before that action was performed.
**Validates: Requirements 10.4**

**Property 36: Action logging**
*For any* account management action (deactivate, delete, reactivate), the system should create a log entry with the administrator's A_Code, action type, target user, and timestamp.
**Validates: Requirements 10.5**

### Concurrent Processing Properties

**Property 37: Request ordering**
*For any* sequence of image upload requests, the requests should be processed in the order they were added to the queue (FIFO).
**Validates: Requirements 11.1, 11.2**

**Property 38: Request processing**
*For any* request in the processing queue, when processing capacity is available, the request should be dequeued and the model should be invoked.
**Validates: Requirements 11.3**

**Property 39: Duplicate processing prevention**
*For any* image processing request that is currently in progress, attempting to process the same request again should be prevented.
**Validates: Requirements 11.4**

**Property 40: Status update after processing**
*For any* completed image processing request, the request status should be updated to 'completed' and the result should be stored.
**Validates: Requirements 11.5**

### Documentation Properties

**Property 41: Documentation completeness**
*For any* data structure used in the system, the documentation should include the data structure name, use case, time complexity, and space complexity.
**Validates: Requirements 12.2**

**Property 42: Code example inclusion**
*For any* documented data structure, the documentation should include at least one code example showing its implementation or usage.
**Validates: Requirements 12.3**

**Property 43: Rationale documentation**
*For any* data structure choice, the documentation should explain why that specific data structure was chosen for its use case.
**Validates: Requirements 12.4**

**Property 44: Performance comparison documentation**
*For any* data structure where alternative choices were considered, the documentation should include performance comparisons between the options.
**Validates: Requirements 12.5**

## Error Handling

### Error Categories

1. **Validation Errors**: Invalid input data (empty fields, malformed emails, weak passwords)
   - Return HTTP 400 with descriptive error messages
   - Maintain form state for user correction

2. **Authentication Errors**: Invalid credentials, expired sessions, unauthorized access
   - Return HTTP 401 for authentication failures
   - Return HTTP 403 for authorization failures
   - Redirect to appropriate login page

3. **Database Errors**: Connection failures, query errors, constraint violations
   - Log detailed error information
   - Return HTTP 500 with generic user message
   - Implement retry logic for transient failures

4. **File Processing Errors**: Invalid file format, corrupted files, model inference failures
   - Return HTTP 400 for invalid files
   - Return HTTP 500 for model errors
   - Store error status in file_uploads table

5. **Resource Not Found**: User not found, file not found
   - Return HTTP 404 with appropriate message

### Error Handling Strategy

**Database Connection Pool**:
- Implement connection pooling with automatic retry
- Maximum 3 retry attempts with exponential backoff
- Graceful degradation if database is unavailable

**Model Loading**:
- Load model once at application startup
- Cache model in memory for reuse
- Implement fallback mechanism if model fails to load

**File Upload**:
- Validate file before saving to disk
- Use temporary storage during processing
- Clean up temporary files after processing or on error

**Session Management**:
- Implement session timeout (30 minutes for farmers, 60 minutes for admins)
- Automatic session cleanup for expired sessions
- Secure session storage with encryption

## Testing Strategy

### Unit Testing

The system will include unit tests for individual components and functions:

**Authentication Tests**:
- Test password hashing and verification
- Test email validation regex
- Test special code validation
- Test session creation and validation

**Data Structure Tests**:
- Test hash table operations (insert, lookup, delete)
- Test AVL tree balancing and rotations
- Test queue enqueue/dequeue operations
- Test trie insert and prefix search
- Test sorting algorithms (quicksort, merge sort)

**Model Integration Tests**:
- Test model loading
- Test image preprocessing
- Test inference with sample images
- Test result formatting

**Database Tests**:
- Test CRUD operations for all tables
- Test foreign key constraints
- Test cascade deletion
- Test transaction rollback

### Property-Based Testing

The system will use **Hypothesis** (Python property-based testing library) to verify universal properties:

**Testing Framework**: Hypothesis for Python
- Minimum 100 iterations per property test
- Each property test tagged with format: `# Feature: crop-disease-detection-system, Property {number}: {property_text}`
- Each correctness property implemented by a SINGLE property-based test

**Property Test Categories**:

1. **Registration Properties** (Properties 1-8):
   - Generate random valid/invalid user data
   - Verify uniqueness constraints
   - Verify validation logic

2. **Authentication Properties** (Properties 9-14):
   - Generate random credentials
   - Verify session creation/destruction
   - Verify error message security

3. **File Processing Properties** (Properties 15-18):
   - Generate random file metadata
   - Verify queue ordering
   - Verify result format

4. **Search and Sort Properties** (Properties 19-26):
   - Generate random datasets
   - Verify sorting correctness
   - Verify search result accuracy

5. **Admin Operations Properties** (Properties 27-36):
   - Generate random admin actions
   - Verify cascade effects
   - Verify logging completeness

6. **Concurrent Processing Properties** (Properties 37-40):
   - Generate concurrent requests
   - Verify FIFO ordering
   - Verify duplicate prevention

7. **Documentation Properties** (Properties 41-44):
   - Verify documentation structure
   - Verify completeness of entries

### Integration Testing

- End-to-end farmer workflow: register → login → upload → view results
- End-to-end admin workflow: login → view users → manage accounts → view statistics
- Concurrent upload handling with multiple farmers
- Database transaction integrity under load

### Performance Testing

- Load testing with 100+ concurrent users
- Stress testing queue processing with 1000+ images
- Memory profiling for data structure efficiency
- Database query optimization verification

## Data Structure Usage Documentation

The system will automatically generate a comprehensive document (`DATA_STRUCTURES.md`) that includes:

### Documentation Structure

1. **Executive Summary**
   - Overview of data structures used
   - Performance characteristics summary
   - Design philosophy

2. **Detailed Data Structure Catalog**

For each data structure:
- **Name**: Hash Table, AVL Tree, Queue, etc.
- **Location**: Module and class name
- **Use Case**: Specific problem it solves
- **Implementation**: Code snippet
- **Time Complexity**: Big-O for all operations
- **Space Complexity**: Memory usage
- **Rationale**: Why this structure was chosen
- **Alternatives Considered**: Other options and why they were rejected
- **Performance Comparison**: Benchmarks if applicable

3. **Data Structure Mapping**

| Component | Data Structure | Primary Operations | Time Complexity |
|-----------|---------------|-------------------|-----------------|
| User Lookup | Hash Table | Insert, Search | O(1) average |
| User Sorting | Binary Search Tree | Insert, Inorder | O(log n) |
| File Timeline | AVL Tree | Insert, Range Query | O(log n) |
| Image Processing | Queue | Enqueue, Dequeue | O(1) |
| Filename Search | Trie | Insert, Prefix Search | O(m) where m=length |
| User Search | Binary Search | Search | O(log n) |
| File Sorting | Merge Sort | Sort | O(n log n) |
| User Sorting | Quick Sort | Sort | O(n log n) average |
| Recent Activity | Circular Buffer | Insert, Read | O(1) |
| Admin Undo | Stack | Push, Pop | O(1) |
| Session Management | Hash Map | Insert, Lookup | O(1) average |

4. **Performance Analysis**
   - Benchmarks for critical operations
   - Comparison with alternative approaches
   - Scalability considerations

5. **Code Examples**
   - Complete implementation examples
   - Usage patterns
   - Best practices

## Deployment Considerations

### System Requirements

- Python 3.8 or higher
- MySQL 8.0 or higher
- 4GB RAM minimum (8GB recommended for model inference)
- 10GB disk space for uploaded images
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Configuration

**Database Configuration**:
```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'Hello! World',
    'database': 'crop'
}
```

**Model Configuration**:
```python
MODEL_CONFIG = {
    'model_path': 'best_model (2).pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 1
}
```

**Application Configuration**:
```python
APP_CONFIG = {
    'secret_key': 'generate-secure-random-key',
    'session_timeout': 1800,  # 30 minutes for farmers
    'admin_session_timeout': 3600,  # 60 minutes for admins
    'max_upload_size': 10 * 1024 * 1024,  # 10MB
    'allowed_extensions': ['jpg', 'jpeg', 'png', 'tif', 'tiff'],
    'admin_special_code': 'ADMIN2024SECURE'
}
```

### Security Considerations

1. **Password Security**:
   - Use bcrypt for password hashing
   - Minimum password length: 8 characters
   - Require mix of uppercase, lowercase, numbers

2. **Session Security**:
   - Use secure, httponly cookies
   - Implement CSRF protection
   - Regenerate session ID after login

3. **File Upload Security**:
   - Validate file types using magic numbers, not just extensions
   - Scan uploaded files for malware
   - Store files outside web root
   - Use unique filenames to prevent overwrites

4. **SQL Injection Prevention**:
   - Use parameterized queries exclusively
   - Never concatenate user input into SQL

5. **Access Control**:
   - Verify user role on every protected route
   - Implement proper authorization checks
   - Log all administrative actions

### Scalability Considerations

1. **Database Optimization**:
   - Index frequently queried columns (email, F_code, A_Code)
   - Implement database connection pooling
   - Consider read replicas for heavy read operations

2. **Caching Strategy**:
   - Cache user data in hash tables
   - Cache model in memory
   - Implement Redis for distributed caching if needed

3. **Asynchronous Processing**:
   - Use Celery for background image processing
   - Implement job queue for long-running tasks
   - Provide real-time status updates via WebSockets

4. **Load Balancing**:
   - Deploy multiple application instances
   - Use nginx for load balancing
   - Implement sticky sessions for session management

## Technology Choices Rationale

### Flask Framework
- Lightweight and flexible
- Excellent for ML model integration
- Easy to implement custom data structures
- Strong community support

### MySQL Database
- Reliable and mature
- Good performance for relational data
- Supports complex queries and joins
- Already specified by user requirements

### PyTorch for Model
- Industry standard for deep learning
- Efficient inference
- Good documentation
- Pre-trained model already available

### Hypothesis for Property Testing
- Mature Python property-based testing library
- Excellent for generating test cases
- Good integration with pytest
- Supports complex data generation strategies

### Custom Data Structure Implementations
- Educational value (demonstrates understanding)
- Performance optimization opportunities
- Full control over behavior
- Meets project requirement for data structure emphasis
