# Crop Disease Detection System

A multi-level web application for detecting crop diseases using hyperspectral image analysis with machine learning.

## Features

- **Farmer Portal**: Upload hyperspectral images for disease detection
- **Admin Portal**: User management, file monitoring, and system statistics
- **ML Integration**: PyTorch-based disease detection model
- **Efficient Data Structures**: Custom implementations for optimal performance

## Prerequisites

- Python 3.8 or higher
- MySQL 8.0 or higher
- 4GB RAM minimum (8GB recommended)
- 10GB disk space for uploaded images

## Installation

1. Clone the repository and navigate to the project directory

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MySQL database:
```sql
CREATE DATABASE crop;
```

4. Configure database credentials in `app/config.py` or set environment variables:
```bash
export DB_HOST=localhost
export DB_PORT=3306
export DB_USER=root
export DB_PASSWORD="Hello! World"
export DB_NAME=crop
```

5. Ensure the model file `best_model (2).pth` is in the project root directory

## Running the Application

1. Start the Flask application:
```bash
python run.py
```

2. Access the application at `http://localhost:5000`

## Project Structure

```
crop-disease-detection-system/
├── app/
│   ├── __init__.py          # Application factory
│   ├── config.py            # Configuration management
│   ├── database.py          # Database connection manager
│   ├── auth/                # Authentication blueprint
│   ├── farmer/              # Farmer blueprint
│   ├── admin/               # Admin blueprint
│   └── templates/           # HTML templates
├── static/
│   ├── css/                 # Stylesheets
│   └── js/                  # JavaScript files
├── uploads/                 # Uploaded images (auto-created)
├── models/                  # Model files (auto-created)
├── run.py                   # Application entry point
└── requirements.txt         # Python dependencies
```

## Database Configuration

The application uses MySQL with connection pooling for optimal performance:
- Pool size: 5 connections
- Max overflow: 10 connections
- Connection timeout: 30 seconds
- Connection recycle: 1 hour

## Security Notes

- Change the `SECRET_KEY` in production
- Update `ADMIN_SPECIAL_CODE` for admin registration
- Use environment variables for sensitive configuration
- Enable HTTPS in production

## Development

The application is structured using Flask blueprints for modularity:
- `auth`: Authentication and session management
- `farmer`: Farmer-specific features
- `admin`: Administrator features

## Testing

Run property-based tests with Hypothesis:
```bash
pytest tests/
```

## License

Copyright © 2024 Crop Disease Detection System
