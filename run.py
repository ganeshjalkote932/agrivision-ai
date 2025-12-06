"""
Main entry point for the Crop Disease Detection System
"""
from app import create_app
from app.config import Config
from app.database import DatabaseManager

# Initialize configuration first
Config.init_app(None)

# Initialize database connection pool and create tables
try:
    DatabaseManager.initialize_pool()
    DatabaseManager.create_tables()
    print("✓ Database initialized successfully")
except Exception as e:
    print(f"✗ Error during database initialization: {e}")

# Create Flask app (this will also start the background worker)
app = create_app()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌾 Crop Disease Detection System")
    print("="*60)
    print("Server starting on http://0.0.0.0:5000")
    print("Press CTRL+C to quit")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
