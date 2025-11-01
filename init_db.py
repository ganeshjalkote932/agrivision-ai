#!/usr/bin/env python3
"""
Database initialization script for production deployment
"""

import os
import psycopg2
from werkzeug.security import generate_password_hash

def init_production_database():
    """Initialize PostgreSQL database for production."""
    
    # Get database URL from environment
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("No DATABASE_URL found, skipping database initialization")
        return
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(database_url, sslmode='require')
    cur = conn.cursor()
    
    # Create users table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create analysis_history table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            filename VARCHAR(255) NOT NULL,
            file_type VARCHAR(10) NOT NULL,
            total_samples INTEGER,
            diseased_samples INTEGER,
            healthy_samples INTEGER,
            avg_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if admin user exists
    cur.execute('SELECT id FROM users WHERE role = %s', ('administrator',))
    admin_exists = cur.fetchone()
    
    if not admin_exists:
        # Create default admin user
        admin_password = generate_password_hash('AgriVision2024!')
        cur.execute('''
            INSERT INTO users (username, email, password_hash, role)
            VALUES (%s, %s, %s, %s)
        ''', ('admin', 'admin@agrivision.ai', admin_password, 'administrator'))
        print("✅ Default admin user created")
        print("   Username: admin")
        print("   Password: AgriVision2024!")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("✅ Database initialized successfully")

if __name__ == '__main__':
    init_production_database()