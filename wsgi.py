from app import create_app
from app.config import Config

# Initialize the Flask app
Config.init_app(None)
app = create_app()

if __name__ == "__main__":
    app.run()
