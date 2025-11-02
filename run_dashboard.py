import sys
import os

# This adds your project root to the Python path
# It allows `frontend.app` to correctly import `src`
sys.path.append(os.path.dirname(__file__))

try:
    # Import the 'app' object from frontend/app.py
    from frontend.app import app
except ImportError as e:
    print(f"Error: Could not import the app from the 'frontend' package.")
    print(f"Details: {e}")
    print("Please ensure your file structure is correct and all dependencies are installed.")
    sys.exit(1)

# This is the entry point to run your Dash server
if __name__ == "__main__":
    print("Starting Dash server...")
    print(f"Dashboard will be running at http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)