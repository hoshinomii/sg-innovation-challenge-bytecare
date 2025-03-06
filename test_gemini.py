"""
Test script to verify that Google Gemini API is working correctly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check Python version
print(f"Python version: {sys.version}")

# Check if Gemini API key is set
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable is not set.")
    print("Please set it in your .env file or export it in your shell.")
    exit(1)

print(f"Gemini API key found: {api_key[:8]}...")

try:
    print("Attempting to import google package...")
    import google
    print(f"Google package version: {google.__version__ if hasattr(google, '__version__') else 'unknown'}")
    
    print("Attempting to import google.generativeai...")
    from google import genai
    print("Successfully imported google.generativeai")
    
    print("Configuring Gemini client with API key...")
    client = genai.Client(api_key=api_key)
    
    # Get available models
    print("Listing available models...")
    try:
        models = client.list_models()
        gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
        print(f"Available Gemini models: {gemini_models}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Test with simple prompt
    print("\nTesting Gemini model with a simple prompt...")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    print(f"Using model: {model_name}")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say hello world"
        )
        print("\nResponse from Gemini:")
        print("-" * 40)
        print(response.text)
        print("-" * 40)
        print("\nTest SUCCESSFUL! The Gemini API is working correctly.")
    except Exception as e:
        print(f"\nError testing model: {e}")
        print("\nTest FAILED! There was a problem with the Gemini API.")
        
except ImportError as e:
    print(f"ERROR: Failed to import required module: {e}")
    print("\nPlease install the package with: pip install --upgrade google-generativeai>=0.7.0")
    
    # Check if package is installed
    import subprocess
    try:
        print("\nChecking installed packages:")
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        packages = result.stdout.lower()
        
        if 'google-generativeai' in packages:
            print("google-generativeai is installed. This might be an import or version compatibility issue.")
        else:
            print("google-generativeai is NOT installed. Please install it.")
    except Exception as pip_error:
        print(f"Could not check installed packages: {pip_error}")
        
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)
