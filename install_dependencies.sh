#!/bin/bash

echo "Installing required dependencies for the ByteCare Inventory Management System"
echo "============================================================================"

# Install main dependencies
echo "Installing main dependencies..."
pip install -r requirements.txt

# Specifically install Google Generative AI package
echo "Ensuring Google Generative AI package is properly installed..."
pip install --force-reinstall --no-cache-dir google-generativeai>=0.7.0

# Verify installation
echo "Verifying installations..."
if python -c "from google import genai; print('Google Generative AI client package successfully imported')"; then
    echo "✓ Google Generative AI package is installed correctly"
else
    echo "✗ Failed to import Google Generative AI package"
    echo "Trying to fix by installing individually..."
    pip install --upgrade pip
    pip install --upgrade setuptools wheel
    pip install --upgrade google-api-python-client
    pip install --upgrade google-generativeai
    
    if python -c "from google import genai; print('Test import successful')"; then
        echo "✓ Fixed the issue with Google Generative AI package"
    else
        echo "✗ Still having issues with Google Generative AI package"
        echo "Please check your Python environment and try again."
        exit 1
    fi
fi

python -c "import pandas; import numpy; import matplotlib; import seaborn; import sklearn; print('Core data science packages successfully imported')"

# Create project directories
echo "Creating project directories..."
python create_directories.py

echo "Installation complete. Please run the test_gemini.py script to verify the API connection."
echo "python test_gemini.py"
