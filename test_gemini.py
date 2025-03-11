"""
Test script to verify that Google Gemini API is working correctly
"""

import os
import pandas as pd
from ai_insights import AIInsightGenerator
from dotenv import load_dotenv
import argparse
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

def test_gemini_connection(output_dir="dist/reports"):
    """Test connection to Google's Gemini API"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Testing connection to Google Gemini API...")
    
    # Check if API key is available
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return False
    
    # Initialize the insight generator with Gemini
    insight_generator = AIInsightGenerator(
        gemini_api_key=gemini_api_key,
        provider="gemini"
    )
    
    if insight_generator.provider != "gemini":
        print(f"Failed to initialize Gemini provider. Current provider: {insight_generator.provider}")
        return False
    
    # Create a simple test dataframe
    test_df = pd.DataFrame({
        'Description': ['Test Product A', 'Test Product B', 'Test Product C'],
        'recommended_restock': [100, 50, 25],
        'safety_stock': [20, 10, 5],
        'predicted_demand': [80, 40, 20]
    })
    test_df.index = ['A001', 'B002', 'C003']
    
    # Generate a simple insight
    try:
        print("Generating a test insight with Gemini...")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"Generate a brief response confirming you're working. Current time: {timestamp}"
        
        response = insight_generator.gemini_client.models.generate_content(
            model=insight_generator.gemini_model_name,
            contents=prompt
        )
        
        # Save the response to output directory
        output_path = f"{output_dir}/gemini_test_result.txt"
        with open(output_path, 'w') as f:
            f.write(f"Test performed at: {timestamp}\n\n")
            f.write("Response from Gemini:\n")
            f.write(response.text)
        
        print(f"Test successful! Response received and saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error testing Gemini API: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test connection to Google Gemini API")
    parser.add_argument("--output-dir", default="dist/reports",
                      help="Directory to save output files")
    
    args = parser.parse_args()
    test_gemini_connection(args.output_dir)
