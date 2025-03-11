import os
import sys
from dotenv import load_dotenv
import configparser
import json

# Load environment variables at module level
load_dotenv()

def load_config(config_file="_config", key_file="_key"):
    """Load configuration from files or environment variables"""
    config = {}
    
    # Try to load from config file
    try:
        parser = configparser.ConfigParser()
        parser.read(config_file)
        if "Azure" in parser:
            config["endpoint"] = parser["Azure"].get("endpoint", "")
            config["deployment"] = parser["Azure"].get("deployment", "")
            config["api_version"] = parser["Azure"].get("api_version", "")
    except Exception as e:
        print(f"Warning: Configuration file '{config_file}' not found")
    
    # Try to load API key from key file
    try:
        with open(key_file, 'r') as f:
            key_data = json.load(f)
            config["api_key"] = key_data.get("api_key", "")
    except Exception as e:
        print(f"Warning: Configuration file '{key_file}' not found")
    
    # Override with environment variables if available
    if "AZURE_OPENAI_ENDPOINT" in os.environ:
        config["endpoint"] = os.environ["AZURE_OPENAI_ENDPOINT"]
    if "AZURE_OPENAI_DEPLOYMENT" in os.environ:
        config["deployment"] = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    if "API_VERSION" in os.environ:
        config["api_version"] = os.environ["API_VERSION"]
    if "AZURE_OPENAI_API_KEY" in os.environ:
        config["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
    
    return config

def get_azure_openai_client():
    """Initialize and return Azure OpenAI client with deployment name"""
    # Debug info
    print("Environment variables:")
    print(f"AZURE_OPENAI_ENDPOINT: {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
    print(f"AZURE_OPENAI_API_KEY exists: {'AZURE_OPENAI_API_KEY' in os.environ}")
    print(f"AZURE_OPENAI_DEPLOYMENT: {os.environ.get('AZURE_OPENAI_DEPLOYMENT')}")
    print(f"API_VERSION: {os.environ.get('API_VERSION')}")
    
    config = load_config()
    
    # Debug the config
    print("Config loaded:")
    for key, value in config.items():
        if key == "api_key" and value:
            print(f"{key}: [REDACTED]")
        else:
            print(f"{key}: {value}")
    
    # Check if required configuration exists
    if not config.get("endpoint") or not config.get("api_key"):
        print("Error: Azure OpenAI endpoint URL or API key not found in environment variables or config files")
        return None
    
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            azure_endpoint=config["endpoint"],
            api_key=config["api_key"],
            api_version=config.get("api_version", "2024-05-01-preview")
        )
        
        # Return both the client and the deployment name
        return client, config.get("deployment", "gpt-4")
        
    except ImportError:
        print("Error: Failed to import Azure OpenAI SDK. Please install it with 'pip install openai'")
        return None
    except Exception as e:
        print(f"Error initializing Azure OpenAI client: {str(e)}")
        return None
