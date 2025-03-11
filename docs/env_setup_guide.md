# Environment Configuration Guide

This guide explains how to set up your environment variables for using AI providers (Azure OpenAI and Google Gemini) in this project.

## Setting Up Your `.env` File

The project uses a `.env` file to store sensitive configuration like API keys. Create or modify this file in the root directory of the project.

### Basic Structure

```properties
# API Keys
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Other Configuration
DEFAULT_AI_PROVIDER=openai     # Options: 'openai' or 'gemini'
ALWAYS_USE_AI=true             # Options: true or false

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_url_here
AZURE_OPENAI_DEPLOYMENT=your_deployment_name_here
API_VERSION=2024-05-01-preview

# Gemini Configuration
GEMINI_MODEL=gemini-2.0-flash
```

## Azure OpenAI Setup Instructions

1. **Create an Azure OpenAI resource**:
   - Go to the [Azure Portal](https://portal.azure.com)
   - Create a new Azure OpenAI service
   - Note down your resource details

2. **Get your API Key**:
   - In the Azure Portal, navigate to your OpenAI resource
   - Go to "Keys and Endpoint" section
   - Copy one of the keys to the `AZURE_OPENAI_API_KEY` in your `.env` file

3. **Configure the endpoint**:
   - Copy the endpoint URL from the "Keys and Endpoint" section
   - Set this as `AZURE_OPENAI_ENDPOINT` in your `.env` file
   - Example: `https://your-resource-name.openai.azure.com/`

4. **Set up deployment**:
   - In your Azure OpenAI resource, go to "Model Deployments"
   - Create a new deployment or note the name of an existing one
   - Set this as `AZURE_OPENAI_DEPLOYMENT` in your `.env` file
   - Example: `gpt-4` or `gpt-35-turbo`

## Google Gemini Setup Instructions

1. **Create a Google AI Studio account**:
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Sign in or create a new account

2. **Generate an API Key**:
   - In Google AI Studio, navigate to the API section
   - Create a new API key
   - Copy the API key to `GEMINI_API_KEY` in your `.env` file

3. **Choose your Gemini model**:
   - Set `GEMINI_MODEL` to the model you want to use
   - Common options include:
     - `gemini-2.0-flash` (faster responses, suitable for most use cases)
     - `gemini-2.0-pro` (more capable, better for complex tasks)

## Configuration Options

- **DEFAULT_AI_PROVIDER**: Choose which AI provider to use by default
  - Set to `openai` to use Azure OpenAI
  - Set to `gemini` to use Google Gemini

- **ALWAYS_USE_AI**: Determines if AI should be used for all operations
  - Set to `true` to always use AI
  - Set to `false` to use fallback methods when appropriate

## Testing Your Configuration

You can test if your environment variables are properly loaded with this script:

```python
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check if variables are loaded
print("Azure OpenAI API Key:", os.environ.get("AZURE_OPENAI_API_KEY", "Not found")[:5] + "..." if os.environ.get("AZURE_OPENAI_API_KEY") else "Not found")
print("Azure OpenAI Endpoint:", os.environ.get("AZURE_OPENAI_ENDPOINT", "Not found"))
print("Gemini API Key:", os.environ.get("GEMINI_API_KEY", "Not found")[:5] + "..." if os.environ.get("GEMINI_API_KEY") else "Not found")
print("Default AI Provider:", os.environ.get("DEFAULT_AI_PROVIDER", "Not found"))
```

## Troubleshooting

- **Variables not loading**: Make sure your `.env` file is in the root directory of your project
- **API errors**: Verify your API keys are correct and have the necessary permissions
- **Module errors**: Ensure you've installed the required packages with `pip install python-dotenv openai google-generativeai`

Remember to never commit your `.env` file to version control. Make sure it's included in your `.gitignore`.
