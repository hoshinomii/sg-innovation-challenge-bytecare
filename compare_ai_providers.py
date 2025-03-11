import os
import pandas as pd
from ai_insights import AIInsightGenerator
from azure_helpers import get_azure_openai_client
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

def compare_providers(force_provider=None):
    """Compare insights from different AI providers"""
    # Create a sample dataframe for demonstration
    restock_df = pd.DataFrame({
        'Description': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
        'recommended_restock': [100, 50, 75, 25, 60],
        'safety_stock': [20, 10, 15, 5, 12],
        'predicted_demand': [80, 40, 60, 20, 48]
    })
    
    # Set index to match expected format
    restock_df.index = ['A001', 'B002', 'C003', 'D004', 'E005']
    
    # Get Azure client for configuration
    azure_client, deployment = get_azure_openai_client()
    
    # Print configuration info
    print("=== AI Provider Configuration ===")
    print(f"AZURE_OPENAI_API_KEY exists: {'AZURE_OPENAI_API_KEY' in os.environ}")
    print(f"AZURE_OPENAI_ENDPOINT exists: {'AZURE_OPENAI_ENDPOINT' in os.environ}")
    print(f"GEMINI_API_KEY exists: {'GEMINI_API_KEY' in os.environ}")
    print(f"DEFAULT_AI_PROVIDER: {os.environ.get('DEFAULT_AI_PROVIDER', 'not set')}")
    print()
    
    # Get the current default provider
    default_provider = os.environ.get('DEFAULT_AI_PROVIDER', 'not set').lower()
    
    if force_provider:
        print(f"Forcing provider: {force_provider}")
        providers_to_test = [force_provider]
    else:
        # Test both providers
        providers_to_test = ["openai", "gemini"]
    
    results = {}
    
    for provider in providers_to_test:
        print(f"\n=== Testing {provider.upper()} Provider ===")
        
        # Initialize the provider-specific insight generator
        insight_generator = AIInsightGenerator(
            provider=provider,
            azure_deployment=deployment
        )
        
        if insight_generator.provider == "template":
            print(f"Failed to initialize {provider} provider, configuration may be missing")
            results[provider] = f"Failed to initialize {provider} provider"
            continue
            
        print(f"Successfully initialized {insight_generator.provider} provider")
        
        # Generate insights
        try:
            start_time = pd.Timestamp.now()
            insights = insight_generator.generate_inventory_insights(restock_df)
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\nGenerated insights with {provider} in {duration:.2f} seconds:")
            print("---")
            print(insights)
            print("---\n")
            
            results[provider] = {
                "insights": insights,
                "duration": duration,
                "status": "success"
            }
        except Exception as e:
            print(f"Error generating insights with {provider}: {e}")
            results[provider] = {
                "insights": f"Error: {str(e)}",
                "duration": 0,
                "status": "error"
            }
    
    # Print comparison
    if len(results) > 1:
        print("\n=== Provider Comparison ===")
        for provider, result in results.items():
            status = result['status'] if isinstance(result, dict) else 'error'
            duration = result.get('duration', 0) if isinstance(result, dict) else 0
            print(f"{provider.upper()}: Status={status}, Time={duration:.2f}s")
        
        # Provide a recommendation
        if all(isinstance(result, dict) and result['status'] == 'success' for result in results.values()):
            gemini_time = results.get('gemini', {}).get('duration', float('inf'))
            openai_time = results.get('openai', {}).get('duration', float('inf'))
            
            if gemini_time < openai_time:
                recommendation = "gemini (faster response)"
            else:
                recommendation = "openai (faster response)"
                
            print(f"\nRecommended provider: {recommendation}")
            print(f"Current default provider: {default_provider}")
            
            if recommendation.split()[0] != default_provider:
                print(f"\nConsider changing DEFAULT_AI_PROVIDER in .env to {recommendation.split()[0]} for better performance")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different AI providers for inventory insights")
    parser.add_argument("--provider", choices=["gemini", "openai"], 
                      help="Force a specific provider instead of comparing both")
    
    args = parser.parse_args()
    compare_providers(args.provider)
