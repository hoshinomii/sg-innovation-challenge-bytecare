import os
import pandas as pd
from ai_insights import AIInsightGenerator
from azure_helpers import get_azure_openai_client, load_config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    # Load configuration and get Azure OpenAI client
    azure_client, deployment = get_azure_openai_client()
    
    if not azure_client:
        print("Failed to initialize Azure OpenAI client. Check your configuration.")
        return
    
    print(f"Successfully initialized Azure OpenAI client with deployment: {deployment}")
    
    # Create a sample dataframe for demonstration
    restock_df = pd.DataFrame({
        'Description': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
        'recommended_restock': [100, 50, 75, 25, 60],
        'safety_stock': [20, 10, 15, 5, 12],
        'predicted_demand': [80, 40, 60, 20, 48]
    })
    
    # Set index to match expected format
    restock_df.index = ['A001', 'B002', 'C003', 'D004', 'E005']
    
    # Debug: Print environment variables to verify they are loaded
    print(f"AZURE_OPENAI_API_KEY exists: {'AZURE_OPENAI_API_KEY' in os.environ}")
    print(f"AZURE_OPENAI_ENDPOINT exists: {'AZURE_OPENAI_ENDPOINT' in os.environ}")
    
    # Initialize the AI Insight Generator specifically with Azure OpenAI
    insight_generator = AIInsightGenerator(
        azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),  # Changed from ENDPOINT_URL to match .env
        azure_deployment=deployment,
        provider="azure"  # Force using Azure
    )
    
    if insight_generator.provider == "azure":
        print("Successfully configured AIInsightGenerator with Azure OpenAI")
        
        # Create output directory if it doesn't exist
        output_dir = "dist/reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate inventory insights
        insights = insight_generator.generate_inventory_insights(restock_df)
        
        # Save insights to a file in the output directory
        output_path = f"{output_dir}/azure_openai_insights.txt"
        with open(output_path, 'w') as f:
            f.write(insights)
        
        print("\n--- Generated Inventory Insights ---\n")
        print(insights)
        print(f"\nInsights saved to {output_path}")
    else:
        print(f"Failed to configure Azure. Current provider: {insight_generator.provider}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("dist/reports", exist_ok=True)
    main()
