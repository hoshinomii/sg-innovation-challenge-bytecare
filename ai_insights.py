import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import importlib.util
import sys

# Check Python version
print(f"Python version: {sys.version}")

# Check if google package is properly installed
print("Checking Google Gemini API availability...")
try:
    from google import genai
    GEMINI_AVAILABLE = True
    print("Google Generative AI package successfully imported")
except ImportError as e:
    GEMINI_AVAILABLE = False
    print(f"Failed to import Google Generative AI: {e}")
    print("Please run: pip install --upgrade google-generativeai>=0.7.0")

class AIInsightGenerator:
    """Class to generate AI-powered insights for inventory management using Google's Gemini"""
    
    def __init__(self, gemini_api_key: Optional[str] = None, gemini_model: Optional[str] = None):
        """Initialize the AI insight generator with Gemini API key and model selection"""
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.gemini_model_name = gemini_model or "gemini-2.0-flash"
        self.gemini_client = None
        self.provider = "template"  # Default to template, will be changed to "gemini" if setup succeeds
        
        # Debug API key (showing only first few characters)
        if self.gemini_api_key:
            print(f"Gemini API Key is set (prefix: {self.gemini_api_key[:8]}...)")
        else:
            print("WARNING: Gemini API key is not set")
            return
        
        # Initialize Gemini client
        if GEMINI_AVAILABLE:
            try:
                print(f"Configuring Google Gemini with API key...")
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                
                # Test if the model works with a simple prompt
                try:
                    print(f"Testing Gemini model: {self.gemini_model_name}")
                    test_response = self.gemini_client.models.generate_content(
                        model=self.gemini_model_name,
                        contents="Hello"
                    )
                    print(f"Gemini test successful. Response received: {test_response.text[:20]}...")
                    self.provider = "gemini"  # Set to gemini only after successful test
                except Exception as e:
                    print(f"Gemini test failed with the client: {e}")
                    print("Falling back to template-based insights")
            except Exception as e:
                print(f"Error initializing Gemini client: {str(e)}")
                print("Falling back to template-based insights")
        else:
            print("WARNING: Google Generative AI package is not available")
            
        print(f"Final provider selection: {self.provider}")
    
    def generate_inventory_insights(self, restock_df: pd.DataFrame) -> str:
        """Generate natural language insights about inventory restocking needs"""
        if self.provider == "template":
            return self._generate_template_insights(restock_df)
            
        # Prepare data summary for the AI
        total_items = len(restock_df)
        urgent_items = len(restock_df[restock_df['recommended_restock'] > restock_df['recommended_restock'].mean()])
        top_items = restock_df.head(5)[['Description', 'recommended_restock']].to_dict('records')
        
        prompt = f"""
        As an inventory management AI assistant, analyze the following restocking data:
        
        Total products requiring restock: {total_items}
        Urgent restock needed (above average): {urgent_items}
        
        Top 5 products to restock:
        {json.dumps(top_items, indent=2)}
        
        Generate a concise executive summary of the inventory situation, including:
        1. Key trends and patterns in the restocking needs
        2. Specific recommendations for inventory managers
        3. Potential cost implications
        4. Any anomalies that require attention
        
        Format the response in clear paragraphs with appropriate headings.
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            return self._generate_template_insights(restock_df)
    
    def explain_product_trends(self, product_data: pd.DataFrame, product_name: str) -> str:
        """Generate explanations for trends in a specific product"""
        if self.provider == "template":
            return f"Analysis for {product_name}: Sales show typical seasonal patterns."
        
        # Calculate key metrics
        avg_qty = product_data['Quantity'].mean()
        max_qty = product_data['Quantity'].max()
        trend = "increasing" if product_data['Quantity'].iloc[-5:].mean() > avg_qty else "decreasing"
        
        prompt = f"""
        As an inventory analyst, explain the sales trends for product "{product_name}" with these metrics:
        - Average daily quantity sold: {avg_qty:.2f}
        - Maximum daily quantity sold: {max_qty}
        - Recent trend: {trend}
        
        Provide a short paragraph explaining what these numbers mean for inventory management.
        Consider seasonality, potential causes, and recommendations.
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error explaining product trends: {e}")
            return f"Analysis for {product_name}: Sales averaging {avg_qty:.2f} units daily, with a {trend} trend."
    
    def generate_restock_reasoning(self, product_info: Dict[str, Any]) -> str:
        """Generate reasoning for the restock recommendation"""
        if self.provider == "template":
            return f"Recommendation based on historical demand patterns and safety stock calculations."
        
        # Handle both key naming conventions (from training.py and predict.py)
        weekly_demand = product_info.get('predicted_weekly_demand', 
                                         product_info.get('predicted_demand', 'unknown'))
        
        prompt = f"""
        Explain the reasoning behind this restock recommendation in 2-3 sentences:
        Product: {product_info['Description']}
        Predicted weekly demand: {weekly_demand}
        Safety stock: {product_info['safety_stock']}
        Recommended restock quantity: {product_info['recommended_restock']}
        """
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error generating restock reasoning: {e}")
            return "Recommended based on predicted demand plus safety stock to prevent stockouts."
    
    def _generate_template_insights(self, restock_df: pd.DataFrame) -> str:
        """Generate template-based insights when AI is not available"""
        total_items = len(restock_df)
        urgent_items = len(restock_df[restock_df['recommended_restock'] > restock_df['recommended_restock'].mean()])
        top_items = restock_df.head(3)['Description'].tolist()
        
        return f"""# Inventory Restock Analysis

## Summary
The system has identified {total_items} products that need restocking, with {urgent_items} items requiring urgent attention (above average restock amounts).

## Key Recommendations
1. Prioritize restocking of: {', '.join(top_items)}
2. Consider bulk ordering for products with similar suppliers
3. Review items with unusually high safety stock requirements for potential demand volatility

## Notes
Restock recommendations are based on historical sales patterns, recent trends, and calculated safety stock levels to minimize stock-outs while optimizing inventory costs.
"""


def generate_report_with_insights(restock_df: pd.DataFrame, 
                                  feature_data: pd.DataFrame, 
                                  gemini_api_key: Optional[str] = None,
                                  gemini_model: Optional[str] = None) -> str:
    """Generate a comprehensive report with AI-enhanced insights"""
    
    insight_generator = AIInsightGenerator(
        gemini_api_key=gemini_api_key, 
        gemini_model=gemini_model or "gemini-2.0-flash"
    )
    
    # Generate overall inventory insights
    inventory_summary = insight_generator.generate_inventory_insights(restock_df)
    
    # Top 10 items section with demand and restock amounts
    top_10_items = restock_df.head(10).copy()
    
    # Generate specific insights for top products
    product_insights = []
    for idx, row in restock_df.head(5).iterrows():
        try:
            product_data = feature_data[feature_data['StockCode'] == idx]
            product_name = row['Description']
            if pd.isna(product_name) or product_name == "Unknown":
                product_name = f"Product {idx}"
                
            trend_explanation = insight_generator.explain_product_trends(product_data, product_name)
            restock_reasoning = insight_generator.generate_restock_reasoning(row)
            
            product_insights.append({
                'product_code': idx,
                'product_name': product_name,
                'recommended_restock': row['recommended_restock'],
                'trend_explanation': trend_explanation,
                'restock_reasoning': restock_reasoning
            })
        except Exception as e:
            print(f"Error generating insights for product {idx}: {e}")
            continue
    
    # Compile the full report with proper markdown formatting - removed indentation
    report = f"""# Inventory Management Report
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary
{inventory_summary.strip()}

## Top 10 Items for Restocking
The following table shows the top 10 items that need attention, with their predicted weekly demand and recommended restock quantities:

"""
    
    # Add top 10 items table
    report += "| Product Code | Description | Predicted Weekly Demand | Recommended Restock |\n"
    report += "|-------------|-------------|------------------------|---------------------|\n"
    
    for idx, row in top_10_items.iterrows():
        product_name = row['Description']
        if pd.isna(product_name) or product_name == "Unknown":
            product_name = f"Product {idx}"
            
        # Handle both key naming conventions
        weekly_demand = row.get('predicted_weekly_demand', row.get('predicted_demand', 'N/A'))
        
        report += f"| {idx} | {product_name} | {weekly_demand} | {row['recommended_restock']} |\n"
    
    report += "\n## Top Product Details\n"
    
    for p in product_insights:
        report += f"""
### {p['product_name']} (Code: {p['product_code']})
**Recommended Restock Quantity:** {p['recommended_restock']}

**Trend Analysis:**
{p['trend_explanation'].strip()}

**Restock Reasoning:**
{p['restock_reasoning'].strip()}
"""
    
    report += """
## Conclusion
The AI-enhanced analysis provides both data-driven recommendations and contextual understanding
of inventory needs. These insights should be reviewed alongside business knowledge and market 
conditions to finalize inventory decisions.
"""
    
    return report
