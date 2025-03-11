import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import importlib.util
import sys
import io
import base64

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

# Check Azure OpenAI availability
print("Checking Azure OpenAI availability...")
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
    print("Azure OpenAI package successfully imported")
except ImportError as e:
    AZURE_OPENAI_AVAILABLE = False
    print(f"Failed to import Azure OpenAI: {e}")
    print("Please run: pip install --upgrade openai>=1.0.0")

# Import the report generator if the module exists
try:
    from report_generator import save_reports, generate_markdown_summary, generate_html_dashboard
    REPORT_GENERATOR_AVAILABLE = True
    print("Report generator module successfully imported")
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False
    print("Report generator module not found - some reporting features will be limited")

class AIInsightGenerator:
    """Class to generate AI-powered insights for inventory management using Google's Gemini or Azure OpenAI"""
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None, 
                 gemini_model: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 azure_deployment: Optional[str] = None,
                 azure_api_version: Optional[str] = "2023-07-01-preview",
                 provider: str = "auto"):
        """Initialize the AI insight generator with API keys and model selection"""
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.gemini_model_name = gemini_model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        self.gemini_client = None
        
        self.azure_api_key = azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")
        self.azure_api_version = azure_api_version
        self.azure_client = None
        
        self.provider = "template"  # Default to template, will be changed if setup succeeds
        
        # Check for DEFAULT_AI_PROVIDER environment variable
        env_provider = os.environ.get("DEFAULT_AI_PROVIDER", "").lower()
        always_use_ai = os.environ.get("ALWAYS_USE_AI", "").lower() in ["true", "1", "yes"]
        
        print(f"Environment variables: DEFAULT_AI_PROVIDER={env_provider}, ALWAYS_USE_AI={always_use_ai}")
        
        # Override provider based on environment variable if specified
        if provider == "auto" and env_provider in ["gemini", "openai"]:
            print(f"Using provider from environment variable: {env_provider}")
            provider = env_provider
        
        # Set up the specified provider
        if provider == "gemini":
            print("Setting up Gemini as the provider")
            self._setup_gemini()
        elif provider == "azure" or provider == "openai":
            print("Setting up Azure OpenAI as the provider")
            self._setup_azure_openai()
        elif provider == "auto":
            # Try both providers in sequence, starting with the default from env if available
            print("Auto-selecting provider")
            
            if env_provider == "gemini":
                # Try Gemini first, then fall back to Azure
                self._setup_gemini()
                if self.provider == "template":  # If Gemini setup failed
                    print("Gemini setup failed, trying Azure OpenAI")
                    self._setup_azure_openai()
            else:
                # Try Azure first, then fall back to Gemini
                self._setup_azure_openai()
                if self.provider == "template":  # If Azure setup failed
                    print("Azure OpenAI setup failed, trying Gemini")
                    self._setup_gemini()
                    
        print(f"Final selected provider: {self.provider}")
    
    def _setup_gemini(self):
        """Setup Google Gemini client"""
        try:
            if GEMINI_AVAILABLE and self.gemini_api_key:
                print(f"Configuring Gemini client with model {self.gemini_model_name}...")
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_client = genai
                
                # Test if the model works
                try:
                    test_response = self.gemini_client.models.generate_content(
                        model=self.gemini_model_name,
                        contents="Hello"
                    )
                    print(f"Gemini test successful")
                    self.provider = "gemini"  # Set to gemini only after successful test
                except Exception as e:
                    print(f"Gemini test failed: {e}")
            else:
                if not GEMINI_AVAILABLE:
                    print("Gemini API not available - package not installed")
                if not self.gemini_api_key:
                    print("Gemini API key not provided")
        except Exception as e:
            print(f"Error initializing Gemini client: {str(e)}")
    
    def _setup_azure_openai(self):
        """Setup Azure OpenAI client"""
        try:
            print(f"Configuring Azure OpenAI client...")
            # Use the newer AzureOpenAI client approach from script.py
            if AZURE_OPENAI_AVAILABLE and self.azure_api_key and self.azure_endpoint:
                self.azure_client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                )
                
                # Test if the model works
                try:
                    print(f"Testing Azure OpenAI model: {self.azure_deployment}")
                    test_response = self.azure_client.chat.completions.create(
                        model=self.azure_deployment,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10
                    )
                    print(f"Azure OpenAI test successful")
                    self.provider = "azure"  # Set to azure only after successful test
                except Exception as e:
                    print(f"Azure OpenAI test failed: {e}")
            else:
                if not AZURE_OPENAI_AVAILABLE:
                    print("Azure OpenAI API not available - package not installed")
                if not self.azure_api_key:
                    print("Azure OpenAI API key not provided")
                if not self.azure_endpoint:
                    print("Azure OpenAI endpoint not provided")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {str(e)}")
    
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
            if self.provider == "gemini":
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt
                )
                return response.text
            elif self.provider == "azure":
                # Updated to use the new Azure OpenAI client pattern
                response = self.azure_client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                return response.choices[0].message.content
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
            if self.provider == "gemini":
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt
                )
                return response.text
            elif self.provider == "azure":
                # Updated to use the new Azure OpenAI client pattern
                response = self.azure_client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                return response.choices[0].message.content
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
            if self.provider == "gemini":
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt
                )
                return response.text
            elif self.provider == "azure":
                # Updated to use the new Azure OpenAI client pattern
                response = self.azure_client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150
                )
                return response.choices[0].message.content
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


def create_product_trend_graph(product_data: pd.DataFrame, product_name: str, stockcode: str) -> str:
    """Create a graph showing the sales trend for a specific product and return as base64 image"""
    plt.figure(figsize=(10, 6))
    
    # Create a time series plot of quantities
    if 'InvoiceDate' in product_data.columns and len(product_data) > 0:
        # Ensure data is sorted by date
        product_data = product_data.sort_values('InvoiceDate')
        # Create the time series plot
        plt.plot(product_data['InvoiceDate'], product_data['Quantity'], marker='o', linestyle='-')
        plt.title(f'Sales Trend for {product_name} (Code: {stockcode})')
        plt.xlabel('Date')
        plt.ylabel('Quantity Sold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        # If no time data available, create a bar chart of quantities
        plt.bar(range(len(product_data)), product_data['Quantity'])
        plt.title(f'Sales Distribution for {product_name} (Code: {stockcode})')
        plt.xlabel('Sales Event')
        plt.ylabel('Quantity Sold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    # Convert plot to base64 string for embedding in markdown
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"![{product_name} Sales Trend](data:image/png;base64,{image_base64})"

def create_top_items_graph(restock_df: pd.DataFrame) -> str:
    """Create a graph showing top items to restock and return as base64 image"""
    # Take top 10 items for the graph
    top_items = restock_df.head(10).copy()
    
    plt.figure(figsize=(12, 8))
    
    # Create a horizontal bar chart with product descriptions and recommended restock quantities
    descriptions = [str(desc)[:30] + '...' if len(str(desc)) > 30 else str(desc) for desc in top_items['Description']]
    y_pos = np.arange(len(descriptions))
    
    plt.barh(y_pos, top_items['recommended_restock'], align='center')
    plt.yticks(y_pos, descriptions)
    plt.xlabel('Recommended Restock Quantity')
    plt.title('Top 10 Items to Restock')
    plt.tight_layout()
    
    # Convert plot to base64 string for embedding in markdown
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"![Top 10 Items to Restock](data:image/png;base64,{image_base64})"

def save_report_as_html(report: str, output_path: str = "inventory_report.html"):
    """Save the markdown report as an HTML file for easy viewing of embedded images
    
    Args:
        report: The markdown report with embedded base64 images
        output_path: Path to save the HTML file
    """
    try:
        # Try to import markdown with extension support
        import markdown
        from markdown.extensions import tables
        
        # Convert markdown to HTML with table extension
        html = markdown.markdown(report, extensions=['tables', 'markdown.extensions.tables'])
        
        # Create a complete HTML document
        html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Management Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; margin-top: 30px; }}
        h3 {{ color: #2980b9; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    {html}
</body>
</html>"""
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_doc)
        
        print(f"Report saved as HTML: {output_path}")
        return output_path
        
    except ImportError:
        print("Markdown module not available. Using manual HTML conversion.")
        try:
            # Manual markdown-to-html conversion for tables
            html_content = report
            
            # Find and convert markdown tables
            import re
            table_pattern = r'\|(.+)\|\n\|[-|]+\|\n((?:\|.+\|\n)+)'
            
            def convert_table(match):
                headers = match.group(1).strip().split('|')
                headers = [h.strip() for h in headers if h.strip()]
                
                rows_text = match.group(2)
                rows = rows_text.strip().split('\n')
                
                html_table = '<table>\n  <thead>\n    <tr>\n'
                for header in headers:
                    html_table += f'      <th>{header.strip()}</th>\n'
                html_table += '    </tr>\n  </thead>\n  <tbody>\n'
                
                for row in rows:
                    if '|' not in row:
                        continue
                    cells = row.strip().split('|')
                    cells = [c.strip() for c in cells if c.strip() != '']
                    
                    html_table += '    <tr>\n'
                    for cell in cells:
                        html_table += f'      <td>{cell.strip()}</td>\n'
                    html_table += '    </tr>\n'
                
                html_table += '  </tbody>\n</table>'
                return html_table
            
            html_content = re.sub(table_pattern, convert_table, html_content)
            
            # Convert headings
            html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
            
            # Convert paragraphs
            html_content = re.sub(r'^\s*$', '<br>', html_content, flags=re.MULTILINE)
            
            # Create complete HTML document
            html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Management Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; margin-top: 30px; }}
        h3 {{ color: #2980b9; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_doc)
            
            print(f"Report saved as HTML: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating manual HTML: {e}")
            
            # Save as plain markdown instead
            with open(output_path.replace('.html', '.md'), 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"Report saved as Markdown: {output_path.replace('.html', '.md')}")
            return output_path.replace('.html', '.md')
    except Exception as e:
        print(f"Error saving report: {e}")
        
        # Save as plain markdown as a last resort
        try:
            markdown_path = output_path.replace('.html', '.md')
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved as Markdown: {markdown_path}")
            return markdown_path
        except:
            print("Failed to save report in any format")
            return None


# Modify the generate_report_with_insights function to also save the report
def generate_report_with_insights(restock_df: pd.DataFrame, 
                                  feature_data: pd.DataFrame, 
                                  gemini_api_key: Optional[str] = None,
                                  gemini_model: Optional[str] = None,
                                  azure_api_key: Optional[str] = None,
                                  azure_endpoint: Optional[str] = None,
                                  azure_deployment: Optional[str] = None,
                                  save_html: bool = True,
                                  output_path: str = "inventory_report.html") -> str:
    """Generate a comprehensive report with AI-enhanced insights and visualizations"""
    
    insight_generator = AIInsightGenerator(
        gemini_api_key=gemini_api_key, 
        gemini_model=gemini_model or "gemini-2.0-flash",
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment
    )
    
    # Generate overall inventory insights
    inventory_summary = insight_generator.generate_inventory_insights(restock_df)
    
    # Top 10 items section with demand and restock amounts
    top_10_items = restock_df.head(10).copy()
    
    # Create visualization for top 10 items
    top_items_graph = create_top_items_graph(restock_df)
    
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
            
            # Create graph for this product
            product_graph = create_product_trend_graph(product_data, product_name, idx)
            
            product_insights.append({
                'product_code': idx,
                'product_name': product_name,
                'recommended_restock': row['recommended_restock'],
                'trend_explanation': trend_explanation,
                'restock_reasoning': restock_reasoning,
                'graph': product_graph
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
The following chart shows the top 10 items that need restocking:

{top_items_graph}

The following table provides detailed information on these items:

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

{p['graph']}

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
    
    # Save report if requested
    if save_html:
        saved_path = save_report_as_html(report, output_path)
        if saved_path:
            report += f"\n\n---\nThis report has been saved to: {saved_path}"
    
    return report

# Add this function at the end of the file
def generate_inventory_reports(restock_df: pd.DataFrame, 
                              feature_data: pd.DataFrame,
                              md_output_path: str = "inventory_summary.md",
                              html_output_path: str = "inventory_dashboard.html",
                              pdf_output_path: str = "inventory_report.pdf") -> Tuple[str, str, Optional[str]]:
    """
    Generate brief markdown summary, HTML dashboard, and PDF report for inventory management
    
    Args:
        restock_df: DataFrame containing restock recommendations
        feature_data: DataFrame with historical sales data
        md_output_path: Path to save the markdown report
        html_output_path: Path to save the HTML dashboard
        pdf_output_path: Path to save the PDF report
        
    Returns:
        Tuple of (markdown_path, html_path, pdf_path) with the saved file paths
    """
    if REPORT_GENERATOR_AVAILABLE:
        # Use the dedicated report generator module
        return save_reports(restock_df, feature_data, md_output_path, html_output_path, pdf_output_path)
    else:
        # Fall back to simplified report generation
        print("Using simplified report generation (report_generator module not available)")
        
        # Generate and save markdown report
        markdown_report = f"""# Inventory Restock Summary
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overview
- Total products to restock: {len(restock_df)}
- Urgent items: {len(restock_df[restock_df['recommended_restock'] > restock_df['recommended_restock'].mean()])}
- Total units to restock: {int(restock_df['recommended_restock'].sum())}

## Top 5 Items to Restock
| Product | Recommended Restock |
|---------|---------------------|"""

        # Add top 5 items
        top_items = restock_df.head(5)
        for idx, row in top_items.iterrows():
            product_name = row['Description'] if not pd.isna(row['Description']) else f"Product {idx}"
            markdown_report += f"\n| {product_name} | {row['recommended_restock']} |"
        
        # Save markdown report
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
            
        # Generate a simplified HTML report
        html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Inventory Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Inventory Dashboard</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <h2>Summary</h2>
    <ul>
        <li>Total products to restock: {len(restock_df)}</li>
        <li>Urgent items: {len(restock_df[restock_df['recommended_restock'] > restock_df['recommended_restock'].mean()])}</li>
        <li>Total units to restock: {int(restock_df['recommended_restock'].sum())}</li>
    </ul>
    
    <h2>Inventory Restock Table</h2>
    <table>
        <tr>
            <th>Product</th>
            <th>Recommended Restock</th>
        </tr>"""
        
        # Add all items to the HTML table
        for idx, row in restock_df.iterrows():
            product_name = row['Description'] if not pd.isna(row['Description']) else f"Product {idx}"
            html_report += f"""
        <tr>
            <td>{product_name}</td>
            <td>{row['recommended_restock']}</td>
        </tr>"""
            
        # Close HTML
        html_report += """
    </table>
</body>
</html>"""
        
        # Save HTML report
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Always try to generate PDF
        pdf_path = None
        try:
            # Check if we can use weasyprint for PDF generation
            print("Attempting to generate PDF report...")
            try:
                import weasyprint
                # Generate PDF from HTML
                pdf = weasyprint.HTML(string=html_report).write_pdf()
                with open(pdf_output_path, 'wb') as f:
                    f.write(pdf)
                pdf_path = pdf_output_path
                print(f"PDF report successfully generated at: {pdf_output_path}")
            except ImportError:
                # Try using pdfkit if weasyprint is not available
                try:
                    import pdfkit
                    pdfkit.from_string(html_report, pdf_output_path)
                    pdf_path = pdf_output_path
                    print(f"PDF report successfully generated at: {pdf_output_path}")
                except ImportError:
                    # Try using a markdown to pdf converter if available
                    if 'markdown_to_pdf' in globals():
                        pdf_path = markdown_to_pdf(markdown_report, pdf_output_path)
                        print(f"PDF report successfully generated at: {pdf_output_path}")
                    else:
                        print("PDF generation libraries not available. Please install weasyprint or pdfkit:")
                        print("  pip install weasyprint")
                        print("  pip install pdfkit")
                except Exception as e:
                    print(f"Error generating PDF with pdfkit: {e}")
        except Exception as e:
            print(f"Error generating PDF: {e}")
            print("PDF generation failed, but HTML and Markdown reports are still available.")
        
        return md_output_path, html_output_path, pdf_path