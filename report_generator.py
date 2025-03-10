import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Optional, Tuple
import os

def generate_markdown_summary(restock_df: pd.DataFrame) -> str:
    """
    Generate a simple markdown report with a brief summary of inventory needs
    
    Args:
        restock_df: DataFrame containing restock recommendations
        
    Returns:
        Markdown formatted string with the report
    """
    # Basic statistics
    total_items = len(restock_df)
    urgent_items = len(restock_df[restock_df['recommended_restock'] > restock_df['recommended_restock'].mean()])
    total_restock = int(restock_df['recommended_restock'].sum())
    top_items = restock_df.head(5)[['Description', 'recommended_restock']].to_dict('records')
    
    # Calculate distribution by restock amount
    restock_ranges = [
        (0, 10, "Low (0-10)"),
        (11, 50, "Medium (11-50)"),
        (51, 200, "High (51-200)"),
        (201, float('inf'), "Very High (>200)")
    ]
    
    distribution = {}
    for low, high, label in restock_ranges:
        count = len(restock_df[(restock_df['recommended_restock'] >= low) & 
                              (restock_df['recommended_restock'] <= high)])
        distribution[label] = count
    
    # Create the markdown report
    report = f"""# Inventory Restock Summary Report
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Key Metrics
- **Total Products Requiring Restock:** {total_items}
- **Products Needing Urgent Attention:** {urgent_items}
- **Total Units to Restock:** {total_restock}

## Restock Distribution
"""

    # Add distribution information
    for label, count in distribution.items():
        percentage = (count / total_items) * 100 if total_items > 0 else 0
        report += f"- **{label}:** {count} products ({percentage:.1f}%)\n"
    
    report += "\n## Top 5 Products to Restock\n"
    
    # Add top items table
    report += "| Description | Recommended Restock |\n"
    report += "|-------------|---------------------|\n"
    
    for item in top_items:
        report += f"| {item['Description']} | {item['recommended_restock']} |\n"
    
    report += """
## Next Steps
1. Review the detailed dashboard for comprehensive analysis
2. Prioritize items marked as urgent
3. Consider bulk ordering for products from the same suppliers
4. Adjust safety stock levels for seasonal items

*For a detailed interactive view, please open the HTML dashboard.*
"""
    
    return report

def generate_html_dashboard(restock_df: pd.DataFrame, feature_data: pd.DataFrame = None) -> str:
    """
    Generate an HTML dashboard with interactive elements for inventory management
    
    Args:
        restock_df: DataFrame containing restock recommendations
        feature_data: Optional DataFrame containing historical sales data
        
    Returns:
        HTML string with the complete dashboard
    """
    # Prepare data for charts
    top_10_items = restock_df.head(10).copy()
    
    # Create base64 encoded images for embedding
    top_items_chart = _create_top_items_chart_base64(restock_df)
    restock_distribution_chart = _create_restock_distribution_chart_base64(restock_df)
    
    # Prepare data tables as JSON for JavaScript use
    restock_data_json = restock_df.head(50).to_json(orient='records')
    
    # Create HTML with Bootstrap and Chart.js for a responsive dashboard
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Management Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ background-color: #f8f9fa; }}
        .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,.1); }}
        .card-header {{ background-color: #343a40; color: white; }}
        .table-container {{ max-height: 400px; overflow-y: auto; }}
        .kpi-card {{ text-align: center; padding: 20px; }}
        .kpi-value {{ font-size: 2rem; font-weight: bold; }}
        .kpi-label {{ font-size: 0.9rem; color: #6c757d; }}
        .chart-container {{ height: 300px; }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="row mb-4">
            <div class="col">
                <h1 class="display-5 fw-bold">Inventory Management Dashboard</h1>
                <p class="text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
        </div>
        
        <!-- KPI Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card kpi-card">
                    <div class="kpi-value text-primary">{len(restock_df)}</div>
                    <div class="kpi-label">Total Products to Restock</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card kpi-card">
                    <div class="kpi-value text-danger">{len(restock_df[restock_df['recommended_restock'] > restock_df['recommended_restock'].mean()])}</div>
                    <div class="kpi-label">Urgent Items (Above Average)</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card kpi-card">
                    <div class="kpi-value text-success">{int(restock_df['recommended_restock'].sum())}</div>
                    <div class="kpi-label">Total Units to Restock</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card kpi-card">
                    <div class="kpi-value text-info">{int(restock_df['recommended_restock'].mean())}</div>
                    <div class="kpi-label">Average Restock Per Product</div>
                </div>
            </div>
        </div>
        
        <!-- Charts Row -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Top 10 Items to Restock</div>
                    <div class="card-body">
                        <img src="data:image/png;base64,{top_items_chart}" class="img-fluid" alt="Top 10 Items Chart">
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Restock Quantity Distribution</div>
                    <div class="card-body">
                        <img src="data:image/png;base64,{restock_distribution_chart}" class="img-fluid" alt="Restock Distribution Chart">
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Data Table -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>Inventory Restock Data</span>
                        <input type="text" id="tableSearch" class="form-control form-control-sm" style="width: 200px;" placeholder="Search items...">
                    </div>
                    <div class="card-body table-container">
                        <table class="table table-striped table-hover" id="restockTable">
                            <thead>
                                <tr>
                                    <th onclick="sortTable(0)">Product Code ↕</th>
                                    <th onclick="sortTable(1)">Description ↕</th>
                                    <th onclick="sortTable(2)">Predicted Demand ↕</th>
                                    <th onclick="sortTable(3)">Current Stock ↕</th>
                                    <th onclick="sortTable(4)">Safety Stock ↕</th>
                                    <th onclick="sortTable(5)">Recommended Restock ↕</th>
                                </tr>
                            </thead>
                            <tbody id="tableBody">
                                <!-- Table data will be populated by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const restockData = {restock_data_json};
        
        // Populate table
        function populateTable(data) {{
            const tableBody = document.getElementById('tableBody');
            tableBody.innerHTML = '';
            
            data.forEach(item => {{
                const row = document.createElement('tr');
                
                // Add urgent class if above average
                if (item.recommended_restock > {restock_df['recommended_restock'].mean()}) {{
                    row.classList.add('table-danger');
                }}
                
                // Create columns
                const productCode = document.createElement('td');
                productCode.textContent = item.StockCode || item.index;
                
                const description = document.createElement('td');
                description.textContent = item.Description || 'Unknown';
                
                const demand = document.createElement('td');
                demand.textContent = item.predicted_weekly_demand || item.predicted_demand || 'N/A';
                
                const currentStock = document.createElement('td');
                currentStock.textContent = item.current_stock || 'N/A';
                
                const safetyStock = document.createElement('td');
                safetyStock.textContent = item.safety_stock || 'N/A';
                
                const restock = document.createElement('td');
                restock.textContent = item.recommended_restock;
                restock.style.fontWeight = 'bold';
                
                // Append all columns to the row
                row.appendChild(productCode);
                row.appendChild(description);
                row.appendChild(demand);
                row.appendChild(currentStock);
                row.appendChild(safetyStock);
                row.appendChild(restock);
                
                // Append row to table
                tableBody.appendChild(row);
            }});
        }}
        
        // Sort table
        function sortTable(columnIndex) {{
            const copyData = [...restockData];
            
            // Get column name based on index
            const columns = ['StockCode', 'Description', 'predicted_weekly_demand', 'current_stock', 'safety_stock', 'recommended_restock'];
            const column = columns[columnIndex];
            
            // Toggle sort order
            const currentSort = document.getElementById('restockTable').getAttribute('data-sort') || '';
            const [currentColumn, currentOrder] = currentSort.split('-');
            
            let newOrder = 'asc';
            if (currentColumn == columnIndex && currentOrder == 'asc') {{
                newOrder = 'desc';
            }}
            
            document.getElementById('restockTable').setAttribute('data-sort', `${{columnIndex}}-${{newOrder}}`);
            
            // Sort data
            copyData.sort((a, b) => {{
                const valA = a[column] || 0;
                const valB = b[column] || 0;
                
                if (newOrder === 'asc') {{
                    return valA > valB ? 1 : -1;
                }} else {{
                    return valA < valB ? 1 : -1;
                }}
            }});
            
            populateTable(copyData);
        }}
        
        // Filter table
        document.getElementById('tableSearch').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            const filteredData = restockData.filter(item => {{
                return (
                    (item.Description && item.Description.toLowerCase().includes(searchTerm)) ||
                    (item.StockCode && item.StockCode.toString().includes(searchTerm))
                );
            }});
            
            populateTable(filteredData);
        }});
        
        // Initial table load
        document.addEventListener('DOMContentLoaded', function() {{
            populateTable(restockData);
        }});
    </script>
</body>
</html>
"""
    
    return html

def _create_top_items_chart_base64(restock_df: pd.DataFrame) -> str:
    """Create a chart of top items to restock and return as base64 image"""
    # Take top 10 items for the graph
    top_items = restock_df.head(10).copy()
    
    plt.figure(figsize=(12, 8))
    
    # Create a horizontal bar chart with product descriptions and recommended restock quantities
    descriptions = [str(desc)[:30] + '...' if len(str(desc)) > 30 else str(desc) for desc in top_items['Description']]
    y_pos = np.arange(len(descriptions))
    
    # Use a color gradient based on restock quantity
    colors = plt.cm.YlOrRd(top_items['recommended_restock'] / top_items['recommended_restock'].max())
    
    plt.barh(y_pos, top_items['recommended_restock'], align='center', color=colors)
    plt.yticks(y_pos, descriptions)
    plt.xlabel('Recommended Restock Quantity')
    plt.title('Top 10 Items to Restock')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def _create_restock_distribution_chart_base64(restock_df: pd.DataFrame) -> str:
    """Create a chart showing the distribution of restock quantities"""
    plt.figure(figsize=(12, 8))
    
    # Define restock ranges
    bins = [0, 10, 50, 100, 200, 500, 1000, restock_df['recommended_restock'].max() + 1]
    labels = ['0-10', '11-50', '51-100', '101-200', '201-500', '501-1000', '>1000']
    
    # Create histogram
    n, bins, patches = plt.hist(restock_df['recommended_restock'], bins=bins, edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Restock Quantity Range')
    plt.ylabel('Number of Products')
    plt.title('Distribution of Recommended Restock Quantities')
    
    # Add text labels above each bar
    for i in range(len(n)):
        if n[i] > 0:  # Only add label if the bar exists
            plt.text(bins[i] + (bins[i+1]-bins[i])/2, n[i] + 0.5, int(n[i]), 
                     ha='center', va='bottom')
    
    # Set x-ticks to the center of each bin
    plt.xticks([(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)], labels)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def save_reports(restock_df: pd.DataFrame, 
                 feature_data: pd.DataFrame = None,
                 md_output_path: str = "inventory_summary.md",
                 html_output_path: str = "inventory_dashboard.html",
                 pdf_output_path: str = None) -> Tuple[str, str, Optional[str]]:
    """
    Generate and save markdown summary, HTML dashboard, and optionally PDF report
    
    Args:
        restock_df: DataFrame containing restock recommendations
        feature_data: Optional DataFrame with historical sales data
        md_output_path: Path to save the markdown report
        html_output_path: Path to save the HTML dashboard
        pdf_output_path: Optional path to save the PDF report
        
    Returns:
        Tuple of (markdown_path, html_path, pdf_path) with the saved file paths
    """
    # Generate reports
    md_report = generate_markdown_summary(restock_df)
    html_dashboard = generate_html_dashboard(restock_df, feature_data)
    
    # Save markdown report
    with open(md_output_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
        
    # Save HTML dashboard
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html_dashboard)
    
    # Generate PDF if requested
    pdf_path = None
    if pdf_output_path:
        pdf_path = markdown_to_pdf(md_report, pdf_output_path)
        
    print(f"Reports saved: {md_output_path} and {html_output_path}" + 
          (f" and {pdf_path}" if pdf_path else ""))
    return md_output_path, html_output_path, pdf_path

def markdown_to_pdf(markdown_content: str, output_path: str) -> Optional[str]:
    """
    Convert markdown content to PDF
    
    Args:
        markdown_content: The markdown content to convert
        output_path: Path to save the PDF
        
    Returns:
        Path to the saved PDF file or None if conversion failed
    """
    try:
        # Try to import required libraries
        import markdown
        from weasyprint import HTML, CSS
        from tempfile import NamedTemporaryFile
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content, 
            extensions=['tables', 'markdown.extensions.tables']
        )
        
        # Create a complete HTML document with styling
        styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Management Report</title>
    <style>
        @page {{ margin: 1cm; }}
        body {{ 
            font-family: 'Helvetica', 'Arial', sans-serif; 
            font-size: 10pt;
            line-height: 1.4;
            margin: 0;
            padding: 0;
        }}
        h1 {{ font-size: 18pt; color: #2c3e50; margin-top: 20pt; }}
        h2 {{ font-size: 14pt; color: #3498db; margin-top: 16pt; }}
        h3 {{ font-size: 12pt; color: #2980b9; margin-top: 12pt; }}
        p {{ margin-top: 8pt; margin-bottom: 8pt; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16pt 0; }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 6pt; 
            font-size: 9pt;
            overflow-wrap: break-word;
            word-wrap: break-word;
        }}
        th {{ 
            background-color: #f2f2f2; 
            font-weight: bold;
            text-align: left;
        }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""
        
        # Use WeasyPrint to convert HTML to PDF
        with NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            tmp_file.write(styled_html.encode('utf-8'))
            tmp_path = tmp_file.name
        
        HTML(filename=tmp_path).write_pdf(output_path)
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        print(f"PDF report successfully created: {output_path}")
        return output_path
    
    except ImportError as e:
        print(f"Error importing required libraries for PDF conversion: {e}")
        print("To enable PDF generation, install required packages:")
        print("pip install markdown weasyprint")
        return None
    
    except Exception as e:
        print(f"Error converting markdown to PDF: {e}")
        return None
