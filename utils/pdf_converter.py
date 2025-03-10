"""
Utility module for converting markdown to PDF
"""
import os
import tempfile
from typing import Optional

def convert_markdown_to_pdf(markdown_content: str, output_path: str) -> Optional[str]:
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
        
        # Write HTML to temporary file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            tmp_file.write(styled_html.encode('utf-8'))
            tmp_path = tmp_file.name
        
        # Convert HTML to PDF
        HTML(filename=tmp_path).write_pdf(output_path)
        
        # Clean up temp file
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

# Example usage
if __name__ == "__main__":
    # Sample markdown content
    sample_content = """# Sample Report
    
## Overview
This is a sample report to test PDF conversion.

## Data Table
| Item | Value |
|------|-------|
| A    | 10    |
| B    | 20    |
| C    | 30    |

## Conclusion
This demonstrates conversion from markdown to PDF.
"""
    
    # Try to convert the sample content
    convert_markdown_to_pdf(sample_content, "sample_report.pdf")
