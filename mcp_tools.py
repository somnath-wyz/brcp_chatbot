from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib

import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional
import logging
import json
import ast
import pandas as pd
import uuid
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import os
import clickhouse_connect
from table_column_names_meanings import column_meanings
from chart_creator import ChartCreator

mcp = FastMCP("wizard_chatbot_mcp")
export_dir = Path("exports")
chart_creator = ChartCreator(export_dir=export_dir)

@mcp.tool()
def create_chat(chart_data: Dict[str, Any], chart_type: str) -> Optional[str]:
    """
    Create a chart using matplotlib and save it as an image.
    
    Args:
        chart_data: Dictionary containing chart configuration with the following format:
            For pie charts:
                {
                    "labels": ["Label1", "Label2", "Label3"],  # Required
                    "values": [10, 20, 30],                    # Required
                    "colors": ["red", "blue", "green"],        # Optional
                    "title": "My Pie Chart"                    # Optional
                }
            
            For bar charts (supports multiple formats):
                Format 1:
                {
                    "x_labels": ["A", "B", "C"],               # Required
                    "y_values": [10, 20, 30],                  # Required
                    "color": "steelblue",                      # Optional
                    "x_label": "Categories",                   # Optional
                    "y_label": "Values",                       # Optional
                    "title": "My Bar Chart"                    # Optional
                }
                
                Format 2 (same as pie):
                {
                    "labels": ["A", "B", "C"],                 # Required
                    "values": [10, 20, 30],                    # Required
                    "color": "steelblue",                      # Optional
                    "x_label": "Categories",                   # Optional
                    "y_label": "Values",                       # Optional
                    "title": "My Bar Chart"                    # Optional
                }
                
                Format 3 (data as list):
                {
                    "data": [("A", 10), ("B", 20), ("C", 30)] # Required (tuples)
                    # OR
                    "data": [{"name": "A", "value": 10}, ...] # Required (dicts)
                    # OR
                    "data": [10, 20, 30],                     # Required (values only)
                    "color": "steelblue",                     # Optional
                    "x_label": "Categories",                  # Optional
                    "y_label": "Values",                      # Optional
                    "title": "My Bar Chart"                   # Optional
                }
            
            For line charts:
                {
                    "x_values": [1, 2, 3, 4],                 # Required
                    "y_values": [10, 20, 15, 25],             # Required
                    "x_label": "Time",                        # Optional
                    "y_label": "Values",                      # Optional
                    "title": "My Line Chart"                  # Optional
                }
            
            For histogram:
                {
                    "data": [1, 2, 2, 3, 3, 3, 4, 4, 5],     # Required
                    "bins": 10,                               # Optional (default: 10)
                    "x_label": "Values",                      # Optional
                    "title": "My Histogram"                   # Optional
                }
        
        chart_type: Type of chart ("pie", "bar", "line", "histogram")
        
    Returns:
        Path to the saved chart image
        
    Raises:
        ValueError: If chart_data format is invalid or required fields are missing
        TypeError: If chart_type is not supported or data types are incorrect
    """
    filename = chart_creator.create_chart(chart_data, chart_type)
    return f"Chart image created successfully: {filename} (Download: /downloads/{filename})"

@mcp.tool()
def export_query_to_csv(query: str, title: str) -> str:
    """
    Run a query in the database and export the response data in csv.

    Args:
        query: SQL query to get the data
        title: Title for the export (for metadata)

    Returns:
        Path to the created CSV file
    """
    try:
        client = clickhouse_connect.get_client(
            host=os.environ.get("db_host", "localhost"), 
            port=int(os.environ.get("db_port", "8123")), 
            username=os.environ.get("db_user", "default"), 
            password=os.environ.get('db_password', ''), 
            database=os.environ.get('db_name', 'Cred')
        )
        result = client.query(query)
        rows_as_dicts = [dict(zip(result.column_names, row)) for row in result.result_rows]

        df = pd.DataFrame(rows_as_dicts)
        
        # Generate filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_').lower()
        csv_filename = f"{safe_title}_{uuid.uuid4().hex[:8]}.csv"
        
        export_dir.mkdir(exist_ok=True)
        csv_path = export_dir / csv_filename
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return f"CSV file created successfully: {csv_filename} (Download: /downloads/{csv_filename})"
    except Exception as e:
        logging.error(f"Error creating CSV export: {e}")
        return f"Error creating CSV file: {str(e)}"

@mcp.tool()
def create_pdf_report(
    content_structure: str, 
    filename: str, title: str="Data Report", 
    description: str="Generated report with the charts and analysis"
    ) -> str:
    """
    Creates a PDF report with charts, tables, and rich text.
    
    Args:
        content_structure: JSON string with report structure including:
            - sections: List of sections with type and content
            - data: Raw data for tables/charts
            - charts: Chart configurations
        filename: Desired filename (without extension)
        title: Title for the PDF report
        description: Description of the report content
    
    Expected content_structure format:
    {
        "sections": [
            {
                "type": "text",
                "content": "Introduction text here...",
                "style": "normal" // or "heading", "subheading"
            },
            {
                "type": "chart",
                "chart_type": "pie", // or "bar", "line", "histogram"
                "data": {
                    "labels": ["Label1", "Label2"],
                    "values": [30, 70],
                    "title": "Chart Title"
                }
            },
            {
                "type": "table",
                "data": [...],
                "headers": [...],
                "title": "Table Title"
            }
        ]
    }
    
    Returns:
        Path to the created PDF file
    """
    try:
        # Parse the content structure
        try:
            structure = json.loads(content_structure) if isinstance(content_structure, str) else content_structure
        except json.JSONDecodeError:
            # Fallback to simple table format
            structure = {
                "sections": [{
                    "type": "table",
                    "data": json.loads(content_structure),
                    "headers": ["Column 1", "Column 2"],
                    "title": "Data Table"
                }]
            }
        
        # Generate unique filename
        pdf_filename = f"{filename}_{uuid.uuid4().hex[:8]}.pdf"
        pdf_path = export_dir / pdf_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=0.5*inch)
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=1,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=1,
            textColor=colors.grey
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.darkslategray,
            fontName='Helvetica-Bold'
        )
        
        # Add header
        elements.append(Paragraph(title, title_style))
        elements.append(Paragraph(description, subtitle_style))
        
        # Add metadata
        metadata = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        elements.append(Paragraph(metadata, styles['Normal']))
        elements.append(Spacer(1, 30))
        
        # Process sections
        sections = structure.get("sections", [])
        
        for i, section in enumerate(sections):
            section_type = section.get("type", "text")
            
            if section_type == "text":
                content = section.get("content", "")
                style_name = section.get("style", "normal")
                
                if style_name == "heading":
                    elements.append(Paragraph(content, heading_style))
                elif style_name == "subheading":
                    elements.append(Paragraph(content, subheading_style))
                else:
                    elements.append(Paragraph(content, styles['Normal']))
                
                elements.append(Spacer(1, 12))
                
            elif section_type == "chart":
                chart_type = section.get("chart_type", "pie")
                chart_data = section.get("data", {})
                
                # Create chart using matplotlib
                chart_filename = chart_creator.create_chart(chart_data, chart_type)
                
                if not chart_filename:
                    chart_filename = ""

                chart_path = export_dir / chart_filename
                
                if chart_path and os.path.exists(chart_path):
                    # Add chart title if provided
                    chart_title = chart_data.get("title", f"{chart_type.title()} Chart")
                    elements.append(Paragraph(chart_title, heading_style))
                    
                    # Add the chart image
                    img = Image(chart_path, width=6*inch, height=3.6*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 20))
                
            elif section_type == "table":
                table_data = section.get("data", [])
                headers = section.get("headers", [])
                table_title = section.get("title", "Data Table")
                
                if table_data and headers:
                    # Add table title
                    elements.append(Paragraph(table_title, heading_style))
                    
                    # Prepare table data
                    formatted_table_data = [headers]
                    
                    for row in table_data:
                        if isinstance(row, dict):
                            formatted_table_data.append([str(row.get(header, '')) for header in headers])
                        elif isinstance(row, list):
                            formatted_table_data.append([str(cell) for cell in row])
                    
                    # Create table
                    table = Table(formatted_table_data)
                    table.setStyle(TableStyle([
                        # Header styling
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        
                        # Data styling
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                    ]))
                    
                    elements.append(table)
                    elements.append(Spacer(1, 20))
            
            elif section_type == "page_break":
                elements.append(PageBreak())
            
            elif section_type == "spacer":
                height = section.get("height", 20)
                elements.append(Spacer(1, height))
        
        # Build PDF
        doc.build(elements)
        
        logging.info(f"Enhanced PDF report created: {pdf_filename}")
        
        return f"Enhanced PDF report created successfully: {pdf_filename} (Download: /downloads/{pdf_filename})"
        
    except Exception as e:
        logging.error(f"Error creating enhanced PDF report: {e}")
        return f"Error creating PDF report: {str(e)}"
    
@mcp.tool()
def analyze_data(data: str, analysis_type: str = "summary") -> str:
    """
    Performs basic analysis on the data.
    
    Args:
        data: JSON string containing the data
        analysis_type: Type of analysis (summary, stats, trends)
    
    Returns:
        Analysis results as a string
    """
    try:
        try:
            rows = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError as e:
            parsed = ast.literal_eval(data)
            if isinstance(parsed, list) and all(isinstance(t, tuple) for t in parsed):
                rows = []
                for tup in parsed:
                    rows.append({
                        f"col_{i+1}": val for i, val in enumerate(tup)
                    })
        
        print(f"Analyzing {len(rows)} rows of data...", rows)

        if not rows:
            return "No data available for analysis."
        
        df = pd.DataFrame(rows)
        
        if analysis_type == "summary":
            return f"Data Summary:\n- Total records: {len(df)}\n- Columns: {list(df.columns)}\n- Data types: {df.dtypes.to_dict()}"
        elif analysis_type == "stats":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return f"Statistical Summary:\n{df[numeric_cols].describe().to_string()}"
            else:
                return "No numeric columns found for statistical analysis."
        
        return "Analysis completed."
        
    except Exception as e:
        return f"Error in data analysis: {str(e)}" 

@mcp.tool()
def get_table_column_names_meaning(table_names: List[str]) -> dict[str, str]:
    """
    Retrieves the meanings of columns for the specified table names.
    ALWAYS use this tool before writing query a database query.

    Args:
        table_names (List[str]): A list of table names for which to retrieve column meanings.

    Returns:
        dict[str, str]: A dictionary mapping each table name to its column meanings.

    Raises:
        KeyError: If a table name is not found in the column_meanings dictionary.
    """
    try:
        return {table_name: column_meanings[table_name] for table_name in table_names}
    except Exception as e:
        logging.error(e)
        raise Exception(f"Failed to get column meanings: {str(e)}")

if __name__ == "__main__":
    mcp.run(transport='stdio')
