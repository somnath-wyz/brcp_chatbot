import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib

import logging
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt


class ChartCreator:
    """
    A class for creating various types of charts using matplotlib.
    
    Supported chart types: pie, bar, line, histogram
    """
    
    VALID_CHART_TYPES = ["pie", "bar", "line", "histogram"]
    
    def __init__(self, export_dir: Path, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the chart creator.
        
        Args:
            export_dir: Directory where chart images will be saved
            figsize: Figure size as (width, height) tuple
        """
        self.export_dir = export_dir
        self.figsize = figsize
    
    def create_chart(self, chart_data: Dict[str, Any], chart_type: str) -> Optional[str]:
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
        self._validate_chart_type(chart_type)
        self._validate_chart_data_format(chart_data, chart_type)
        
        try:
            return self._create_and_save_chart(chart_data, chart_type)
        except Exception as e:
            logging.error(f"Error creating matplotlib chart: {str(e)}")
            plt.close()  # Ensure we close the figure even on error
            raise e
    
    def _validate_chart_type(self, chart_type: str) -> None:
        """Validate that the chart type is supported."""
        if chart_type not in self.VALID_CHART_TYPES:
            raise ValueError(f"Invalid chart_type '{chart_type}'. Must be one of: {self.VALID_CHART_TYPES}")
    
    def _validate_chart_data_format(self, chart_data: Dict[str, Any], chart_type: str) -> None:
        """Validate chart_data format based on chart type."""
        if not isinstance(chart_data, dict):
            raise TypeError("chart_data must be a dictionary")
        
        validator_method = getattr(self, f'_validate_{chart_type}_data')
        validator_method(chart_data)
    
    def _validate_list_not_empty(self, data: Any, field_name: str) -> None:
        """Validate that data is a non-empty list."""
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"{field_name} must be a non-empty list")
    
    def _validate_numeric_list(self, data: List, field_name: str) -> None:
        """Validate that all elements in the list are numeric."""
        if not all(isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '').replace('-', '').isdigit()) for x in data):
            raise ValueError(f"{field_name} must contain only numeric values")
    
    def _validate_pie_data(self, chart_data: Dict[str, Any]) -> None:
        """Validate data format for pie charts."""
        if "labels" not in chart_data:
            raise ValueError("Pie chart requires 'labels' field")
        if "values" not in chart_data:
            raise ValueError("Pie chart requires 'values' field")
        
        labels = chart_data["labels"]
        values = chart_data["values"]
        
        self._validate_list_not_empty(labels, "labels")
        self._validate_list_not_empty(values, "values")
        self._validate_numeric_list(values, "values")
        
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length")
    
    def _validate_bar_data(self, chart_data: Dict[str, Any]) -> None:
        """Validate data format for bar charts."""
        has_x_y_format = "x_labels" in chart_data and "y_values" in chart_data
        has_labels_values = "labels" in chart_data and "values" in chart_data
        has_data_format = "data" in chart_data
        
        if not (has_x_y_format or has_labels_values or has_data_format):
            raise ValueError(
                "Bar chart requires one of: "
                "('x_labels' and 'y_values'), "
                "('labels' and 'values'), "
                "or 'data' field"
            )
        
        if has_x_y_format:
            self._validate_x_y_format(chart_data)
        elif has_labels_values:
            self._validate_labels_values_format(chart_data)
        elif has_data_format:
            self._validate_data_format(chart_data)
    
    def _validate_x_y_format(self, chart_data: Dict[str, Any]) -> None:
        """Validate x_labels and y_values format."""
        x_labels = chart_data["x_labels"]
        y_values = chart_data["y_values"]
        
        self._validate_list_not_empty(x_labels, "x_labels")
        self._validate_list_not_empty(y_values, "y_values")
        self._validate_numeric_list(y_values, "y_values")
        
        if len(x_labels) != len(y_values):
            raise ValueError("x_labels and y_values must have the same length")
    
    def _validate_labels_values_format(self, chart_data: Dict[str, Any]) -> None:
        """Validate labels and values format."""
        labels = chart_data["labels"]
        values = chart_data["values"]
        
        self._validate_list_not_empty(labels, "labels")
        self._validate_list_not_empty(values, "values")
        self._validate_numeric_list(values, "values")
        
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length")
    
    def _validate_data_format(self, chart_data: Dict[str, Any]) -> None:
        """Validate data format for bar charts."""
        data = chart_data["data"]
        self._validate_list_not_empty(data, "data")
        
        if isinstance(data[0], tuple):
            if not all(isinstance(item, tuple) and len(item) == 2 for item in data):
                raise ValueError("data tuples must have exactly 2 elements each")
            second_elements = [item[1] for item in data]
            self._validate_numeric_list(second_elements, "data tuple values")
        elif isinstance(data[0], dict):
            if not all(isinstance(item, dict) and len(item) >= 2 for item in data):
                raise ValueError("data dictionaries must have at least 2 keys each")
            keys = list(data[0].keys())
            if not all(set(item.keys()) == set(keys) for item in data):
                raise ValueError("all data dictionaries must have the same keys")
            values = [item[keys[1]] for item in data]
            self._validate_numeric_list(values, "data dictionary values")
        else:
            self._validate_numeric_list(data, "data values")
    
    def _validate_line_data(self, chart_data: Dict[str, Any]) -> None:
        """Validate data format for line charts."""
        if "x_values" not in chart_data:
            raise ValueError("Line chart requires 'x_values' field")
        if "y_values" not in chart_data:
            raise ValueError("Line chart requires 'y_values' field")
        
        x_values = chart_data["x_values"]
        y_values = chart_data["y_values"]
        
        self._validate_list_not_empty(x_values, "x_values")
        self._validate_list_not_empty(y_values, "y_values")
        self._validate_numeric_list(x_values, "x_values")
        self._validate_numeric_list(y_values, "y_values")
        
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length")
    
    def _validate_histogram_data(self, chart_data: Dict[str, Any]) -> None:
        """Validate data format for histograms."""
        if "data" not in chart_data:
            raise ValueError("Histogram requires 'data' field")
        
        data = chart_data["data"]
        self._validate_list_not_empty(data, "data")
        self._validate_numeric_list(data, "data")
        
        if "bins" in chart_data:
            bins = chart_data["bins"]
            if not isinstance(bins, int) or bins <= 0:
                raise ValueError("bins must be a positive integer")
    
    def _create_and_save_chart(self, chart_data: Dict[str, Any], chart_type: str) -> str:
        """Create the chart and save it to file."""
        # Set style for better looking charts
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create the appropriate chart type
        chart_creator = getattr(self, f'_create_{chart_type}_chart')
        chart_creator(ax, chart_data)
        
        # Set title
        title = chart_data.get("title", f"{chart_type.title()} Chart")
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Improve layout
        plt.tight_layout()
        if chart_type == "bar":
            plt.subplots_adjust(bottom=0.2)
        
        # Save chart
        chart_filename = f"chart_{uuid.uuid4().hex[:8]}.png"
        chart_path = self.export_dir / chart_filename
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        return chart_filename
    
    def _create_pie_chart(self, ax, chart_data: Dict[str, Any]) -> None:
        """Create a pie chart."""
        labels = chart_data["labels"]
        values = chart_data["values"]
        colors_list = chart_data.get("colors", plt.cm.Set3.colors[:len(values)]) # type: ignore
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%', 
            colors=colors_list, startangle=90
        )
        
        # Enhance text appearance with dynamic color selection
        for i, autotext in enumerate(autotexts):
            # Get the color of the wedge
            wedge_color = wedges[i].get_facecolor()
            # Calculate text color based on wedge brightness
            text_color = self._get_contrasting_text_color(wedge_color)
            autotext.set_color(text_color)
            autotext.set_weight('bold')

    def _get_contrasting_text_color(self, background_color) -> str:
        """
        Calculate contrasting text color (black or white) based on background color brightness.
        
        Args:
            background_color: RGB tuple or matplotlib color
            
        Returns:
            'black' or 'white' depending on which provides better contrast
        """
        # Convert color to RGB if it's not already
        if hasattr(background_color, '__len__') and len(background_color) >= 3:
            r, g, b = background_color[:3]
        else:
            # Fallback for other color formats
            return 'black'
        
        # Calculate relative luminance using the standard formula
        # Values are normalized to 0-1 range
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Return black text for light backgrounds, white text for dark backgrounds
        return 'black' if luminance > 0.5 else 'white'
    
    def _create_bar_chart(self, ax, chart_data: Dict[str, Any]) -> None:
        """Create a bar chart."""
        x_labels, y_values = self._extract_bar_data(chart_data)
        
        # Convert y_values to numeric
        y_values = self._convert_to_numeric(y_values)
        
        color = chart_data.get("color", "steelblue")
        
        # Create bars
        bars = ax.bar(range(len(x_labels)), y_values, color=color, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        self._add_bar_labels(ax, bars, y_values)
        
        # Set formatting
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel(chart_data.get("x_label", "Categories"), fontsize=12, fontweight='bold')
        ax.set_ylabel(chart_data.get("y_label", "Values"), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    
    def _create_line_chart(self, ax, chart_data: Dict[str, Any]) -> None:
        """Create a line chart."""
        x_values = chart_data["x_values"]
        y_values = chart_data["y_values"]
        
        ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel(chart_data.get("x_label", "X Axis"))
        ax.set_ylabel(chart_data.get("y_label", "Y Axis"))
        ax.grid(True, alpha=0.3)
    
    def _create_histogram_chart(self, ax, chart_data: Dict[str, Any]) -> None:
        """Create a histogram."""
        data = chart_data["data"]
        bins = chart_data.get("bins", 10)
        
        ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel(chart_data.get("x_label", "Values"))
        ax.set_ylabel("Frequency")
    
    def _extract_bar_data(self, chart_data: Dict[str, Any]) -> Tuple[List, List]:
        """Extract x_labels and y_values from various bar chart formats."""
        if "x_labels" in chart_data and "y_values" in chart_data:
            return chart_data["x_labels"], chart_data["y_values"]
        elif "labels" in chart_data and "values" in chart_data:
            return chart_data["labels"], chart_data["values"]
        elif "data" in chart_data:
            data = chart_data["data"]
            if isinstance(data[0], tuple):
                return [item[0] for item in data], [item[1] for item in data]
            elif isinstance(data[0], dict):
                keys = list(data[0].keys())
                return [item[keys[0]] for item in data], [item[keys[1]] for item in data]
            else:
                return [f"Item {i+1}" for i in range(len(data))], data
        else:
            return ["No Data"], [0]
    
    def _convert_to_numeric(self, values: List) -> List[Union[int, float]]:
        """Convert string values to numeric where possible."""
        try:
            converted = []
            for val in values:
                if isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit():
                    converted.append(float(val))
                elif isinstance(val, (int, float)):
                    converted.append(val)
                else:
                    converted.append(0)
            return converted
        except:
            return [1] * len(values)
    
    def _add_bar_labels(self, ax, bars, y_values: List) -> None:
        """Add value labels on top of bars."""
        for bar, value in zip(bars, y_values):
            height = bar.get_height()
            label_text = f'{value:.1f}' if isinstance(value, float) and value % 1 != 0 else f'{int(value)}'
            ax.text(bar.get_x() + bar.get_width()/2., height + max(y_values) * 0.01,
                   label_text, ha='center', va='bottom', fontweight='bold', fontsize=10)
