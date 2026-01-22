"""
Utility functions for CSV analysis and DataFrame description.
"""

# Standard library imports
import os
import json

# Third-party imports
import numpy as np
import pandas as pd

# System prompts for code generation
system_prompt0 = """
        You are a Python code generation agent specialized in data analysis.
        Your task is to generate Python code that answers the user's question about their data.
        
        The code you generate will be executed in a secure Docker container with these libraries available:
        - pandas
        - numpy
        - matplotlib
        - seaborn
        - scikit-learn
        - statsmodels
        
        Generate complete, well-commented Python code that:
        1. Loads the data using the filename only, DO NOT sample the data
        2. analyze and fix data types and values, because you need to use them in the code
        3. Select the columns or the rows that are most relevant to the user's question, Pay attention to colmun types and values
        4. Performs appropriate analysis to answer the user's question
        5. If required, train a moddel(Classification/Regression/Clustering) and use it to answer the question
        6. Creates visualizations when helpful
        7. Prints clear explanations of findings
        
        The code will run in a container with no network access, so don't include any external API calls or downloads.
        """
system_prompt1 = """
**System Prompt: Python Data Analysis Code Generation Agent**

You are a specialized Python code generation agent focused on data analysis. Your task is to generate complete, well-commented Python code that precisely answers the user's data-related question using the given dataset. The environment where your code will run is a secure Docker container with the following libraries available:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

When generating your code, please adhere to the following instructions:

1. **Data Loading:**  
   - Load the dataset using the provided filename (do not sample or subset the data initially).  
   - Ensure that the data is loaded into a pandas DataFrame.

2. **Data Cleaning and Preprocessing:**  
   - Analyze the dataset's columns and data types.  
   - Identify and fix any inconsistencies in data types or erroneous/missing values that may impact the analysis.  
   - Document any assumptions or transformations performed during cleaning.

3. **Feature Selection:**  
   - Carefully select the columns or rows most relevant to the user's question, Pay attention to colmun types and values.  
   - Justify your selection by considering data types, values, and relevance to the analysis question.

4. **Exploratory Data Analysis (EDA):**  
   - Provide a detailed exploratory analysis to understand key trends, correlations, and distributions in the data.  
   - Create visualizations (e.g., histograms, scatter plots, box plots) to support your findings.
   - Create better visualizations using seaborn and matplotlib.

5. **Analytical Approach:**  
   - Depending on the user's question, decide on an appropriate analysis strategy, such as statistical tests, regression, classification, or clustering.  
   - If the task requires a predictive model, choose the correct model type (Classification/Regression/Clustering) and explain why it is appropriate.

6. **Model Training and Evaluation (if applicable):**  
   - Train the selected model using the relevant portion of the dataset.  
   - Evaluate the model's performance with appropriate metrics and include visualizations of the results (e.g., confusion matrix, ROC curve, residual plots).  
   - Clearly explain the outcomes and the model's implications regarding the user's question.

7. **Visualization and Explanation:**  
   - Generate visualizations where helpful to illustrate key insights and results.  
   - Include clear, inline comments and print statements to explain the findings and the significance of each analysis step.

8. **Constraints:**  
   - Ensure that your code is complete and self-contained, ready to be executed in the Docker container with no external network access.  
   - Do not include any external API calls, downloads, or dependencies beyond the specified libraries.

Your final output should be a complete, runnable Python script that not only performs the necessary analysis but also educates the user through comprehensive commentary and explanations.
"""
system_prompt2 = """
**System Prompt: Python Visualization Code Generation Agent**

You are a specialized Python code generation agent focused on creating visualizations from datasets.
Your task is to generate complete, well-commented Python code that creates visualizations based on the user's instructions.

The code will run in a secure Docker container with the following libraries available:
- pandas
- numpy
- matplotlib
- seaborn

Instructions:
1. **Data Loading:**
   - Load the dataset using the provided filename (do not sample the data initially).
   - Ensure that the dataset is loaded into a pandas DataFrame.

2. **Data Cleaning and Preprocessing:**
   - Inspect the dataset's columns and data types.
   - Identify and fix any inconsistencies in data types or missing values that might affect the visualization.
   - Include inline comments describing any cleaning or transformation steps.

3. **User-Directed Visualization:**
   - Focus on generating visualizations as directed by the user's command (e.g., "plot column for year", "plot sales over time").
   - Select the appropriate plot type based on the data and the command (e.g., line plot for time series data, bar chart for categorical data, scatter plot for continuous variables).
   - Add titles, axis labels, and legends as needed to make the visualization self-explanatory.

4. **Visualization Generation:**
   - Use matplotlib and seaborn to generate the plots.
   - Ensure the plots are clear and informative.

5. **Code Readability and Explanations:**
   - Include detailed inline comments and print statements explaining each step of the process.
   - Make sure the code is complete, self-contained, and ready to run without external API calls or downloads.

Your final output should be a complete, runnable Python script that loads the dataset,
performs necessary preprocessing, and creates the requested visualization with clear commentary.
"""

def analyze_csv(file_path):
    """
    Analyze a CSV file and return a structured summary suitable for an LLM prompt.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        str: Formatted analysis of the CSV file
    """
    print(f"Analyzing {file_path}...")
    
    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"
    
    # Basic file info
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    row_count = len(df)
    col_count = len(df.columns)
    
    # Detect date columns and convert them
    date_columns = []
    for col in df.columns:
        # Try to infer if this is a date column by name
        if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
            try:
                df[col] = pd.to_datetime(df[col])
                date_columns.append(col)
            except (ValueError, TypeError):
                # Column cannot be converted to datetime, skip it
                pass
    
    # Get time range if date columns exist
    time_range = "N/A"
    if date_columns:
        for col in date_columns:
            min_date = df[col].min()
            max_date = df[col].max()
            if pd.notna(min_date) and pd.notna(max_date):
                time_range = f"{min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"
                break
    
    # Column analysis
    column_info = []
    for col in df.columns:
        # Determine data type
        if col in date_columns:
            dtype = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                dtype = "integer"
            else:
                dtype = "float"
        elif pd.api.types.is_bool_dtype(df[col]):
            dtype = "boolean"
        else:
            dtype = "string"
            
        # Get unique values info
        unique_count = df[col].nunique()
        unique_ratio = unique_count / row_count if row_count > 0 else 0
        
        # Determine if column could be categorical
        is_categorical = unique_ratio < 0.1 and unique_count < 20
        
        # Get stats for numeric columns
        stats = {}
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "median": df[col].median()
            }
        
        # Missing values
        missing_count = df[col].isna().sum()
        missing_percentage = (missing_count / row_count) * 100 if row_count > 0 else 0
        
        # Sample values (up to 5)
        if is_categorical:
            sample_values = df[col].dropna().unique()[:5].tolist()
        else:
            sample_values = df[col].dropna().head(5).tolist()
        
        # Convert sample values to strings
        if dtype == "datetime":
            sample_values = [str(val)[:10] for val in sample_values]
        else:
            sample_values = [str(val) for val in sample_values]
            
        column_info.append({
            "name": col,
            "dtype": dtype,
            "missing_count": int(missing_count),
            "missing_percentage": round(missing_percentage, 2),
            "unique_count": int(unique_count),
            "is_categorical": is_categorical,
            "stats": {k: float(v) if not pd.isna(v) else None for k, v in stats.items()} if stats else {},
            "sample_values": sample_values
        })
    
    # Get data sample as a formatted string
    sample_rows = min(10, row_count)
    data_sample = df.head(sample_rows)
    
    # Prepare the markdown table for data sample
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---" for _ in df.columns]) + " |"
    rows = []
    
    for _, row in data_sample.iterrows():
        formatted_row = []
        for col, val in row.items():
            if pd.isna(val):
                formatted_val = "null"
            elif col in date_columns:
                try:
                    formatted_val = val.strftime('%Y-%m-%d')
                except (AttributeError, ValueError, TypeError):
                    formatted_val = str(val)
            else:
                formatted_val = str(val)
                
            # Truncate long values
            if len(formatted_val) > 20:
                formatted_val = formatted_val[:17] + "..."
                
            formatted_row.append(formatted_val)
            
        rows.append("| " + " | ".join(formatted_row) + " |")
    
    sample_table = "\n".join([header, separator] + rows)
    
    # Detect potential issues
    issues = []
    
    # Check for missing values
    cols_with_missing = [col["name"] for col in column_info if col["missing_percentage"] > 0]
    if cols_with_missing:
        missing_desc = ", ".join([f"{col} ({next(c['missing_percentage'] for c in column_info if c['name'] == col)}%)" 
                                  for col in cols_with_missing[:3]])
        if len(cols_with_missing) > 3:
            missing_desc += f", and {len(cols_with_missing) - 3} more columns"
        issues.append(f"Missing values in {missing_desc}")
    
    # Check for potential duplicates
    if df.duplicated().any():
        dupe_count = df.duplicated().sum()
        issues.append(f"Contains {dupe_count} duplicate rows ({(dupe_count/row_count)*100:.1f}% of dataset)")
    
    # Prepare recommendations based on the data
    recommendations = []
    
    # Check if there's time series data
    if date_columns:
        recommendations.append("Time series analysis could be performed on this dataset")
    
    # Check if there are categorical columns
    cat_cols = [col["name"] for col in column_info if col["is_categorical"]]
    if cat_cols:
        cat_desc = ", ".join(cat_cols[:3])
        if len(cat_cols) > 3:
            cat_desc += f", and {len(cat_cols) - 3} more"
        recommendations.append(f"Categorical analysis could be performed on: {cat_desc}")
    
    # Check if there are numeric columns for statistical analysis
    num_cols = [col["name"] for col in column_info if col["dtype"] in ["integer", "float"]]
    if len(num_cols) >= 2:
        recommendations.append("Correlation analysis could be performed between numeric columns")
    
    # Format the final output
    output = f"""# CSV File Analysis for LLM Prompt

## Basic Information
- **Filename**: {file_name}
- **Size**: {file_size:.2f} MB
- **Rows**: {row_count:,}
- **Columns**: {col_count}
- **Time Range**: {time_range}

## Column Structure
"""
    
    for col in column_info:
        stats_text = ""
        if col["dtype"] in ["integer", "float"] and col["stats"]:
            stats = col["stats"]
            stats_text = f" (min: {stats.get('min')}, max: {stats.get('max')}, mean: {stats.get('mean', 0):.2f})"
        
        missing_text = ""
        if col["missing_count"] > 0:
            missing_text = f" - {col['missing_percentage']}% missing values"
            
        sample_text = ""
        if col["is_categorical"]:
            sample_text = f" - Values include: {', '.join(col['sample_values'])}"
            
        output += f"- **{col['name']}** ({col['dtype']}): {col['unique_count']} unique values{stats_text}{missing_text}{sample_text}\n"
    
    output += "\n## Data Sample\n"
    output += sample_table
    
    if issues:
        output += "\n\n## Data Quality Issues\n"
        for issue in issues:
            output += f"- {issue}\n"
    
    if recommendations:
        output += "\n## Recommended Analysis Approaches\n"
        for rec in recommendations:
            output += f"- {rec}\n"
    
    # Template for LLM prompt
#     output += """
# Replace the placeholders in curly braces with specific details about your analysis goals.
# """
    return output

def generate_detailed_dataframe_description(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive descriptive summary of a pandas DataFrame.
    
    This function creates an in-depth text summary of the dataset, including:
      - Overall dataset dimensions.
      - Sample data from the head and tail of the dataset with inferred data types.
      - For each column:
          • Column name and data type.
          • Inferred data type (if different from stored type).
          • Count and percentage of non-null and missing values.
          • Number of unique values.
          • For numeric columns:
              - Summary statistics: count, mean, std, min, 25th percentile, median, 75th percentile, max.
              - Skewness and kurtosis.
              - A brief note on the range and potential outlier hints.
          • For non-numeric columns:
              - A preview of up to 5 unique sample values.
              - The most frequent values with their counts.
      - Additional dataset-wide insights:
          • List of columns with missing values and their missing percentage.
          • A correlation matrix for numeric columns (if more than one numeric column exists) as a textual summary.
          
    This detailed description aims to provide an AI or data analyst with ample context about the dataset for creating visualizations,
    projections, predictions, or forecasts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to be described.
    
    Returns:
    --------
    description : str
        A formatted string that thoroughly describes the DataFrame.
    """
    lines = []
    
    # Overall dataset info
    total_rows, total_columns = df.shape
    lines.append(f"Dataset Summary:")
    lines.append(f"  - Total rows: {total_rows}")
    lines.append(f"  - Total columns: {total_columns}\n")
    
    # Function to infer data type from a series
    def infer_data_type(series):
        # Skip if already numeric or datetime
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_dtype(series):
            return None
        
        # Only process string/object columns
        if not pd.api.types.is_object_dtype(series):
            return None
        
        # Sample non-null values for testing
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return None
        
        # Try to convert to numeric
        numeric_conversion = pd.to_numeric(sample, errors='coerce')
        if numeric_conversion.notna().all():
            # Check if it's integer-like
            if (numeric_conversion % 1 == 0).all():
                return 'integer'
            else:
                return 'float'
        
        # Try to convert to datetime
        datetime_conversion = pd.to_datetime(sample, errors='coerce')
        if datetime_conversion.notna().all():
            return 'datetime'
        
        # Check for boolean-like values
        bool_values = {'true', 'false', 'yes', 'no', 'y', 'n', 't', 'f', '1', '0'}
        lower_sample = sample.astype(str).str.lower()
        if lower_sample.isin(bool_values).all():
            return 'boolean'
        
        # If we can't infer a specific type, return None
        return None
    
    # Add sample data from head and tail with inferred data types
    lines.append("Sample Data:")
    
    # Get sample rows from head
    lines.append("  - First 3 rows:")
    head_sample = df.head(3)
    for i, row in head_sample.iterrows():
        lines.append(f"    Row {i}:")
        for col, val in row.items():
            # Infer data type for this specific value
            if pd.isna(val):
                val_type = "missing"
                val_str = "NA"
            elif isinstance(val, (int, np.integer)):
                val_type = "integer"
                val_str = str(val)
            elif isinstance(val, (float, np.floating)):
                val_type = "float"
                val_str = str(val)
            elif isinstance(val, str):
                # Try to infer more specific type for string values
                try:
                    # Check if it's a number
                    num_val = float(val)
                    if num_val.is_integer():
                        val_type = "integer_string"
                    else:
                        val_type = "float_string"
                except ValueError:
                    # Check if it's a date
                    try:
                        pd.to_datetime(val)
                        val_type = "date_string"
                    except (ValueError, TypeError):
                        val_type = "string"
                val_str = f'"{val}"'
            elif isinstance(val, (pd.Timestamp, np.datetime64)):
                val_type = "datetime"
                val_str = str(val)
            elif isinstance(val, (bool, np.bool_)):
                val_type = "boolean"
                val_str = str(val)
            else:
                val_type = type(val).__name__
                val_str = str(val)
            lines.append(f"      {col}: {val_str} ({val_type})")
    
    # Get sample rows from tail
    lines.append("  - Last 3 rows:")
    tail_sample = df.tail(3)
    for i, row in tail_sample.iterrows():
        lines.append(f"    Row {i}:")
        for col, val in row.items():
            # Infer data type for this specific value
            if pd.isna(val):
                val_type = "missing"
                val_str = "NA"
            elif isinstance(val, (int, np.integer)):
                val_type = "integer"
                val_str = str(val)
            elif isinstance(val, (float, np.floating)):
                val_type = "float"
                val_str = str(val)
            elif isinstance(val, str):
                # Try to infer more specific type for string values
                try:
                    # Check if it's a number
                    num_val = float(val)
                    if num_val.is_integer():
                        val_type = "integer_string"
                    else:
                        val_type = "float_string"
                except ValueError:
                    # Check if it's a date
                    try:
                        pd.to_datetime(val)
                        val_type = "date_string"
                    except (ValueError, TypeError):
                        val_type = "string"
                val_str = f'"{val}"'
            elif isinstance(val, (pd.Timestamp, np.datetime64)):
                val_type = "datetime"
                val_str = str(val)
            elif isinstance(val, (bool, np.bool_)):
                val_type = "boolean"
                val_str = str(val)
            else:
                val_type = type(val).__name__
                val_str = str(val)
            lines.append(f"      {col}: {val_str} ({val_type})")
    
    lines.append("")  # Add blank line for separation
    
    missing_columns = []
    
    # Detailed info for each column
    for col in df.columns:
        col_data = df[col]
        col_type = col_data.dtype
        non_null_count = col_data.count()
        missing_count = total_rows - non_null_count
        missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        unique_count = col_data.nunique(dropna=True)
        
        lines.append(f"Column: '{col}'")
        lines.append(f"  - Data type: {col_type}")
        
        # Try to infer the real data type if it's an object/string column
        inferred_type = infer_data_type(col_data)
        if inferred_type:
            lines.append(f"  - Inferred data type: {inferred_type}")
        
        lines.append(f"  - Non-null count: {non_null_count} ({(non_null_count/total_rows)*100:.2f}%)")
        lines.append(f"  - Missing count: {missing_count} ({missing_percentage:.2f}%)")
        lines.append(f"  - Unique values: {unique_count}")
        
        if missing_count > 0:
            missing_columns.append(f"{col} ({missing_percentage:.2f}%)")
        
        # For numeric columns, add detailed statistical analysis
        if pd.api.types.is_numeric_dtype(col_data):
            desc = col_data.describe()
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()
            lines.append("  - Numeric Summary Statistics:")
            lines.append(f"      Count: {desc['count']}")
            lines.append(f"      Mean: {desc['mean']:.3f}")
            lines.append(f"      Std Dev: {desc['std']:.3f}")
            lines.append(f"      Min: {desc['min']}")
            lines.append(f"      25%: {desc['25%']}")
            lines.append(f"      Median: {desc['50%']}")
            lines.append(f"      75%: {desc['75%']}")
            lines.append(f"      Max: {desc['max']}")
            lines.append(f"      Skewness: {skewness:.3f}")
            lines.append(f"      Kurtosis: {kurtosis:.3f}")
            lines.append("  - Additional notes:")
            lines.append(f"      Range: {desc['min']} to {desc['max']}")
            # Outlier hint: using IQR method
            iqr = desc['75%'] - desc['25%']
            lower_bound = desc['25%'] - 1.5 * iqr
            upper_bound = desc['75%'] + 1.5 * iqr
            potential_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            lines.append(f"      Potential outliers count (IQR method): {potential_outliers.count()}")
        else:
            # For non-numeric columns, show a preview of unique values and mode frequencies.
            sample_values = col_data.dropna().unique()[:5]
            sample_values_str = ", ".join([str(val) for val in sample_values])
            lines.append(f"  - Sample unique values (up to 5): {sample_values_str}")
            
            # Display top frequent values if available
            top_freq = col_data.value_counts().head(3)
            if not top_freq.empty:
                lines.append("  - Top frequent values:")
                for value, count in top_freq.items():
                    lines.append(f"      {value}: {count} times")
        
        lines.append("")  # Blank line for separation between columns
    
    # Summary of columns with missing values
    if missing_columns:
        lines.append("Columns with missing values:")
        lines.append("  " + ", ".join(missing_columns))
        lines.append("")
    
    # If more than one numeric column exists, compute a correlation matrix summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        lines.append("Correlation Matrix (Numeric Columns):")
        corr_matrix = df[numeric_cols].corr()
        # Instead of full matrix, provide summary stats on correlations for brevity.
        for col in corr_matrix.columns:
            # Exclude self-correlation by dropping the value 1.0
            other_corrs = corr_matrix[col][corr_matrix[col] != 1.0]
            lines.append(f"  - {col}: Mean correlation = {other_corrs.mean():.3f}, "
                         f"Min = {other_corrs.min():.3f}, Max = {other_corrs.max():.3f}")
    
    return "\n".join(lines)

#  def _evaluate_file_relevance(self, file_info: Dict[str, Any], user_query: str) -> Tuple[Dict[str, Any], float, bool]:
#         """
#         Evaluate a single file's relevance to the user query.
        
#         Args:
#             file_info: Dictionary containing file metadata
#             user_query: The user's query
            
#         Returns:
#             Tuple of (file_info, confidence_score, is_certain)
#         """
#         try:
#             # Define Pydantic model for structured output
#             from pydantic import BaseModel, Field
            
#             class FileRelevance(BaseModel):
#                 score: float = Field(..., description="A number from 0 to 10 indicating relevance (10 being perfect match)")
#                 certain: bool = Field(..., description="Whether you're certain this is the correct file")
#                 reason: str = Field(..., description="A brief explanation of your reasoning")
            
#             # Create parser for the model
#             from langchain.output_parsers import PydanticOutputParser
#             parser = PydanticOutputParser(pydantic_object=FileRelevance)
            
#             # Create messages for the model to determine relevance
#             messages = [
#                 SystemMessage(content=f"""
#                 You are a file selection assistant. Based on the user's query and a file's metadata,
#                 determine how relevant this file is to the query.
                
#                 {parser.get_format_instructions()}
                
#                 Only mark "certain" as true if you are absolutely confident this is the correct file.
#                 """),
#                 HumanMessage(content=f"""
#                 User query: {user_query}
                
#                 File metadata:
#                 Filename: {file_info.get("filename", "")}
#                 Summary: {file_info.get("summary", "")}
                
#                 How relevant is this file to the query?
#                 """)
#             ]
            
#             # Get response from the model
#             response = self.model_mini.invoke(messages)
#             response_text = response.content.strip()
            
#             # Parse the response using the Pydantic parser
#             result = parser.parse(response_text)
            
#             return (file_info, result.score, result.certain)
        
#         except Exception as e:
#             print(f"Error evaluating file relevance: {str(e)}")
#             return (file_info, 0, False)

def json_to_text(data, indent=0, max_items=10):
    """
    Convert JSON data to formatted text with improved readability.
    
    Args:
        data: The JSON data to convert
        indent: The current indentation level
        max_items: Maximum number of items to show in arrays/dictionaries
        
    Returns:
        Formatted text representation of the JSON data
    """
    output = ""
    prefix = "  " * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            # Skip large value_counts dictionaries
            if key == "value_counts" and isinstance(value, dict) and len(value) > max_items:
                output += f"{prefix}{key}: [too many values to display - {len(value)} items]\n"
                continue
                
            # Special handling for OID and DATEUPD fields
            if key in ["OID", "DATEUPD"] and isinstance(value, dict) and "value_counts" in value:
                output += f"{prefix}{key}:\n"
                output += f"{prefix}  type: {value.get('type', 'Unknown')}\n"
                output += f"{prefix}  n_distinct: {value.get('n_distinct', 'Unknown')}\n"
                output += f"{prefix}  is_unique: {value.get('is_unique', 'Unknown')}\n"
                output += f"{prefix}  n_missing: {value.get('n_missing', 'Unknown')}\n"
                output += f"{prefix}  count: {value.get('count', 'Unknown')}\n"
                output += f"{prefix}  value_counts: [too many values to display - {len(value.get('value_counts', {}))} items]\n"
                continue
                
            if isinstance(value, (dict, list)):
                output += f"{prefix}{key}:\n" + json_to_text(value, indent + 1, max_items)
            else:
                output += f"{prefix}{key}: {value}\n"
    elif isinstance(data, list):
        if len(data) > max_items:
            # For large lists, show only the first few items
            for i, item in enumerate(data[:max_items]):
                if isinstance(item, (dict, list)):
                    output += f"{prefix}- \n" + json_to_text(item, indent + 1, max_items)
                else:
                    output += f"{prefix}- {item}\n"
            output += f"{prefix}... [{len(data) - max_items} more items]\n"
        else:
            for item in data:
                if isinstance(item, (dict, list)):
                    output += f"{prefix}- \n" + json_to_text(item, indent + 1, max_items)
                else:
                    output += f"{prefix}- {item}\n"
    else:
        output += f"{prefix}{data}\n"
    return output

