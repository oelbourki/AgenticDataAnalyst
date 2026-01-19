# Datasets Folder

This folder contains CSV data files and a metadata file that describes each dataset.

## Structure

```
datasets/
├── metadata.json          # JSON file describing all CSV files
├── sales_data.csv         # Example dataset
├── customer_data.csv      # Example dataset
└── ...
```

## Metadata File Format

The `metadata.json` file should contain an array of objects, each describing a CSV file:

```json
[
  {
    "filename": "sales_data.csv",
    "file_path": "datasets/sales_data.csv",
    "description": "Detailed description of what the CSV contains, including column names and data types",
    "summary": "Brief summary of the dataset"
  }
]
```

### Required Fields

- **filename**: The name of the CSV file (e.g., "sales_data.csv")
- **file_path**: Full path to the file (can be relative to project root or absolute)
- **description**: Detailed description that will be used by the LLM to match user queries to the right dataset
- **summary**: Brief summary (optional, can be same as description)

### Description Best Practices

The description should include:
- What the data represents
- Key columns and their types
- Time period covered (if applicable)
- Any important characteristics

Example:
```json
{
  "filename": "sales_data.csv",
  "file_path": "datasets/sales_data.csv",
  "description": "Monthly sales data from 2020-2024 with product categories (Electronics, Clothing, Food), regions (North, South, East, West), revenue amounts, units sold, and customer IDs. Contains date column for time series analysis.",
  "summary": "Sales data with time series and regional breakdown"
}
```

## How It Works

1. User asks a question (e.g., "Show me sales trends by region")
2. `FileAccessAgent` reads `metadata.json`
3. LLM analyzes each description and scores relevance to the query
4. Best matching CSV file is selected
5. Code is generated to create charts from that CSV
6. Charts are executed and embedded in final markdown

## Creating Your Metadata

1. Copy `metadata.json.example` to `metadata.json`
2. Add entries for each CSV file in your datasets folder
3. Write detailed descriptions - the better the description, the better the file matching!

## Example Metadata Entry

```json
{
  "filename": "customer_data.csv",
  "file_path": "datasets/customer_data.csv",
  "description": "Customer demographics and purchase history dataset. Contains: customer_id (string), age (integer), gender (string: Male/Female/Other), location (string: city names), total_purchases (integer), average_order_value (float), last_purchase_date (date). Suitable for customer segmentation, purchase behavior analysis, and demographic insights.",
  "summary": "Customer analytics for segmentation and behavior analysis"
}
```
