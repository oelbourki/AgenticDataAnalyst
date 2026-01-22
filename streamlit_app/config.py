"""
Configuration and constants for the Streamlit app.
"""

import os
from pathlib import Path

# Example prompts for each dataset
EXAMPLE_PROMPTS = {
    "titanic.csv": [
        "Show me the survival rate by passenger class",
        "Create a chart comparing survival rates between males and females",
        "Visualize the age distribution of passengers",
        "Show survival rate by age groups",
        "Compare survival rates across different embarkation ports"
    ],
    "iris.csv": [
        "Create a scatter plot of sepal length vs sepal width colored by species",
        "Show the distribution of petal lengths for each species",
        "Visualize the relationship between all flower measurements",
        "Compare average measurements across different species",
        "Create a box plot showing petal width by species"
    ],
    "sales_data.csv": [
        "Show me sales trends over time",
        "Create a chart comparing sales by region",
        "Visualize sales by product category",
        "Show monthly sales trends",
        "Compare average sales amount across regions"
    ],
    "customer_data.csv": [
        "Show customer age distribution",
        "Create a chart comparing purchases by location",
        "Visualize average order value by gender",
        "Show customer distribution by location",
        "Compare total purchases across different age groups"
    ]
}

# Paths
DATASETS_DIR = Path("datasets")
METADATA_PATH = DATASETS_DIR / "metadata.json"
TEMP_DIRS = [
    "/tmp/temp_code_files",
    "/tmp/downloads",
]

# Backend configuration
USE_DOCKER = os.environ.get("USE_DOCKER_BACKEND", "").lower() == "true"

# API Configuration
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Page configuration
PAGE_CONFIG = {
    "page_title": "AgenticDataAnalyst",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
