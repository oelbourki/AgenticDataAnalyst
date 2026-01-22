"""
Dataset preview tab component.
"""

import streamlit as st
import pandas as pd

from ..services.dataset_service import DatasetService


def render_preview_tab(selected_dataset: str, dataset_service: DatasetService) -> None:
    """Render the dataset preview tab."""
    st.header("Dataset Preview")
    
    # Get dataset info
    datasets = dataset_service.get_datasets_info()
    selected_info = next((d for d in datasets if d["name"] == selected_dataset), None)
    
    if not selected_info:
        st.error(f"Dataset {selected_dataset} not found!")
        return
    
    # Load and display dataset
    try:
        df = dataset_service.load_dataset(selected_info["file_path"])
        
        st.subheader(f"Preview: {selected_dataset}")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(dtype) for dtype in df.dtypes],
            "Non-Null Count": df.count().values,
            "Null Count": df.isnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.subheader("Numeric Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
