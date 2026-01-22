"""
Main entry point for the Streamlit app.
"""

import streamlit as st

from .config import PAGE_CONFIG
from .state import AppState
from .services.api_key_service import APIKeyService
from .services.backend_service import BackendService
from .services.dataset_service import DatasetService
from .services.metadata_service import MetadataService
from .services.query_service import QueryService
from .components.sidebar import render_sidebar
from .components.status_display import render_backend_status
from .components.query_interface import render_query_interface
from .components.examples_tab import render_examples_tab
from .components.preview_tab import render_preview_tab


def main():
    """Main application entry point."""
    # Page config - must be first Streamlit command
    st.set_page_config(**PAGE_CONFIG)
    
    # Initialize session state
    AppState.initialize()
    
    # Initialize services
    api_key_service = APIKeyService()
    backend_service = BackendService()
    dataset_service = DatasetService()
    metadata_service = MetadataService()
    query_service = QueryService()
    
    # Title
    st.title("🤖 AgenticDataAnalyst")
    st.markdown("**AI-Powered Data Analysis Platform** - Upload datasets, ask questions, get insights!")
    
    # Show backend status
    render_backend_status(backend_service)
    
    # Sidebar
    with st.sidebar:
        selected_dataset = render_sidebar(
            api_key_service,
            dataset_service,
            metadata_service
        )
    
    # Main content area
    datasets = dataset_service.get_datasets_info()
    if not datasets:
        st.info("👆 Upload a dataset in the sidebar to get started!")
        return
    
    if not selected_dataset:
        st.warning("Please select a dataset from the sidebar!")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Query Interface", "📝 Example Prompts", "📊 Dataset Preview"])
    
    with tab1:
        render_query_interface(
            selected_dataset,
            backend_service,
            api_key_service,
            query_service
        )
    
    with tab2:
        render_examples_tab(selected_dataset)
    
    with tab3:
        render_preview_tab(selected_dataset, dataset_service)


if __name__ == "__main__":
    main()
