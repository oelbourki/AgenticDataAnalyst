"""
Query interface component.
"""

import streamlit as st

from ..services.api_key_service import APIKeyService
from ..services.backend_service import BackendService
from ..services.query_service import QueryService
from ..state import AppState
from ..config import EXAMPLE_PROMPTS
from ..utils.file_utils import cleanup_old_images


def render_query_interface(
    selected_dataset: str,
    backend_service: BackendService,
    api_key_service: APIKeyService,
    query_service: QueryService
) -> None:
    """Render the query interface tab."""
    st.header("Ask a Question")
    st.markdown("Enter a question about the selected dataset, and the AI will generate charts and insights automatically!")
    
    # Check if code execution is available
    execution_ready = backend_service.is_execution_ready
    backend_status = backend_service.check_backend_status()
    backend_type = backend_status.get("backend", "unknown")
    
    if not execution_ready:
        if not backend_service.codibox_available:
            st.warning("""
            ⚠️ **Code execution is not available**
            
            Codibox package is not installed. Install with:
            ```bash
            pip install codibox
            ```
            """)
        elif not backend_status.get("ready", False):
            st.warning("""
            ⚠️ **Code execution backend is not ready**
            
            The execution backend is initializing. This may take a moment.
            You can still:
            - Upload and manage datasets
            - Generate metadata
            - Preview datasets
            """)
        else:
            st.warning("⚠️ Workflow not available. Please check the console for errors.")
    
    # Check API key
    api_key = api_key_service.get_api_key()
    if not api_key:
        st.warning("⚠️ Please set your Google Gemini API key in the sidebar to use the query interface.")
    
    # Example prompts for selected dataset
    if selected_dataset and selected_dataset in EXAMPLE_PROMPTS:
        st.subheader("💡 Quick Examples")
        example_cols = st.columns(min(3, len(EXAMPLE_PROMPTS[selected_dataset])))
        for i, example in enumerate(EXAMPLE_PROMPTS[selected_dataset][:3]):
            with example_cols[i]:
                if st.button(f"Use: {example[:40]}...", key=f"example_{i}", use_container_width=True):
                    AppState.set_user_query(example)
                    AppState.set_query_input(example)
                    st.rerun()
    
    # Query input
    AppState.initialize()
    user_query = st.text_area(
        "Your Question",
        height=100,
        placeholder=f"e.g., 'Show me sales trends by region' for {selected_dataset}",
        help="Ask a question about the dataset. The AI will automatically select the right file and generate charts.",
        key="query_input",
        disabled=not (execution_ready and backend_service.workflow_available and api_key)
    )
    
    # Sync user_query with query_input when user types
    if user_query:
        AppState.set_user_query(user_query)
    
    # Generate button
    if st.button("🚀 Generate Analysis", type="primary", use_container_width=True, disabled=not (execution_ready and backend_service.workflow_available and api_key)):
        # Get the current query from the text area
        current_query = user_query.strip() if user_query else ""
        
        if not current_query:
            st.warning("Please enter a question!")
        elif not api_key:
            st.error("Please set your Google Gemini API key in the sidebar first!")
        elif not execution_ready:
            st.error(f"Execution backend ({backend_type}) is not ready. Please wait or check status.")
        elif not backend_service.workflow_available:
            st.error("Workflow not available. Please check the console for errors.")
        else:
            # Update session state with the current query
            AppState.set_user_query(current_query)
            
            # Clean up old images and charts before new execution
            cleanup_old_images()
            
            with st.spinner("🤖 AI is analyzing your data... This may take a minute..."):
                try:
                    # Process the query
                    result = query_service.process_query(current_query, selected_dataset)
                    
                    # Display results
                    query_service.display_results(result)
                
                except Exception as e:
                    query_service.handle_query_error(e)
