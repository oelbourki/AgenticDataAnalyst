"""
Status display component.
"""

import streamlit as st

from ..services.backend_service import BackendService


def render_backend_status(backend_service: BackendService) -> None:
    """Render backend status indicator."""
    backend_status = backend_service.check_backend_status()
    execution_ready = backend_status.get("ready", False)
    backend_type = backend_status.get("backend", "unknown")
    
    # Show backend status
    if backend_status["available"]:
        if backend_type == "host":
            st.success("✅ Host backend ready - Code execution available")
            st.caption("Using fast host execution mode (default) - works on Streamlit Cloud")
        elif backend_type == "docker":
            if execution_ready:
                st.success("✅ Docker backend ready - Code execution available")
                st.caption("Using secure Docker execution mode")
            else:
                with st.expander("⚠️ Docker Container Status", expanded=False):
                    st.info("""
                    **Docker container is not running.**
                    
                    The app will automatically set up the container when needed.
                    You can still upload datasets and generate metadata.
                    """)
    else:
        with st.expander("⚠️ Execution Backend", expanded=False):
            st.warning("""
            **Code execution not available.**
            
            Codibox package is not installed. You can still:
            - Upload and manage datasets
            - Generate AI-powered metadata
            - Preview datasets
            """)
