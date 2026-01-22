"""
Session state management.
"""

import streamlit as st
from typing import Optional


class AppState:
    """Centralized session state management."""
    
    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables."""
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        
        if "query_input" not in st.session_state:
            st.session_state.query_input = ""
    
    @staticmethod
    def get_user_query() -> str:
        """Get user query from session state."""
        return st.session_state.get("user_query", "")
    
    @staticmethod
    def set_user_query(query: str) -> None:
        """Set user query in session state."""
        st.session_state.user_query = query
    
    @staticmethod
    def get_query_input() -> str:
        """Get query input from session state."""
        return st.session_state.get("query_input", "")
    
    @staticmethod
    def set_query_input(query: str) -> None:
        """Set query input in session state."""
        st.session_state.query_input = query
    
    @staticmethod
    def sync_query_state() -> None:
        """Sync query_input with user_query."""
        query_input = st.session_state.get("query_input", "")
        if query_input:
            st.session_state.user_query = query_input
