"""
Error handling utilities.
"""

import re
import streamlit as st


class ErrorHandler:
    """Centralized error handling for the application."""
    
    @staticmethod
    def is_rate_limit_error(error: Exception) -> bool:
        """Check if error is a rate limit/quota error."""
        error_str = str(error)
        return (
            "RESOURCE_EXHAUSTED" in error_str or
            "429" in error_str or
            "quota" in error_str.lower() or
            "Rate Limit" in error_str
        )
    
    @staticmethod
    def handle_rate_limit_error(error: Exception) -> None:
        """Display user-friendly rate limit error message."""
        st.error("🚫 **API Rate Limit Exceeded**")
        error_str = str(error)
        
        # Try to extract retry delay
        retry_delay = "50-60"
        if "retry in" in error_str.lower():
            delay_match = re.search(r"retry in ([\d.]+)s", error_str, re.IGNORECASE)
            if delay_match:
                retry_delay = str(int(float(delay_match.group(1))))
        
        st.markdown(f"""
        ⚠️ **Your Gemini API quota has been exceeded.**
        
        The free tier allows **20 requests per day per model**.
        
        **Solutions:**
        1. Wait {retry_delay} seconds before trying again
        2. Check your quota: https://ai.dev/rate-limit
        3. Upgrade your API plan for higher limits
        4. Try again later (quota resets daily)
        """)
        
        with st.expander("🔍 Technical Details"):
            st.code(error_str[:500])
    
    @staticmethod
    def handle_api_error(error: Exception) -> None:
        """Display user-friendly API error message."""
        error_msg = str(error)
        st.error("⚠️ **API Error**")
        st.markdown(error_msg)
    
    @staticmethod
    def handle_generic_error(error: Exception, context: str = "") -> None:
        """Display generic error message."""
        st.error(f"❌ **Error occurred{': ' + context if context else ''}**")
        st.markdown(f"An unexpected error occurred while processing your request.")
        with st.expander("🔍 Error Details"):
            st.exception(error)
    
    @staticmethod
    def handle_value_error(error: ValueError) -> None:
        """Handle ValueError (often used for API errors)."""
        error_msg = str(error)
        if ErrorHandler.is_rate_limit_error(error):
            ErrorHandler.handle_rate_limit_error(error)
        else:
            ErrorHandler.handle_api_error(error)
