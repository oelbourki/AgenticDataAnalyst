"""
API Key management service.
"""

import os
import streamlit as st
from ..config import API_KEY_ENV_VAR


class APIKeyService:
    """Service for managing API keys."""
    
    @staticmethod
    def get_api_key() -> str:
        """Get API key from Streamlit secrets or environment variables."""
        # Try Streamlit secrets first (for Streamlit Cloud)
        # Note: Accessing st.secrets may raise an exception if secrets.toml doesn't exist
        # This is expected behavior - we catch it and fall back to environment variables
        try:
            if hasattr(st, 'secrets'):
                # Try to access secrets - Streamlit raises exception if secrets.toml doesn't exist
                # This is normal for local development, so we catch all exceptions
                try:
                    # Access st.secrets - this may raise if secrets.toml is missing
                    secrets_obj = st.secrets
                    if secrets_obj and hasattr(secrets_obj, 'get'):
                        api_key = secrets_obj.get(API_KEY_ENV_VAR, None)
                        if api_key:
                            return api_key
                    elif secrets_obj and API_KEY_ENV_VAR in secrets_obj:
                        return secrets_obj[API_KEY_ENV_VAR]
                except Exception:
                    # Any error accessing secrets (e.g., "No secrets found", FileNotFoundError, etc.)
                    # This is expected when secrets.toml doesn't exist - fall back to env
                    pass
        except Exception:
            # Any error with st.secrets attribute - fall back to env
            pass
        
        # Fallback to environment variable
        return os.getenv(API_KEY_ENV_VAR, "")
    
    @staticmethod
    def set_api_key(key: str) -> None:
        """Set API key in environment (for current session)."""
        os.environ[API_KEY_ENV_VAR] = key
    
    @staticmethod
    def is_configured() -> bool:
        """Check if API key is configured."""
        return bool(APIKeyService.get_api_key())
    
    @staticmethod
    def is_from_secrets() -> bool:
        """Check if API key is from Streamlit secrets."""
        try:
            if hasattr(st, 'secrets'):
                try:
                    secrets_obj = st.secrets
                    if secrets_obj:
                        if hasattr(secrets_obj, 'get'):
                            return secrets_obj.get(API_KEY_ENV_VAR, None) is not None
                        elif API_KEY_ENV_VAR in secrets_obj:
                            return True
                except Exception:
                    pass
        except Exception:
            pass
        return False
