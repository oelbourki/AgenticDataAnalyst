"""
Metadata generation service.
"""

import json
import streamlit as st
import pandas as pd
from typing import Dict

from ..services.api_key_service import APIKeyService
from ..utils.error_handler import ErrorHandler
from ..config import GEMINI_MODEL


class MetadataService:
    """Service for generating dataset metadata."""
    
    def __init__(self):
        self.api_key_service = APIKeyService()
        self._workflow_available = None
    
    def _check_workflow(self) -> bool:
        """Check if workflow utilities are available."""
        if self._workflow_available is None:
            try:
                from agent_coder.utils import generate_detailed_dataframe_description
                self._workflow_available = True
            except ImportError:
                self._workflow_available = False
        return self._workflow_available
    
    def _get_detailed_description(self, df: pd.DataFrame, filename: str) -> str:
        """Get detailed description of the dataset."""
        if self._check_workflow():
            try:
                from agent_coder.utils import generate_detailed_dataframe_description
                return generate_detailed_dataframe_description(df)
            except Exception:
                pass
        
        # Fallback basic description
        return f"""
Dataset: {filename}
Rows: {len(df)}
Columns: {len(df.columns)}
Column names: {', '.join(df.columns[:10])}
Data types: {', '.join([f"{col}: {str(dtype)}" for col, dtype in df.dtypes.items()][:5])}
"""
    
    def _extract_json_from_response(self, content: str) -> dict:
        """Extract JSON from LLM response."""
        # Extract JSON from response
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract just the JSON object
            # Look for first { and last }
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(content[start_idx:end_idx+1])
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse JSON from response: {str(e)}")
            raise ValueError(f"Could not parse JSON from response: {str(e)}")
    
    def _get_fallback_metadata(self, df: pd.DataFrame, filename: str) -> Dict[str, str]:
        """Get fallback metadata when AI generation fails."""
        return {
            "description": f"Dataset with {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}",
            "summary": f"{filename} dataset with {len(df)} rows"
        }
    
    def generate_metadata(self, df: pd.DataFrame, filename: str) -> Dict[str, str]:
        """Generate AI-powered description and summary for a dataset."""
        try:
            # Try to import Gemini
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            api_key = self.api_key_service.get_api_key()
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            
            model = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                temperature=0.0,
                google_api_key=api_key
            )
            
            # Generate detailed analysis
            detailed_description = self._get_detailed_description(df, filename)
            
            system_prompt = """You are a data analyst expert. Your task is to create a comprehensive description and a brief summary for a dataset.

Given the detailed dataset analysis, create:
1. A detailed description (2-3 sentences) that includes:
   - What the data represents
   - Key columns and their data types
   - Important characteristics or patterns
   - What kind of analysis this dataset is suitable for

2. A brief summary (one short sentence) that captures the essence of the dataset.

Format your response as JSON:
{
  "description": "detailed description here",
  "summary": "brief summary here"
}

Only return valid JSON, no additional text."""
            
            user_prompt = f"""Dataset filename: {filename}

Detailed Dataset Analysis:
{detailed_description}

Please generate a description and summary for this dataset."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            try:
                response = model.invoke(messages)
                content = response.content.strip()
                
                metadata = self._extract_json_from_response(content)
                return metadata
            except Exception as invoke_error:
                # Handle errors during API call
                if ErrorHandler.is_rate_limit_error(invoke_error):
                    st.warning("⚠️ API rate limit reached. Using basic dataset description. Try again later for AI-generated metadata.")
                else:
                    st.warning(f"Could not generate AI metadata: {str(invoke_error)[:100]}. Using basic description.")
                return self._get_fallback_metadata(df, filename)
        
        except Exception as e:
            # Handle errors during setup
            if ErrorHandler.is_rate_limit_error(e):
                st.warning("⚠️ API rate limit reached. Using basic dataset description. Try again later for AI-generated metadata.")
            else:
                st.warning(f"Could not generate AI metadata: {str(e)[:100]}. Using basic description.")
            return self._get_fallback_metadata(df, filename)
