"""
Dataset management service.
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config import DATASETS_DIR, METADATA_PATH


class DatasetService:
    """Service for managing datasets."""
    
    @st.cache_data
    def get_datasets_info(_self) -> List[Dict[str, Any]]:
        """Get information about available datasets (cached)."""
        datasets = []
        
        if DATASETS_DIR.exists():
            for csv_file in DATASETS_DIR.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    datasets.append({
                        "name": csv_file.name,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": list(df.columns),
                        "file_path": str(csv_file)
                    })
                except Exception as e:
                    st.warning(f"Could not load {csv_file.name}: {e}")
        
        return datasets
    
    @st.cache_data
    def get_metadata(_self) -> List[Dict[str, Any]]:
        """Load dataset metadata (cached)."""
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)
        return []
    
    def get_dataset_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific dataset."""
        metadata = self.get_metadata()
        return next((m for m in metadata if m.get("filename") == filename), None)
    
    def update_metadata(self, filename: str, file_path: str, description: str, summary: str) -> List[Dict[str, Any]]:
        """Add or update entry in metadata.json."""
        metadata = self.get_metadata()
        
        # Remove existing entry if present
        metadata = [m for m in metadata if m.get("filename") != filename]
        
        # Add new entry
        metadata.append({
            "filename": filename,
            "file_path": file_path,
            "description": description,
            "summary": summary
        })
        
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clear cache to force reload
        self.get_metadata.clear()
        
        return metadata
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load a dataset from file path."""
        return pd.read_csv(file_path)
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        try:
            self.get_datasets_info.clear()
            self.get_metadata.clear()
        except Exception:
            # Cache might not exist yet
            pass
