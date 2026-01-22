"""
File utility functions.
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Any

from ..config import TEMP_DIRS


def cleanup_old_images() -> None:
    """Delete old images and charts from temp directories before new execution."""
    for temp_dir in TEMP_DIRS:
        if os.path.exists(temp_dir):
            try:
                # Delete all files in the directory
                for file_path in glob.glob(os.path.join(temp_dir, "*")):
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        # Continue even if some files can't be deleted
                        print(f"Warning: Could not delete {file_path}: {e}")
                
                # Also check for nested temp_code_files directory
                nested_dir = os.path.join(temp_dir, "temp_code_files")
                if os.path.exists(nested_dir):
                    for file_path in glob.glob(os.path.join(nested_dir, "*")):
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(f"Warning: Could not delete {file_path}: {e}")
            except Exception as e:
                # Don't fail if cleanup fails
                print(f"Warning: Could not cleanup {temp_dir}: {e}")


def save_uploaded_file(uploaded_file: Any, datasets_dir: Path) -> Path:
    """
    Save uploaded CSV file to datasets directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        datasets_dir: Path to datasets directory
        
    Returns:
        Path to saved file
    """
    """Save uploaded CSV file to datasets directory."""
    file_path = datasets_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def ensure_datasets_dir() -> Path:
    """Ensure datasets directory exists and return Path."""
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    return datasets_dir
