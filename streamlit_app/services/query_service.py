"""
Query processing service.
"""

import os
import streamlit as st
from typing import Dict, Any, Optional, List

from ..utils.error_handler import ErrorHandler


class QueryService:
    """Service for processing queries and displaying results."""
    
    def __init__(self):
        self._simple_coder = None
    
    def _get_simple_coder(self) -> Any:
        """Lazy load simple_coder."""
        if self._simple_coder is None:
            from agent_coder.simple_workflow import simple_coder
            self._simple_coder = simple_coder
        return self._simple_coder
    
    def process_query(self, query: str, selected_dataset: str) -> Dict[str, Any]:
        """Process a user query and return results."""
        simple_coder = self._get_simple_coder()
        from agent_coder.simple_workflow import process_query as workflow_process_query
        
        return workflow_process_query(simple_coder, query, selected_dataset=selected_dataset)
    
    def find_images(self) -> List[str]:
        """Find generated images from execution."""
        display_images = []
        
        # Try ImageProcessor as fallback
        try:
            from codibox import ImageProcessor
            processor = ImageProcessor(image_base_dir="/tmp/temp_code_files")
            found_images = processor.find_images()
            
            if found_images:
                display_images = found_images
        except Exception as e:
            st.debug(f"Image processor error: {e}")
        
        return display_images
    
    def display_results(self, result: Dict[str, Any]) -> None:
        """Display query results."""
        if not result or "messages" not in result:
            st.error("No results returned. Check the console for errors.")
            return
        
        last_message = result["messages"][-1]
        
        # Show execution status
        st.success("✅ Analysis completed successfully!")
        
        # Display markdown content
        if hasattr(last_message, 'response_metadata'):
            metadata_result = last_message.response_metadata
            
            # Show the final markdown
            st.subheader("📊 Analysis Results")
            
            # Try to display images
            display_images = []
            
            # First, check metadata for images
            if "images" in metadata_result and metadata_result["images"]:
                display_images = metadata_result["images"]
            
            # Also try ImageProcessor as fallback
            if not display_images:
                display_images = self.find_images()
            
            # Display all found images in a collapsible expander (collapsed by default)
            if display_images:
                with st.expander(f"🖼️ View Generated Visualizations ({len(display_images)})", expanded=False):
                    for img_path in display_images:
                        try:
                            if os.path.exists(img_path):
                                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                            else:
                                st.warning(f"Image file not found: {img_path}")
                        except Exception as e:
                            st.warning(f"Could not display {img_path}: {e}")
            else:
                st.info("💡 Tip: Make sure code saves images to 'temp_code_files/' directory using plt.savefig()")
            
            if "markdown_content" in metadata_result:
                # Process markdown to ensure base64 images render properly in HTML
                from ..utils.markdown_processor import process_markdown_for_html
                markdown_content = metadata_result["markdown_content"]
                markdown_content = process_markdown_for_html(markdown_content)
                # Display markdown with HTML rendering (images will be inline)
                st.markdown(markdown_content, unsafe_allow_html=True)
            
            # Show raw markdown if available
            if "markdown" in metadata_result:
                with st.expander("📄 View Raw Execution Output"):
                    st.markdown(metadata_result["markdown"], unsafe_allow_html=True)
            
            # Show CSV files if generated
            if "csv_file_list" in metadata_result and metadata_result["csv_file_list"]:
                st.subheader("📁 Generated Files")
                for csv_file in metadata_result["csv_file_list"]:
                    st.write(f"  - {csv_file}")
        else:
            st.write("Response:", last_message.content)
    
    def handle_query_error(self, error: Exception) -> None:
        """Handle errors during query processing."""
        if isinstance(error, ValueError):
            ErrorHandler.handle_value_error(error)
        elif ErrorHandler.is_rate_limit_error(error):
            ErrorHandler.handle_rate_limit_error(error)
        else:
            ErrorHandler.handle_generic_error(error, "while processing your query")
