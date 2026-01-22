"""
Markdown processing utilities.
"""

import re


def process_markdown_for_html(markdown_content: str) -> str:
    """Process markdown content to ensure base64 images render properly in HTML."""
    # Pattern to match base64 images in HTML img tags
    # Handles: <img src='data:image/png;base64,...'> or <img src="data:image/png;base64,...">
    # Also handles img tags with various attribute orders
    base64_pattern = r"<img\s+([^>]*?)src=['\"]data:image/([^;]+);base64,([^'\">\s]+)['\"]([^>]*?)>"
    
    def enhance_image_tag(match):
        """Enhance image tag for better HTML rendering in Streamlit."""
        attrs_before = match.group(1) or ""
        image_format = match.group(2).lower()
        base64_data = match.group(3)
        attrs_after = match.group(4) or ""
        
        # Check if style attribute already exists
        has_style = 'style=' in attrs_before or 'style=' in attrs_after
        
        # Check if class attribute already exists
        has_class = 'class=' in attrs_before or 'class=' in attrs_after
        
        # Build enhanced attributes
        enhanced_attrs = []
        
        # Add style for proper rendering in Streamlit
        if not has_style:
            enhanced_attrs.append('style="max-width: 100%; height: auto; display: block; margin: 1em auto; border-radius: 4px;"')
        
        # Add class for responsive images
        if not has_class:
            enhanced_attrs.append('class="img-fluid"')
        
        # Combine all attributes
        all_attrs = " ".join([attrs_before.strip(), *enhanced_attrs, attrs_after.strip()]).strip()
        
        # Reconstruct the img tag with proper formatting
        return f'<img {all_attrs} src="data:image/{image_format};base64,{base64_data}">'
    
    # Replace all base64 image tags with enhanced versions
    processed_content = re.sub(base64_pattern, enhance_image_tag, markdown_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Also wrap the content in a div with proper styling for better HTML rendering
    # This ensures the HTML renders properly in Streamlit
    if processed_content.strip():
        # Don't double-wrap if already wrapped
        if not processed_content.strip().startswith('<div'):
            processed_content = f'<div style="line-height: 1.6;">{processed_content}</div>'
    
    return processed_content
