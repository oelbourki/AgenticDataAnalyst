"""
Example prompts tab component.
"""

import streamlit as st

from ..config import EXAMPLE_PROMPTS
from ..state import AppState


def render_examples_tab(selected_dataset: str) -> None:
    """Render the example prompts tab."""
    st.header("Example Prompts")
    st.markdown("Click any example to use it in the Query Interface")
    
    if selected_dataset and selected_dataset in EXAMPLE_PROMPTS:
        for i, prompt in enumerate(EXAMPLE_PROMPTS[selected_dataset]):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{i+1}.** {prompt}")
            with col2:
                if st.button("Use", key=f"use_{i}"):
                    AppState.set_user_query(prompt)
                    AppState.set_query_input(prompt)
                    st.rerun()
    else:
        st.info("No example prompts available for this dataset. Try asking your own question!")
