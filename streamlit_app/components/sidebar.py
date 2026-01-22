"""
Sidebar component for configuration and dataset management.
"""

import streamlit as st

from ..services.api_key_service import APIKeyService
from ..services.dataset_service import DatasetService
from ..services.metadata_service import MetadataService
from ..utils.file_utils import save_uploaded_file, ensure_datasets_dir


def render_api_key_section(api_key_service: APIKeyService) -> None:
    """Render API key management section."""
    st.subheader("🔑 API Key")
    current_key = api_key_service.get_api_key()
    
    # Check if key is set via secrets
    key_from_secrets = api_key_service.is_from_secrets()
    
    if key_from_secrets:
        st.success("✅ API key configured via Streamlit secrets")
        st.caption("To change, update secrets in Streamlit Cloud settings")
    else:
        # Show manual input if not using secrets
        api_key_input = st.text_input(
            "Google Gemini API Key",
            value=current_key if current_key else "",
            type="password",
            help="Enter your Google Gemini API key. Get one from https://aistudio.google.com/app/apikey",
            key="api_key_input"
        )
        
        if st.button("💾 Save API Key", use_container_width=True):
            if api_key_input:
                api_key_service.set_api_key(api_key_input)
                st.success("✅ API key saved for this session!")
                st.rerun()
            else:
                st.error("Please enter an API key")
        
        if current_key:
            st.success("✅ API key configured")
        else:
            st.warning("⚠️ API key not set")
            st.info("💡 Tip: Set `GOOGLE_API_KEY` in Streamlit secrets for persistent storage")


def render_dataset_upload_section(
    dataset_service: DatasetService,
    metadata_service: MetadataService,
    api_key_service: APIKeyService
) -> None:
    """Render dataset upload section."""
    st.header("📤 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload a CSV file to analyze. The AI will automatically generate a description."
    )
    
    if uploaded_file is not None:
        datasets_dir = ensure_datasets_dir()
        
        # Check if file already exists
        file_path = datasets_dir / uploaded_file.name
        if file_path.exists():
            st.warning(f"⚠️ File {uploaded_file.name} already exists. Uploading will overwrite it.")
        
        if st.button("📥 Upload & Process", use_container_width=True, type="primary"):
            with st.spinner("Processing dataset..."):
                try:
                    # Save file
                    save_uploaded_file(uploaded_file, datasets_dir)
                    st.success(f"✅ File saved: {uploaded_file.name}")
                    
                    # Load and analyze
                    df = dataset_service.load_dataset(str(file_path))
                    st.info(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
                    
                    # Check API key for metadata generation
                    if not api_key_service.is_configured():
                        st.warning("⚠️ API key not set. Generating basic metadata...")
                        metadata = {
                            "description": f"Dataset with {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns[:10])}",
                            "summary": f"{uploaded_file.name} dataset with {len(df)} rows"
                        }
                    else:
                        # Generate metadata
                        with st.spinner("🤖 AI is generating description..."):
                            metadata = metadata_service.generate_metadata(df, uploaded_file.name)
                    
                    # Update metadata.json
                    dataset_service.update_metadata(
                        uploaded_file.name,
                        f"datasets/{uploaded_file.name}",
                        metadata["description"],
                        metadata["summary"]
                    )
                    
                    st.success("✅ Dataset processed and added to metadata!")
                    with st.expander("View Generated Metadata"):
                        st.json(metadata)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
                    st.exception(e)


def render_dataset_selection_section(dataset_service: DatasetService) -> str:
    """Render dataset selection section and return selected dataset."""
    st.header("📁 Available Datasets")
    
    datasets = dataset_service.get_datasets_info()
    metadata = dataset_service.get_metadata()
    
    if not datasets:
        st.warning("No datasets found! Upload a CSV file above.")
        return None
    
    # Dataset selector
    dataset_names = [d["name"] for d in datasets]
    selected_dataset = st.selectbox(
        "Select Dataset",
        dataset_names,
        help="Choose a dataset to work with",
        key="dataset_selector"
    )
    
    # Show dataset info
    selected_info = next(d for d in datasets if d["name"] == selected_dataset)
    st.subheader("Dataset Info")
    st.write(f"**Rows:** {selected_info['rows']:,}")
    st.write(f"**Columns:** {selected_info['columns']}")
    
    with st.expander("View Columns"):
        for col in selected_info['column_names']:
            st.write(f"  - `{col}`")
    
    # Show metadata description if available
    dataset_meta = dataset_service.get_dataset_metadata(selected_dataset)
    if dataset_meta:
        st.subheader("Description")
        st.info(dataset_meta.get("description", "No description available"))
        st.caption(f"Summary: {dataset_meta.get('summary', 'N/A')}")
    
    return selected_dataset


def render_sidebar(
    api_key_service: APIKeyService,
    dataset_service: DatasetService,
    metadata_service: MetadataService
) -> str:
    """Render the entire sidebar and return selected dataset."""
    st.header("⚙️ Configuration")
    
    # API Key Management
    render_api_key_section(api_key_service)
    
    st.divider()
    
    # Dataset Upload
    render_dataset_upload_section(dataset_service, metadata_service, api_key_service)
    
    st.divider()
    
    # Dataset Selection
    selected_dataset = render_dataset_selection_section(dataset_service)
    
    return selected_dataset
