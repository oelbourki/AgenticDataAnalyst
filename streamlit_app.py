"""
AgenticDataAnalyst - AI-Powered Data Analysis Dashboard
"""

import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="AgenticDataAnalyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import codibox (supports both Docker and Host backends)
CODIBOX_AVAILABLE = False
try:
    from codibox import CodeExecutor
    CODIBOX_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Codibox package not available. Dataset upload and metadata generation will still work.")

# Backend selection: Host by default, Docker only if explicitly requested
import os
USE_DOCKER = os.environ.get("USE_DOCKER_BACKEND", "").lower() == "true"
DOCKER_AVAILABLE = False

if USE_DOCKER and CODIBOX_AVAILABLE:
    # Check if Docker is actually available (only if explicitly requested)
    DOCKER_AVAILABLE = (
        os.path.exists("/var/run/docker.sock") or 
        os.path.exists("/.dockerenv")
    )

# Try to import workflow (may fail if Docker not available)
WORKFLOW_AVAILABLE = False
try:
    from agent_coder.simple_workflow import process_query, simple_coder
    from agent_coder.utils import analyze_csv, generate_detailed_dataframe_description
    WORKFLOW_AVAILABLE = True
except Exception as e:
    if DOCKER_AVAILABLE:
        st.error(f"Error loading workflow: {e}")

# Check execution backend status
def check_execution_backend():
    """Check execution backend status (Host by default, Docker if requested)."""
    if not CODIBOX_AVAILABLE:
        return {"available": False, "backend": None, "ready": False}
    
    try:
        # Host backend is default - always ready (no container needed)
        if not USE_DOCKER:
            return {
                "available": True,
                "backend": "host",
                "ready": True,
                "status": "ready"
            }
        
        # Docker backend (only if explicitly requested)
        if DOCKER_AVAILABLE:
            executor = CodeExecutor(backend="docker", container_name="sandbox")
            container_running = executor.check_container()
            return {
                "available": True,
                "backend": "docker",
                "ready": container_running,
                "status": "running" if container_running else "not_running"
            }
        else:
            # Docker requested but not available - fall back to host
            return {
                "available": True,
                "backend": "host",
                "ready": True,
                "status": "ready (docker not available, using host)"
            }
    except Exception as e:
        # On error, fall back to host backend
        return {
            "available": True,
            "backend": "host",
            "ready": True,
            "status": f"ready (fallback from error: {str(e)})"
        }

# Get API key from Streamlit secrets or environment
def get_api_key():
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
                    api_key = secrets_obj.get('GOOGLE_API_KEY', None)
                    if api_key:
                        return api_key
                elif secrets_obj and 'GOOGLE_API_KEY' in secrets_obj:
                    return secrets_obj['GOOGLE_API_KEY']
            except Exception:
                # Any error accessing secrets (e.g., "No secrets found", FileNotFoundError, etc.)
                # This is expected when secrets.toml doesn't exist - fall back to env
                pass
    except Exception:
        # Any error with st.secrets attribute - fall back to env
        pass
    
    # Fallback to environment variable
    return os.getenv("GOOGLE_API_KEY", "")

def set_api_key_env(key: str):
    """Set API key in environment (for current session)."""
    os.environ["GOOGLE_API_KEY"] = key

def generate_dataset_metadata(df: pd.DataFrame, filename: str) -> dict:
    """Generate AI-powered description and summary for a dataset."""
    try:
        # Try to import Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        api_key = get_api_key()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.0,
            google_api_key=api_key
        )
        
        # Generate detailed analysis
        if WORKFLOW_AVAILABLE:
            detailed_description = generate_detailed_dataframe_description(df)
        else:
            # Fallback basic description
            detailed_description = f"""
Dataset: {filename}
Rows: {len(df)}
Columns: {len(df.columns)}
Column names: {', '.join(df.columns[:10])}
Data types: {', '.join([f"{col}: {str(dtype)}" for col, dtype in df.dtypes.items()][:5])}
"""
        
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
        
        response = model.invoke(messages)
        content = response.content.strip()
        
        # Extract JSON from response
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        metadata = json.loads(content)
        return metadata
        
    except Exception as e:
        st.warning(f"Could not generate AI metadata: {e}. Using basic description.")
        # Fallback to basic description
        return {
            "description": f"Dataset with {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}",
            "summary": f"{filename} dataset with {len(df)} rows"
        }

def save_uploaded_file(uploaded_file, datasets_dir: Path):
    """Save uploaded CSV file to datasets directory."""
    file_path = datasets_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def update_metadata_json(filename: str, file_path: str, description: str, summary: str, metadata_path: Path):
    """Add or update entry in metadata.json."""
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []
    
    # Remove existing entry if present
    metadata = [m for m in metadata if m.get("filename") != filename]
    
    # Add new entry
    metadata.append({
        "filename": filename,
        "file_path": file_path,
        "description": description,
        "summary": summary
    })
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def load_metadata():
    """Load dataset metadata."""
    metadata_path = Path("datasets/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return []

def load_datasets_info():
    """Get information about available datasets."""
    datasets_dir = Path("datasets")
    datasets = []
    
    if datasets_dir.exists():
        for csv_file in datasets_dir.glob("*.csv"):
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

# Example prompts for each dataset
EXAMPLE_PROMPTS = {
    "titanic.csv": [
        "Show me the survival rate by passenger class",
        "Create a chart comparing survival rates between males and females",
        "Visualize the age distribution of passengers",
        "Show survival rate by age groups",
        "Compare survival rates across different embarkation ports"
    ],
    "iris.csv": [
        "Create a scatter plot of sepal length vs sepal width colored by species",
        "Show the distribution of petal lengths for each species",
        "Visualize the relationship between all flower measurements",
        "Compare average measurements across different species",
        "Create a box plot showing petal width by species"
    ],
    "sales_data.csv": [
        "Show me sales trends over time",
        "Create a chart comparing sales by region",
        "Visualize sales by product category",
        "Show monthly sales trends",
        "Compare average sales amount across regions"
    ],
    "customer_data.csv": [
        "Show customer age distribution",
        "Create a chart comparing purchases by location",
        "Visualize average order value by gender",
        "Show customer distribution by location",
        "Compare total purchases across different age groups"
    ]
}

# Initialize session state
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

def main():
    st.title("🤖 AgenticDataAnalyst")
    st.markdown("**AI-Powered Data Analysis Platform** - Upload datasets, ask questions, get insights!")
    
    # Check execution backend status (non-blocking)
    backend_status = check_execution_backend()
    execution_ready = backend_status.get("ready", False)
    backend_type = backend_status.get("backend", "unknown")
    
    # Show backend status
    if backend_status["available"]:
        if backend_type == "host":
            st.success("✅ Host backend ready - Code execution available")
            st.caption("Using fast host execution mode (default) - works on Streamlit Cloud")
        elif backend_type == "docker":
            if execution_ready:
                st.success("✅ Docker backend ready - Code execution available")
                st.caption("Using secure Docker execution mode")
            else:
                with st.expander("⚠️ Docker Container Status", expanded=False):
                    st.info("""
                    **Docker container is not running.**
                    
                    The app will automatically set up the container when needed.
                    You can still upload datasets and generate metadata.
                    """)
    else:
        with st.expander("⚠️ Execution Backend", expanded=False):
            st.warning("""
            **Code execution not available.**
            
            Codibox package is not installed. You can still:
            - Upload and manage datasets
            - Generate AI-powered metadata
            - Preview datasets
            """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Management
        st.subheader("🔑 API Key")
        current_key = get_api_key()
        
        # Check if key is set via secrets (handle gracefully)
        # Note: Accessing st.secrets may raise exception if secrets.toml doesn't exist - this is normal
        key_from_secrets = False
        try:
            if hasattr(st, 'secrets'):
                try:
                    # Try to access secrets - Streamlit raises exception if secrets.toml doesn't exist
                    secrets_obj = st.secrets
                    if secrets_obj:
                        if hasattr(secrets_obj, 'get'):
                            api_key_from_secrets = secrets_obj.get('GOOGLE_API_KEY', None)
                        elif 'GOOGLE_API_KEY' in secrets_obj:
                            api_key_from_secrets = secrets_obj['GOOGLE_API_KEY']
                        else:
                            api_key_from_secrets = None
                        
                        if api_key_from_secrets:
                            key_from_secrets = True
                            st.success("✅ API key configured via Streamlit secrets")
                            st.caption("To change, update secrets in Streamlit Cloud settings")
                except Exception:
                    # Any error accessing secrets (e.g., "No secrets found") - this is normal
                    # Secrets file doesn't exist - use manual input instead
                    # This is expected behavior for local development
                    pass
        except Exception:
            # Any error with st.secrets attribute - use manual input
            pass
        
        # Show manual input if not using secrets
        if not key_from_secrets:
            api_key_input = st.text_input(
                "Google Gemini API Key",
                value=current_key if current_key else "",
                type="password",
                help="Enter your Google Gemini API key. Get one from https://aistudio.google.com/app/apikey",
                key="api_key_input"
            )
            
            if st.button("💾 Save API Key", use_container_width=True):
                if api_key_input:
                    set_api_key_env(api_key_input)
                    st.success("✅ API key saved for this session!")
                    st.rerun()
                else:
                    st.error("Please enter an API key")
            
            if current_key:
                st.success("✅ API key configured")
            else:
                st.warning("⚠️ API key not set")
                st.info("💡 Tip: Set `GOOGLE_API_KEY` in Streamlit secrets for persistent storage")
        
        st.divider()
        
        # Dataset Upload
        st.header("📤 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file to analyze. The AI will automatically generate a description."
        )
        
        if uploaded_file is not None:
            datasets_dir = Path("datasets")
            datasets_dir.mkdir(exist_ok=True)
            
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
                        df = pd.read_csv(file_path)
                        st.info(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
                        
                        # Check API key for metadata generation
                        if not get_api_key():
                            st.warning("⚠️ API key not set. Generating basic metadata...")
                            metadata = {
                                "description": f"Dataset with {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns[:10])}",
                                "summary": f"{uploaded_file.name} dataset with {len(df)} rows"
                            }
                        else:
                            # Generate metadata
                            with st.spinner("🤖 AI is generating description..."):
                                metadata = generate_dataset_metadata(df, uploaded_file.name)
                        
                        # Update metadata.json
                        metadata_path = datasets_dir / "metadata.json"
                        update_metadata_json(
                            uploaded_file.name,
                            f"datasets/{uploaded_file.name}",
                            metadata["description"],
                            metadata["summary"],
                            metadata_path
                        )
                        
                        st.success("✅ Dataset processed and added to metadata!")
                        with st.expander("View Generated Metadata"):
                            st.json(metadata)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
                        st.exception(e)
        
        st.divider()
        
        # Dataset Selection
        st.header("📁 Available Datasets")
        
        datasets = load_datasets_info()
        metadata = load_metadata()
        
        if not datasets:
            st.warning("No datasets found! Upload a CSV file above.")
            selected_dataset = None
        else:
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
            dataset_meta = next((m for m in metadata if m["filename"] == selected_dataset), None)
            if dataset_meta:
                st.subheader("Description")
                st.info(dataset_meta.get("description", "No description available"))
                st.caption(f"Summary: {dataset_meta.get('summary', 'N/A')}")
    
    # Main content area
    if not datasets:
        st.info("👆 Upload a dataset in the sidebar to get started!")
        return
    
    tab1, tab2, tab3 = st.tabs(["🎯 Query Interface", "📝 Example Prompts", "📊 Dataset Preview"])
    
    with tab1:
        st.header("Ask a Question")
        st.markdown("Enter a question about the selected dataset, and the AI will generate charts and insights automatically!")
        
        # Check if code execution is available
        if not execution_ready or not WORKFLOW_AVAILABLE:
            if not CODIBOX_AVAILABLE:
                st.warning("""
                ⚠️ **Code execution is not available**
                
                Codibox package is not installed. Install with:
                ```bash
                pip install codibox
                ```
                """)
            elif not execution_ready:
                st.warning("""
                ⚠️ **Code execution backend is not ready**
                
                The execution backend is initializing. This may take a moment.
                You can still:
                - Upload and manage datasets
                - Generate metadata
                - Preview datasets
                """)
            else:
                st.warning("⚠️ Workflow not available. Please check the console for errors.")
        
        # Check API key
        api_key = get_api_key()
        if not api_key:
            st.warning("⚠️ Please set your Google Gemini API key in the sidebar to use the query interface.")
        
        # Example prompts for selected dataset
        if selected_dataset and selected_dataset in EXAMPLE_PROMPTS:
            st.subheader("💡 Quick Examples")
            example_cols = st.columns(min(3, len(EXAMPLE_PROMPTS[selected_dataset])))
            for i, example in enumerate(EXAMPLE_PROMPTS[selected_dataset][:3]):
                with example_cols[i]:
                    if st.button(f"Use: {example[:40]}...", key=f"example_{i}", use_container_width=True):
                        st.session_state.user_query = example
                        st.session_state.query_input = example  # Sync with text area
                        st.rerun()
        
        # Query input - sync with session state
        # Initialize query_input in session state if not exists
        if "query_input" not in st.session_state:
            st.session_state.query_input = st.session_state.get("user_query", "")
        
        # Update query_input when user_query changes (from example buttons)
        if st.session_state.get("user_query", "") != st.session_state.get("query_input", ""):
            st.session_state.query_input = st.session_state.get("user_query", "")
        
        user_query = st.text_area(
            "Your Question",
            value=st.session_state.get("query_input", ""),
            height=100,
            placeholder=f"e.g., 'Show me sales trends by region' for {selected_dataset}",
            help="Ask a question about the dataset. The AI will automatically select the right file and generate charts.",
            key="query_input",
            disabled=not (execution_ready and WORKFLOW_AVAILABLE and api_key)
        )
        
        # Sync user_query with query_input when user types
        if user_query != st.session_state.get("user_query", ""):
            st.session_state.user_query = user_query
        
        # Generate button
        if st.button("🚀 Generate Analysis", type="primary", use_container_width=True, disabled=not (execution_ready and WORKFLOW_AVAILABLE and api_key)):
            if not user_query:
                st.warning("Please enter a question!")
            elif not api_key:
                st.error("Please set your Google Gemini API key in the sidebar first!")
            elif not execution_ready:
                st.error(f"Execution backend ({backend_type}) is not ready. Please wait or check status.")
            elif not WORKFLOW_AVAILABLE:
                st.error("Workflow not available. Please check the console for errors.")
            else:
                with st.spinner("🤖 AI is analyzing your data... This may take a minute..."):
                    try:
                        # Process the query
                        result = process_query(simple_coder, user_query)
                        
                        # Display results
                        if result and "messages" in result:
                            last_message = result["messages"][-1]
                            
                            # Show execution status
                            st.success("✅ Analysis completed successfully!")
                            
                            # Display markdown content
                            if hasattr(last_message, 'response_metadata'):
                                metadata_result = last_message.response_metadata
                                
                                # Show the final markdown
                                st.subheader("📊 Analysis Results")
                                
                                # Try to display images directly from execution result
                                display_images = []
                                
                                # First, check metadata for images
                                if "images" in metadata_result and metadata_result["images"]:
                                    display_images = metadata_result["images"]
                                
                                # Also try ImageProcessor as fallback
                                if not display_images:
                                    try:
                                        from codibox import ImageProcessor
                                        processor = ImageProcessor(image_base_dir="/tmp/temp_code_files")
                                        found_images = processor.find_images()
                                        
                                        if found_images:
                                            display_images = found_images
                                    except Exception as e:
                                        st.debug(f"Image processor error: {e}")
                                
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
                                    st.markdown(metadata_result["markdown_content"], unsafe_allow_html=True)
                                
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
                        else:
                            st.error("No results returned. Check the console for errors.")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.exception(e)
    
    with tab2:
        st.header("Example Prompts")
        st.markdown("Click any example to use it in the Query Interface")
        
        if selected_dataset and selected_dataset in EXAMPLE_PROMPTS:
            for i, prompt in enumerate(EXAMPLE_PROMPTS[selected_dataset]):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{i+1}.** {prompt}")
                with col2:
                    if st.button("Use", key=f"use_{i}"):
                        st.session_state.user_query = prompt
                        st.session_state.query_input = prompt  # Sync with text area
                        st.rerun()
        else:
            st.info("No example prompts available for this dataset. Try asking your own question!")
    
    with tab3:
        st.header("Dataset Preview")
        
        # Load and display dataset
        try:
            df = pd.read_csv(selected_info["file_path"])
            
            st.subheader(f"Preview: {selected_dataset}")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            # Data types
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(dtype) for dtype in df.dtypes],
                "Non-Null Count": df.count().values,
                "Null Count": df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.subheader("Numeric Statistics")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main()
