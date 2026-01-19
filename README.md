# Agentic Data Analyst

An AI-powered data analysis system that generates Python code from natural language queries and executes it securely in Docker containers.

## ✅ Setup Status

**The `codibox` package is now included in `requirements.txt` and will be installed from pip.** Run `python setup_project.py` or `pip install -r requirements.txt` to install all dependencies including codibox.

## Project Structure

```
AgenticDataAnalyst/
├── agent_coder/              # AI workflow system
│   ├── agents.py             # FileAccessAgent, CodeGenerationAgent, CodeExecutor
│   ├── workflow.py           # LangGraph workflow definitions
│   ├── simple_workflow.py    # Enhanced workflow with result processing
│   ├── utils.py              # Utility functions for CSV analysis
│   ├── main.py               # CLI interface
├── datasets/                 # Data files and metadata
│   ├── *.csv                 # Sample datasets
│   └── metadata.json         # Dataset descriptions
├── docker/                   # Docker configuration
│   ├── Dockerfile            # Container definition
│   └── requirements.txt      # Python packages for container
├── docs/                     # Documentation
│   ├── GEMINI_DEFAULT_FIX.md
│   ├── IMAGE_DISPLAY_FIX.md
│   ├── IMPROVED_FILE_SELECTION.md
│   └── DOCKER_*.md           # Docker-related documentation
├── scripts/                  # Setup and utility scripts
│   ├── setup_gemini.py       # Gemini API key setup
│   ├── setup_docker.py       # Docker container setup
│   ├── check_docker.py       # Docker status checker
│   ├── download_datasets.py  # Download sample datasets
│   ├── create_metadata.py    # Generate dataset metadata
│   └── *.sh                  # Shell scripts
├── streamlit_app.py          # Web dashboard
├── setup_project.py          # Main project setup script
├── requirements.txt          # All project dependencies
├── README.md                 # This file
└── STRUCTURE_ANALYSIS.md     # Structure analysis document
```

## Features

- **Natural Language to Code**: Converts user queries into executable Python code
- **Automatic File Selection**: Intelligently selects relevant datasets based on queries
- **Secure Execution**: Runs code in isolated Docker containers
- **Visualization Generation**: Automatically creates charts and visualizations
- **Error Handling**: Automatic code refinement on execution errors
- **Multiple Interfaces**: CLI, Streamlit web app, and programmatic API

## Architecture

### Core Components

1. **FileAccessAgent**: Selects appropriate data files based on user queries
2. **CodeGenerationAgent**: Generates Python code using LLM (Gemini/OpenAI)
3. **CodeExecutor**: Executes code in Docker containers with error handling
4. **Workflow**: LangGraph-based orchestration of agents

### Execution Flow

```
User Query
    ↓
FileAccessAgent (selects dataset)
    ↓
CodeGenerationAgent (generates Python code)
    ↓
CodeExecutor (executes in Docker)
    ↓
Result Processing (extracts images, CSV files)
    ↓
Display Results
```

## Setup

### Prerequisites

- Python 3.10+
- Docker installed and running
- Google Gemini API key (or OpenAI API key as fallback)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Gemini API key:**
   ```bash
   python scripts/setup_gemini.py
   ```
   Or create a `.env` file with:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

3. **Download sample datasets:**
   ```bash
   python scripts/download_datasets.py
   ```

4. **Set up Docker container:**
   ```bash
   python scripts/setup_docker.py
   ```

### Running the Application

**Streamlit Web App:**
```bash
streamlit run streamlit_app.py
```

**CLI Interface:**
```bash
python -m agent_coder.main
```

## Usage Examples

### Streamlit App

1. Start the app: `streamlit run streamlit_app.py`
2. Select a dataset from the sidebar
3. Enter a natural language query
4. View generated charts and results

### Programmatic Usage

```python
from agent_coder.simple_workflow import process_query, simple_coder

# Process a query
result = process_query(simple_coder, "Show me sales trends by region")

# Access results
print(result["messages"][-1].content)
```

## Configuration

### LLM Models

The system uses Google Gemini by default, with OpenAI as fallback:
- **Preferred**: `gemini-2.5-flash-lite`
- **Fallback**: `gpt-4o-mini` (if Gemini unavailable)

### Docker Container

- **Container Name**: `sandbox`
- **Image**: `python_sandbox:latest`
- **Security**: Network isolation, non-root user, resource limits

## Dependencies

### Core Libraries
- `langchain` - LLM orchestration
- `langgraph` - Workflow management
- `langchain-google-genai` - Gemini integration
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `streamlit` - Web interface

### Docker Container Packages
- Jupyter, nbconvert - Notebook execution
- Data science libraries (pandas, numpy, matplotlib, seaborn)
- Machine learning libraries (scikit-learn, statsmodels)

## Troubleshooting

### Import Errors

If you see `ImportError: No module named 'codibox'`:
- The codibox package is missing and needs to be restored or refactored

### Docker Container Issues

Check container status:
```bash
python scripts/check_docker.py
```

Or manually:
```bash
docker ps -a | grep sandbox
docker start sandbox
```

### API Key Issues

Ensure your API key is set:
```bash
python scripts/setup_gemini.py
```

Or check `.env` file exists with `GOOGLE_API_KEY`.

## Documentation

- **Setup Guides**: See `docs/` folder for detailed documentation
- **Fix Documentation**: `docs/GEMINI_DEFAULT_FIX.md`, `docs/IMAGE_DISPLAY_FIX.md`
- **Feature Documentation**: `docs/IMPROVED_FILE_SELECTION.md`

## Development

### Project Status

⚠️ **Broken**: Missing codibox package dependency

### Known Issues

1. All codibox imports will fail until package is restored or refactored
2. Docker container setup may fail without codibox

### Future Improvements

- Restore or refactor codibox dependency
- Add unit tests
- Improve error messages
- Add result caching
- Support for multiple file inputs

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
