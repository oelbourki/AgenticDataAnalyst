from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from typing import Dict, Any, List, Tuple
from .workflow import create_code_interpreter_graph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import os
# Try to import Gemini first, fallback to OpenAI for backward compatibility
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
    except ImportError:
        # Fallback: define the error class if import fails
        ChatGoogleGenerativeAIError = Exception
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAIError = Exception
    from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
# from core import get_model, settings  # Commented out - module not found
class AgentState(MessagesState, total=False):
    user_query: str
    result: str
    execution_result: str
    new_messages: List[BaseMessage]
    csv_file_list: List[str]
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

# def wrap_model(model: BaseChatModel, instructions: str) -> RunnableSerializable[AgentState, AIMessage]:
#     # model = model.bind_tools(tools)
#     preprocessor = RunnableLambda(
#         lambda state: [SystemMessage(content=instructions)] + state["messages"],
#         name="StateModifier",
#     )
#     return preprocessor | model

# def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
#     m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
#     model_runnable = wrap_model(m)
#     response = model_runnable.invoke(state, config)
class QueryProcessor:
    """A node that processes the initial user query before passing it to the code interpreter."""
    
    def __init__(self):
        pass
        
    def __call__(self, state: AgentState) -> AgentState:
        """Process the user query and prepare it for the code interpreter."""
        # print("QueryProcessor", state)
        # Get the last user message
        last_message = state["messages"][-1]
        user_query = last_message.content
        
        # You can add preprocessing logic here if needed
        processed_content = f"Processing query: {user_query}"
        
        # Add a system message with the processed query
        return {"user_query": user_query}

class StateAdapter:
    """Adapter node that transforms MessagesState into the format expected by code_interpreter."""
    
    def __init__(self):
        pass
    
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Transform MessagesState to code_interpreter input format."""
        # print("StateAdapter", state)
        # Extract the user query from the messages
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not user_messages:
            raise ValueError("No user message found in state")
        
        # Get the most recent user message
        user_query = user_messages[-1].content
        
        # Create a filtered list with only HumanMessage and AIMessage in same order
        new_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)]
        
        # Return the state in the format expected by code_interpreter
        return {
            "user_query": user_query,
            "new_messages": new_messages
        }
import base64
from pathlib import Path
# from utilities import load_bootstrap

# load_bootstrap()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html
import re
# Find all markdown image references with pattern ![png](temp_code_files/filename.png)
pattern = r'!\[png\]\(temp_code_files/([^)]+)\)'

# Replace each match with the HTML version using img_to_html
def replace_with_html(match):
    filename = match.group(1)
    try:
        result = img_to_html(f'/tmp/temp_code_files/{filename}')
    except Exception as e:
        result = img_to_html(f'/tmp/temp_code_files/temp_code_files/{filename}')
    return result
def remove_code_blocks(markdown_text):
    """Remove code blocks from markdown text."""
    # Pattern to match code blocks (both fenced and indented)
    code_block_pattern = r'```[\s\S]*?```|`[\s\S]*?`'
    return re.sub(code_block_pattern, '', markdown_text)
import re
import os
import base64
from pathlib import Path

# Find all markdown image references with pattern ![png](temp_code_files/filename.png)
pattern = r'!\[(.*?)\]\(temp_code_files/([^)]+)\)'

# Replace each match with the HTML version using img_to_html
def replace_with_html(match):
    alt_text = match.group(1)
    filename = match.group(2)
    
    # Try multiple possible paths where images might be located
    possible_paths = [
        f'/tmp/temp_code_files/{filename}',
        f'/tmp/temp_code_files/temp_code_files/{filename}',
        f'/tmp/{filename}',
        f'/tmp/temp_code_files/{os.path.basename(filename)}',
    ]
    
    # Also try to find any image file that matches the pattern (in case filename differs)
    # This handles cases where nbconvert generates different filenames
    base_name = os.path.splitext(os.path.basename(filename))[0]
    if base_name:
        possible_paths.extend([
            f'/tmp/temp_code_files/{base_name}.png',
            f'/tmp/temp_code_files/{base_name}.jpg',
            f'/tmp/temp_code_files/{base_name}.jpeg',
        ])
    
    # Try each path
    for img_path in possible_paths:
        if os.path.exists(img_path):
            try:
                return img_to_html(img_path)
            except Exception as e:
                print(f"Warning: Could not convert image {img_path}: {e}")
                continue
    
    # If no image found, try to find any image in the temp_code_files directory
    # This is a fallback for when the filename doesn't match exactly
    temp_dir = '/tmp/temp_code_files'
    if os.path.exists(temp_dir):
        try:
            # Look for any PNG files
            import glob
            image_files = glob.glob(f'{temp_dir}/*.png') + glob.glob(f'{temp_dir}/temp_code_files/*.png')
            if image_files:
                # Use the most recently modified image
                image_files.sort(key=os.path.getmtime, reverse=True)
                return img_to_html(image_files[0])
        except Exception as e:
            print(f"Warning: Could not find fallback image: {e}")
    
    # If still not found, return a placeholder
    print(f"Warning: Image not found: {filename}. Tried paths: {possible_paths}")
    return f'<p style="color: red;">⚠️ Image not found: {filename}</p>'

# Replace image references with base64 encoded images
def replace_with_base64(markdown_text):
    """Replace image references in markdown with base64 encoded images."""
    return re.sub(pattern, replace_with_html, markdown_text)

# Inject base64 image bytes directly
def inject_image_bytes(image_path):
    """Convert an image file to base64 and return the HTML img tag with embedded bytes."""
    try:
        img_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return f"<img src='data:image/png;base64,{encoded}' class='img-fluid'>"
    except Exception as e:
        return f"<p>Error loading image: {str(e)}</p>"
system_prompt = """
            You are a data analysis results interpreter.
            Your task is to create a clear, concise markdown response that answers the user's original question.
            
            You will be given:
            1. The original user query
            2. A markdown document containing the results of executing a Jupyter notebook
            
            Your job is to:
            1. Extract the key insights from the notebook execution results
            2. Focus on the specific information that answers the user's question
            3. Present the findings in a well-structured markdown format
            4. Include relevant visualizations and data points from the results
            5. Provide a concise summary that directly addresses what the user wanted to know
            
            Make your response conversational and accessible, avoiding technical jargon unless necessary.
            Highlight the most important findings at the beginning of your response.
            If the results don't fully answer the user's question, acknowledge this limitation.
            """
class ResultAdapter:
    """Adapter node that transforms code_interpreter results back to MessagesState."""
    
    def __init__(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Try Gemini first (default)
        if GEMINI_AVAILABLE:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    temperature=0.0,
                    google_api_key=api_key
                )
                self.system_prompt = """
            You are a comprehensive data analysis results interpreter.
            Your task is to create a detailed, thorough markdown response that fully answers the user's original question with extensive explanations.
            
            You will be given:
            1. The original user query
            2. A markdown document containing the results of executing a Jupyter notebook
            
            Your job is to:
            1. Extract ALL key insights from the notebook execution results, leaving nothing out
            2. Focus on providing comprehensive information that answers the user's question in detail
            3. Present the findings in a well-structured markdown format with multiple sections and subsections
            4. Include ALL visualizations and data points from the results - every chart, graph, and plot should be incorporated
            5. Provide an extensive analysis that thoroughly addresses what the user wanted to know
            6. Explain the significance of each visualization in detail
            7. Include all tables and numerical results with explanations of their importance
            
            Make your response informative and detailed, explaining technical concepts when they appear.
            Present a comprehensive summary of findings at the beginning of your response.
            Include all visualizations with thorough captions explaining what they show.
            Don't include any code in your response.
            If the results don't fully answer the user's question, acknowledge this limitation and suggest what additional analysis might help.
            """
                return
            else:
                print("Warning: GOOGLE_API_KEY not found. Trying OpenAI fallback...")
        
        # Fallback to OpenAI only if Gemini not available or no key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print("Using OpenAI as fallback (Gemini not available or no key)")
            self.model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.0,
                api_key=openai_key
            )
            self.system_prompt = """
            You are a comprehensive data analysis results interpreter.
            Your task is to create a detailed, thorough markdown response that fully answers the user's original question with extensive explanations.
            
            You will be given:
            1. The original user query
            2. A markdown document containing the results of executing a Jupyter notebook
            
            Your job is to:
            1. Extract ALL key insights from the notebook execution results, leaving nothing out
            2. Focus on providing comprehensive information that answers the user's question in detail
            3. Present the findings in a well-structured markdown format with multiple sections and subsections
            4. Include ALL visualizations and data points from the results - every chart, graph, and plot should be incorporated
            5. Provide an extensive analysis that thoroughly addresses what the user wanted to know
            6. Explain the significance of each visualization in detail
            7. Include all tables and numerical results with explanations of their importance
            
            Make your response informative and detailed, explaining technical concepts when they appear.
            Present a comprehensive summary of findings at the beginning of your response.
            Include all visualizations with thorough captions explaining what they show.
            Don't include any code in your response.
            If the results don't fully answer the user's question, acknowledge this limitation and suggest what additional analysis might help.
            """
        else:
            raise ValueError(
                "No API key found. Please set either:\n"
                "  - GOOGLE_API_KEY (preferred) in .env file or run 'python scripts/setup_gemini.py'\n"
                "  - OPENAI_API_KEY in .env file\n"
                "Gemini is preferred. Install with: pip install langchain-google-genai"
            )
    def __call__(self, state: AgentState) -> AgentState:
        """Transform code_interpreter result to MessagesState format."""
        # print("ResultAdapter", state)
        # Extract the result from the code interpreter
        # Assuming the code interpreter returns a result field or similar
        # Adjust this based on the actual structure of the code interpreter's output
        csv_file_list = state.get("csv_file_list", [])
        
        # Prefer markdown_with_images if available (has base64-embedded images, ready for AI)
        # Otherwise use execution_result (has resolved paths from markdown_processed)
        markdown_with_images = state.get("markdown_with_images", None)
        result_contentX = markdown_with_images if markdown_with_images else state.get("execution_result", "No result provided")
        
        # Also get images from the execution result if available
        # The CodeExecutor might have image paths in the state
        execution_images = state.get("images", [])
        if execution_images:
            print(f"Found {len(execution_images)} images from execution: {execution_images}")
        
        print("result_contentX>>>\n\n", result_contentX[:200] if len(result_contentX) > 200 else result_contentX)
        print("\n\n")
        # markdown = re.sub(pattern, replace_with_html, result_content)
        # Remove code blocks from markdown to make the output cleaner

        
        # Clean up the markdown by removing code blocks if needed
        # Only apply this if we want to hide code in the output
        # markdown = remove_code_blocks(markdown)
        
        # You can uncomment the line above to remove code blocks
        # or add a condition to selectively remove them based on state
        # markdown = remove_code_blocks(markdown)
        user_query = state.get("user_query", "")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            User question: {user_query}
            --------------------------------
            markdown: {result_contentX}         
            --------------------------------
            Generate markdown to answer the question.
            """)
        ]
        
        # Get response from the model with error handling
        try:
            response = self.model.invoke(messages)
        except Exception as e:
            # Handle rate limit and quota errors
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "quota" in error_str.lower():
                # Extract retry delay if available
                retry_delay = "50 seconds"
                if "retry in" in error_str.lower():
                    import re
                    delay_match = re.search(r"retry in ([\d.]+)s", error_str, re.IGNORECASE)
                    if delay_match:
                        retry_delay = f"{delay_match.group(1)} seconds"
                
                # Raise a user-friendly error
                raise ValueError(
                    f"⚠️ **API Rate Limit Exceeded**\n\n"
                    f"Your Gemini API quota has been exceeded. The free tier allows 20 requests per day per model.\n\n"
                    f"**Solutions:**\n"
                    f"1. Wait {retry_delay} before trying again\n"
                    f"2. Check your quota and billing: https://ai.dev/rate-limit\n"
                    f"3. Upgrade your API plan for higher limits\n"
                    f"4. Try again later (quota resets daily)\n\n"
                    f"**Error details:** {error_str[:200]}..."
                )
            elif "ChatGoogleGenerativeAIError" in str(type(e).__name__):
                # Other Gemini API errors
                raise ValueError(
                    f"⚠️ **Gemini API Error**\n\n"
                    f"An error occurred while calling the Gemini API:\n\n"
                    f"**Error:** {error_str[:500]}"
                )
            else:
                # Re-raise other errors as-is
                raise
        
        # Extract the Python code from the response
        # This is a simplified approach - in a real system, you'd want more robust parsing
        content = response.content
        
        # Process images in both the LLM-generated content and raw execution result
        # If markdown_with_images was used, images are already embedded as base64
        # Otherwise, process the markdown to add base64 images
        if markdown_with_images:
            # Images are already embedded as base64, just use as-is
            result_content = result_contentX
            # Still process LLM-generated content in case it has image references
            content = re.sub(pattern, replace_with_html, content)
        else:
            # Process the raw execution result to find all images and convert to base64
            result_content = re.sub(pattern, replace_with_html, result_contentX)
            # Also process images in the LLM-generated summary
            content = re.sub(pattern, replace_with_html, content)
            
            # Additionally, try using codibox ImageProcessor for better image handling
            try:
                from codibox import ImageProcessor
                processor = ImageProcessor(image_base_dir="/tmp/temp_code_files")
                # Find all images and ensure they're included
                found_images = processor.find_images()
                if found_images:
                    print(f"Found {len(found_images)} images: {found_images}")
                    # Process the markdown with ImageProcessor
                    result_content = processor.process_markdown(result_contentX)
                    content = processor.process_markdown(content)
            except Exception as e:
                print(f"Warning: Could not use ImageProcessor: {e}")
                # Continue with regex-based replacement

        # Include images in metadata
        image_paths = execution_images if execution_images else []
        
        # Also try to find images using ImageProcessor
        try:
            from codibox import ImageProcessor
            processor = ImageProcessor(image_base_dir="/tmp/temp_code_files")
            found_images = processor.find_images()
            if found_images:
                # Combine with execution images, avoiding duplicates
                for img in found_images:
                    if img not in image_paths:
                        image_paths.append(img)
        except Exception as e:
            print(f"Warning: Could not use ImageProcessor: {e}")
        
        return {
            "messages": AIMessage(
                content=str("Execution successful!"), 
                response_metadata={
                    "markdown": result_content, 
                    "markdown_content": content, 
                    "csv_file_list": csv_file_list,
                    "images": image_paths  # Include images in metadata
                }, 
                name="code_interpreter"
            )
        }

def create_simple_graph(name: str = "simple_workflow"):
    """Create a simple graph that uses the code interpreter as a subgraph."""
    
    # Initialize the query processor and adapters
    query_processor = QueryProcessor()
    state_adapter = StateAdapter()
    result_adapter = ResultAdapter()
    
    # Get the code interpreter graph
    code_interpreter = create_code_interpreter_graph(name="code_interpreter_subgraph")
    
    # Create a new graph with MessagesState
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    workflow.add_node("query_processor", query_processor)
    workflow.add_node("state_adapter", state_adapter)
    workflow.add_node("code_interpreter", code_interpreter)
    workflow.add_node("result_adapter", result_adapter)
    
    # Define the edges
    workflow.add_edge("query_processor", "state_adapter")
    workflow.add_edge("state_adapter", "code_interpreter")
    workflow.add_edge("code_interpreter", "result_adapter")
    workflow.add_edge("result_adapter", END)
    
    # Set the entry point
    workflow.set_entry_point("query_processor")
    
    # Compile the graph
    return workflow.compile(name=name)

def process_query(graph, user_query: str) -> Dict[str, Any]:
    """Process a user query through the simple workflow."""
    
    # Initialize the state with the user query as a message
    initial_state = AgentState(messages=[
        HumanMessage(content=user_query)
    ])
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result

# Create the graph
simple_coder = create_simple_graph(name="simple_coder") 

# # Process a user query
# user_query = "What is the distribution of accident counts? Is it normally distributed or skewed?"
# result = process_query(simple_graph, user_query)
# print(result)

