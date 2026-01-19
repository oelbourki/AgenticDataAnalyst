from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from typing import Dict, Any, List, Tuple
from .agents import FileAccessAgent, CodeGenerationAgent, CodeExecutor
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.messages import AIMessage, SystemMessage
# from core import get_model, settings  # Commented out - module not found

class AgentState(MessagesState, total=False):
    user_query: str
    execution_result: str
    python_code: str
    code_explanation: str
    file_info: str
    file_content: str
    filename: str
    file_path: str
    report: str
    report_text: str
    df_description: str
    csv_file_list: List[str]
    images: List[str]  # Add images to state
    """`total=False` is PEP589 specs.
    
    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

def wrap_model(model: BaseChatModel, instructions: str) -> RunnableSerializable[AgentState, AIMessage]:
    # model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

# def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
#     m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
#     model_runnable = wrap_model(m)
#     response = model_runnable.invoke(state, config)
def create_code_interpreter_graph(name: str = "code_interpreter"):
    """Create a LangGraph workflow for the code interpreter."""
    
    # Initialize the agents (all use Gemini now)
    file_access_agent = FileAccessAgent()
    code_generation_agent = CodeGenerationAgent(model_name="gemini-2.5-flash-lite")
    code_executor = CodeExecutor()
    
    # Create a new graph
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    workflow.add_node("file_access", file_access_agent)
    workflow.add_node("code_generation", code_generation_agent)
    workflow.add_node("code_execution", code_executor)
    
    # Define the edges
    workflow.add_edge("file_access", "code_generation")
    workflow.add_edge("code_generation", "code_execution")
    workflow.add_edge("code_execution", END)
    
    # Set the entry point
    workflow.set_entry_point("file_access")
    
    # Compile the graph
    return workflow.compile(name=name)

def process_user_query(graph, user_query: str) -> Dict[str, Any]:
    """Process a user query through the workflow."""
    
    # Initialize the state with the user query
    initial_state = {"user_query": user_query}
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result 

graph = create_code_interpreter_graph(name="code_interpreter")
