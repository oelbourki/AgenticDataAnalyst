"""
Agent classes for file access, code generation, and code execution.
"""

# Standard library imports
import os
import re
import json
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# LangChain core imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

# Try to import LangChain output parsers (modern first, fallback to legacy)
try:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
except ImportError:
    # Fallback for older LangChain versions
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

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

# Try to import embedding models for semantic file search
EMBEDDINGS_AVAILABLE = False
OPENAI_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        # Alternative: use OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings
        OPENAI_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        pass

# Try to import sklearn for cosine similarity (optional)
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Local imports
from .utils import generate_detailed_dataframe_description, analyze_csv, json_to_text

# Load environment variables
load_dotenv()

class QueryInfo(BaseModel):
    """Information extracted from a user query."""
    file_names: List[str] = Field(default_factory=list, description="Potential file names mentioned in the query")
    column_names: List[str] = Field(default_factory=list, description="Potential column names or data fields mentioned in the query")
    data_types: List[str] = Field(default_factory=list, description="Data types or file formats mentioned (e.g., CSV, Excel, JSON)")
    analysis_type: Optional[str] = Field(None, description="Type of analysis requested (e.g., visualization, regression, classification)")
    time_periods: List[str] = Field(default_factory=list, description="Time periods mentioned (e.g., years, months, quarters)")

class FileAccessAgent:
    """Agent responsible for reading files and providing context."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        # Set metadata_path first, before any early returns
        # Check for datasets folder first, then fallback to artifacts
        if os.path.exists("datasets/metadata.json"):
            self.metadata_path = "datasets/metadata.json"
        elif os.path.exists("artifacts/file_metadata.json"):
            self.metadata_path = "artifacts/file_metadata.json"
        else:
            self.metadata_path = "datasets/metadata.json"  # Default to datasets
        
        # Set system prompt and parser (used by all code paths)
        self.system_prompt = """
        You are a file access agent. Your job is to read files and provide their contents.
        You have access to the host file system, but you can only read files, not write to them.
        When asked about a file, read it and provide its contents in a structured format.
        If the user doesn't specify a file directly, determine the most relevant file based on their query.
        """
        # Initialize the Pydantic output parser
        self.parser = PydanticOutputParser(pydantic_object=QueryInfo)
        
        # Try Gemini first (default)
        if GEMINI_AVAILABLE:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.0,
                    google_api_key=api_key
                )
                self.model_mini = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",  # Use flash lite for mini tasks
                    temperature=0.0,
                    google_api_key=api_key
                )
                print("current working directory", os.getcwd())
                return
            else:
                print("Warning: GOOGLE_API_KEY not found. Trying OpenAI fallback...")
        
        # Fallback to OpenAI only if Gemini not available or no key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print("Using OpenAI as fallback (Gemini not available or no key)")
            self.model = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=openai_key)
            self.model_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai_key)
        else:
            raise ValueError(
                "No API key found. Please set either:\n"
                "  - GOOGLE_API_KEY (preferred) in .env file or run 'python scripts/setup_gemini.py'\n"
                "  - OPENAI_API_KEY in .env file\n"
                "Gemini is preferred. Install with: pip install langchain-google-genai"
            )

        print("current working directory", os.getcwd())
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and read file if requested."""
        
        # Extract the user query
        user_query = state.get("user_query", "")
        messages = state.get("messages", [])
        selected_dataset = state.get("selected_dataset")  # Dataset selected in Streamlit UI
        
        # Determine which file to use based on the query and metadata
        filename, file_info = self._determine_file(user_query, messages, selected_dataset)
        
        # Handle case where file_info is None
        if file_info is None:
            file_info = {}
        
        report = file_info.get("report", None) if file_info else None
        report_text = file_info.get("report_text", None) if file_info else None
        df_description = file_info.get("df_description", None) if file_info else None
        print("filename", filename)
        # Read the file if it exists and is a valid file path (not a query word)
        file_content = ""
        # Only process if filename is not empty and is a real file (not a directory or query word)
        if filename and filename.strip() and os.path.exists(filename) and os.path.isfile(filename):
            try:
                with open(filename, 'r') as f:
                    file_content = f.read()
                
                print(f"Cleaning sandbox and copying file to Docker container: {filename}")
                # First delete everything in the sandbox
                # Note: Docker operations are deprecated - codibox handles this now
                # Keeping for backward compatibility but these may fail if Docker not available
                try:
                    cleanup_result = subprocess.run(
                        ["docker", "exec", "sandbox", "sh", "-c", "rm -rf /tmp/*"],
                        capture_output=True,
                        timeout=30,
                        check=False
                    )
                    if cleanup_result.returncode != 0:
                        print(f"Warning: Docker cleanup failed (non-critical): {cleanup_result.stderr.decode()}")
                    
                    # Copy the file to the Docker container
                    copy_result = subprocess.run(
                        ["docker", "cp", filename, f"sandbox:/tmp/{os.path.basename(filename)}"],
                        capture_output=True,
                        timeout=30,
                        check=False
                    )
                    if copy_result.returncode == 0:
                        print(f"File copied to Docker container: {filename}")
                    else:
                        print(f"Warning: Docker copy failed (non-critical): {copy_result.stderr.decode()}")
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    # Docker not available or command timed out - this is OK, codibox will handle it
                    print(f"Info: Docker operations skipped (codibox will handle file transfer): {str(e)}")
                # Preview the first few lines
                preview = "\n".join(file_content.split("\n")[:10])
                # file_info = f"File '{filename}' has been read and copied to the sandbox. Preview:\n{preview}"
                file_info = f"File '{filename}' has been read and copied to the sandbox."

                print(f"File info: {file_info}")
            except Exception as e:
                file_info = f"Error reading file: {str(e)}"
        elif filename and filename.strip():
            # Only show error if filename was actually provided (not empty)
            file_info = f"File '{filename}' does not exist or is not a valid file."
            print(f"Warning: File '{filename}' not found. Will proceed without file-specific information.")
        else:
            # No file specified - this is OK, the system can work with sample data
            file_info = "No specific file selected. Will generate code that can work with sample data or handle data loading."
            print("Info: No file specified. Code generation will proceed without specific file.")
        
        # Update the state
        return {
            **state,
            "file_info": file_info,
            "file_content": file_content,
            "filename": os.path.basename(filename) if filename else "",
            "file_path": filename if filename else "",
            "report": report,
            "report_text": report_text,
            "df_description": df_description
        }
    
    
    def _determine_file(
        self, 
        user_query: str, 
        messages: List[BaseMessage], 
        selected_dataset: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Determine which file to use based on the user query and metadata.
        
        Args:
            user_query: The user's query/question
            messages: Previous messages in the conversation
            selected_dataset: Dataset filename selected in Streamlit UI (used by default unless user specifies otherwise)
        """
        # First, check if the user explicitly specified a different file in the query
        query_words = user_query.strip().split()
        explicit_file_mentioned = False
        explicit_file_path = None
        
        for word in query_words:
            if os.path.exists(word) and os.path.isfile(word):
                explicit_file_mentioned = True
                explicit_file_path = word
                break
        
        # If user explicitly mentioned a file, use it
        if explicit_file_mentioned and explicit_file_path:
            print(f"User explicitly specified file: {explicit_file_path}")
            return explicit_file_path, {"file_path": explicit_file_path, "filename": os.path.basename(explicit_file_path)}
        
        # If selected_dataset is provided, use it by default (unless user explicitly mentioned another file)
        if selected_dataset:
            print(f"Using selected dataset from Streamlit UI: {selected_dataset}")
            try:
                # Try to find the selected dataset in metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Find the selected dataset in metadata
                    for item in metadata:
                        # Match by filename (exact match or basename match)
                        item_filename = item.get("filename", "")
                        if item_filename == selected_dataset or item_filename == os.path.basename(selected_dataset):
                            file_path = item.get("file_path", "")
                            # Resolve file path - try multiple locations
                            if file_path and not os.path.isabs(file_path):
                                if os.path.exists(f"datasets/{file_path}"):
                                    file_path = f"datasets/{file_path}"
                                elif os.path.exists(f"datasets/{item.get('filename', '')}"):
                                    file_path = f"datasets/{item.get('filename', '')}"
                                elif os.path.exists(file_path):
                                    pass  # Path is already correct
                            
                            # Also check if file_path ends with the selected_dataset
                            if not file_path or not os.path.exists(file_path):
                                if item.get("file_path", "").endswith(selected_dataset) or item.get("file_path", "").endswith(os.path.basename(selected_dataset)):
                                    file_path = item.get("file_path", "")
                                    if not os.path.isabs(file_path):
                                        if os.path.exists(f"datasets/{file_path}"):
                                            file_path = f"datasets/{file_path}"
                            
                            if file_path and os.path.exists(file_path) and os.path.isfile(file_path):
                                print(f"✅ Using selected dataset from metadata: {file_path}")
                                return file_path, item
                            else:
                                print(f"⚠️  Found metadata entry for {selected_dataset} but file not found at: {file_path}")
                
                # If not found in metadata, try direct path resolution
                possible_paths = [
                    f"datasets/{selected_dataset}",
                    selected_dataset,
                    f"datasets/{os.path.basename(selected_dataset)}"
                ]
                for path in possible_paths:
                    if os.path.exists(path) and os.path.isfile(path):
                        print(f"✅ Using selected dataset (direct path): {path}")
                        # Try to get metadata for this file
                        if os.path.exists(self.metadata_path):
                            with open(self.metadata_path, 'r') as f:
                                metadata = json.load(f)
                            for item in metadata:
                                if item.get("filename") == os.path.basename(path) or item.get("file_path", "").endswith(os.path.basename(path)):
                                    return path, item
                        return path, {"filename": os.path.basename(path), "file_path": path}
                
                print(f"⚠️  Selected dataset '{selected_dataset}' not found. Will fall back to intelligent file selection.")
            except Exception as e:
                print(f"❌ Error using selected dataset: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue to fallback logic below
        
        # If no selected_dataset or it wasn't found, use metadata to find the most relevant file
        try:
            print("metadata_path", self.metadata_path)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                print("metadata", metadata)
                
                # Filter for relevant file types
                important_info = []
                for item in metadata:
                    file_info = {
                        "filename": item.get("filename"),
                        "summary": item.get("summary") or item.get("description", ""),  # Support both 'summary' and 'description'
                        "file_path": item.get("file_path"),
                        "description": item.get("description", item.get("summary", "")),  # Add description field
                        "report": item.get("report"),
                        "report_text": item.get("report_text"),
                        "df_description": item.get("df_description")
                    }
                    # Resolve file path - check if it's relative to datasets folder
                    if file_info["file_path"] and not os.path.isabs(file_info["file_path"]):
                        # Try datasets folder first
                        if os.path.exists(f"datasets/{file_info['file_path']}"):
                            file_info["file_path"] = f"datasets/{file_info['file_path']}"
                        elif os.path.exists(file_info["file_path"]):
                            pass  # Path is already correct
                        elif os.path.exists(f"datasets/{file_info['filename']}"):
                            file_info["file_path"] = f"datasets/{file_info['filename']}"
                    
                    if file_info["filename"].endswith(("csv", "xlsx", "json")):
                        important_info.append(file_info)
                # If no files found, return empty
                if not important_info:
                    return "", {}
                print("===============\n")
                print("important_info", important_info)
                print("===============\n")
                
                # Prepare batch of messages for evaluation
                batch_messages = []
                
                # Define FileRelevance model (moved outside loop for efficiency)
                class FileRelevance(BaseModel):
                    score: float = Field(..., description="A number from 0 to 10 indicating relevance (10 being perfect match)")
                    certain: bool = Field(..., description="Whether you're certain this is the correct file")
                    reason: str = Field(..., description="A brief explanation of your reasoning")
                
                # Create parser once (outside loop)
                parser = PydanticOutputParser(pydantic_object=FileRelevance)
                
                for file_info in important_info:
                    
                    messagesx = messages + [
                        SystemMessage(content=f"""
                        You are a file selection assistant. Based on the user's query and a file's metadata,
                        determine how relevant this file is to the query.
                        
                        {parser.get_format_instructions()}
                        
                        Only mark "certain" as true if you are absolutely confident this is the correct file.
                        """),
                        HumanMessage(content=f"""
                        User query: {user_query}
                        
                        File metadata:
                        Filename: {file_info.get("filename", "")}
                        Summary: {file_info.get("summary", "")}
                        Description: {file_info.get("description", file_info.get("summary", ""))}
                        
                        How relevant is this file to the query?
                        """)
                    ]
                    batch_messages.append(messagesx)
                
                print("batch_messages", batch_messages)
                # Batch evaluate all files at once
                batch_responses = self.model_mini.batch(batch_messages)
                print("batch_responses", batch_responses)
                # Process responses
                results = []
                for i, response in enumerate(batch_responses):
                    try:
                        file_info = important_info[i]
                        response_text = response.content.strip()
                        parser = PydanticOutputParser(pydantic_object=FileRelevance)
                        result = parser.parse(response_text)
                        results.append((file_info, result.score, result.certain))
                    except Exception as e:
                        print(f"Error parsing response for file {important_info[i].get('filename')}: {str(e)}")
                        results.append((important_info[i], 0, False))
                print("results", results)
                # Check if any file is a certain match
                certain_matches = [r for r in results if r[2]]
                if certain_matches:
                    # Return the certain match with highest score
                    best_match = max(certain_matches, key=lambda x: x[1])
                    print(f"Found certain match: {best_match[0]['filename']} with score {best_match[1]}")
                    return best_match[0]["file_path"], best_match[0]
                
                # Otherwise return the highest scoring file
                if results:
                    best_match = max(results, key=lambda x: x[1])
                    if best_match[1] > 0:  # Only return if score is positive
                        print(f"Best match: {best_match[0]['filename']} with score {best_match[1]}")
                        return best_match[0]["file_path"], best_match[0]
                
                # If LLM scoring didn't find good matches, try embedding-based search
                print("LLM scoring didn't find good matches. Trying embedding-based search...")
                embedding_match = self._find_file_with_embeddings(user_query, important_info)
                if embedding_match:
                    print(f"Found file using embeddings: {embedding_match['filename']}")
                    return embedding_match["file_path"], embedding_match
                
                # If no good matches, return the first file as fallback
                if metadata and len(metadata) > 0:
                    print(f"Using fallback: first file in metadata")
                    return metadata[0].get("file_path", ""), metadata[0]
        except Exception as e:
            print(f"Error determining file from metadata: {str(e)}")
        
        # Try embedding-based search on all available files as last resort
        print("Trying embedding-based search on all files...")
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                important_info = [item for item in metadata if item.get("filename", "").endswith(("csv", "xlsx", "json"))]
                embedding_match = self._find_file_with_embeddings(user_query, important_info)
                if embedding_match:
                    print(f"Found file using embeddings: {embedding_match['filename']}")
                    return embedding_match.get("file_path", ""), embedding_match
        except Exception as e:
            print(f"Error in embedding search: {str(e)}")
        
        # Don't use query words as filenames - return empty and let the system handle it
        # The code generation agent can work without a specific file if needed
        print("Warning: No matching file found. Will proceed without specific file and generate sample data.")
        return "", {}
    
    def _find_file_with_embeddings(self, user_query: str, file_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the most relevant file using embedding-based semantic search.
        
        Args:
            user_query: The user's query
            file_list: List of file metadata dictionaries
            
        Returns:
            Best matching file info or None
        """
        if not file_list:
            return None
        
        try:
            # Try sentence transformers first
            if EMBEDDINGS_AVAILABLE:
                return self._find_file_with_sentence_transformers(user_query, file_list)
            # Fallback to OpenAI embeddings
            elif OPENAI_EMBEDDINGS_AVAILABLE:
                return self._find_file_with_openai_embeddings(user_query, file_list)
            # Fallback to simple keyword matching
            else:
                return self._find_file_with_keywords(user_query, file_list)
        except Exception as e:
            print(f"Error in embedding search: {str(e)}")
            return self._find_file_with_keywords(user_query, file_list)
    
    def _find_file_with_sentence_transformers(self, user_query: str, file_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find file using sentence transformers embeddings."""
        try:
            # Use a lightweight model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create text representations for each file
            file_texts = []
            for file_info in file_list:
                text_parts = [
                    file_info.get("filename", ""),
                    file_info.get("summary", ""),
                    file_info.get("description", ""),
                    file_info.get("report_text", "")[:500] if file_info.get("report_text") else ""
                ]
                file_texts.append(" ".join([p for p in text_parts if p]))
            
            # Get embeddings
            query_embedding = model.encode([user_query])
            file_embeddings = model.encode(file_texts)
            
            # Calculate cosine similarity
            if SKLEARN_AVAILABLE:
                similarities = cosine_similarity(query_embedding, file_embeddings)[0]
            else:
                # Fallback: manual cosine similarity calculation
                query_vec = np.array(query_embedding[0])
                similarities = []
                for file_vec in file_embeddings:
                    file_vec = np.array(file_vec)
                    similarity = np.dot(query_vec, file_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(file_vec))
                    similarities.append(similarity)
                similarities = np.array(similarities)
            
            # Find best match
            best_idx = similarities.argmax()
            best_score = similarities[best_idx]
            
            if best_score > 0.3:  # Threshold for relevance
                print(f"Embedding match found with similarity: {best_score:.3f}")
                return file_list[best_idx]
            else:
                print(f"Best embedding similarity too low: {best_score:.3f}")
                return None
                
        except ImportError:
            print("sentence-transformers or sklearn not available")
            return None
        except Exception as e:
            print(f"Error in sentence transformer search: {str(e)}")
            return None
    
    def _find_file_with_openai_embeddings(self, user_query: str, file_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find file using OpenAI embeddings."""
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
            
            # Create text representations
            file_texts = []
            for file_info in file_list:
                text_parts = [
                    file_info.get("filename", ""),
                    file_info.get("summary", ""),
                    file_info.get("description", "")
                ]
                file_texts.append(" ".join([p for p in text_parts if p]))
            
            # Get embeddings
            query_embedding = embeddings.embed_query(user_query)
            file_embeddings = embeddings.embed_documents(file_texts)
            
            # Calculate cosine similarity
            query_vec = np.array(query_embedding)
            similarities = []
            for file_vec in file_embeddings:
                file_vec = np.array(file_vec)
                similarity = np.dot(query_vec, file_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(file_vec))
                similarities.append(similarity)
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score > 0.3:
                print(f"OpenAI embedding match found with similarity: {best_score:.3f}")
                return file_list[best_idx]
            else:
                return None
                
        except Exception as e:
            print(f"Error in OpenAI embedding search: {str(e)}")
            return None
    
    def _find_file_with_keywords(self, user_query: str, file_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Fallback: Find file using simple keyword matching."""
        query_lower = user_query.lower()
        query_words = set(query_lower.split())
        
        best_match = None
        best_score = 0
        
        for file_info in file_list:
            score = 0
            # Check filename
            filename_lower = file_info.get("filename", "").lower()
            for word in query_words:
                if word in filename_lower:
                    score += 2
            
            # Check summary/description
            summary_lower = (file_info.get("summary", "") + " " + file_info.get("description", "")).lower()
            for word in query_words:
                if word in summary_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = file_info
        
        if best_score > 0:
            print(f"Keyword match found with score: {best_score}")
            return best_match
        
        return None

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def _extract_query_info(self, query: str) -> QueryInfo:
        """Extract structured information from the user query using LangChain's PydanticOutputParser."""
        try:
            # Get the format instructions from the parser
            format_instructions = self.parser.get_format_instructions()
            
            # Create a prompt template
            template = """
            Extract structured information from this user query about data analysis.
            
            User query: {query}
            
            {format_instructions}
            """
            
            # Create a human message prompt template
            human_message_prompt = HumanMessagePromptTemplate.from_template(template)
            chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
            
            # Create the chain: prompt -> LLM -> parser
            chain = chat_prompt | self.model_mini | self.parser
            
            # Invoke the chain with the query
            query_info = chain.invoke({"query": query, "format_instructions": format_instructions})
            
            return query_info
        except Exception as e:
            print(f"Error extracting query info with PydanticOutputParser: {str(e)}")
            # If the structured approach fails, fall back to the previous method
            try:
                # Create a system prompt that instructs the LLM to extract structured information
                system_prompt = """
                You are an AI assistant that extracts structured information from user queries about data analysis.
                Extract the following information from the user's query:
                1. Potential file names mentioned
                2. Potential column names or data fields mentioned
                3. Data types or file formats mentioned (e.g., CSV, Excel, JSON)
                4. Type of analysis requested (e.g., visualization, regression, classification)
                5. Time periods mentioned (e.g., years, months, quarters)
                
                Format your response as a valid JSON object that matches this Pydantic model:
                ```
                class QueryInfo:
                    file_names: List[str]  # Potential file names mentioned
                    column_names: List[str]  # Potential column names mentioned
                    data_types: List[str]  # Data types mentioned (e.g., csv, json, excel)
                    analysis_type: Optional[str]  # Type of analysis requested
                    time_periods: List[str]  # Time periods mentioned
                ```
                
                Return ONLY the JSON object, nothing else.
                """
                
                # Create messages for the model
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Extract structured information from this query: {query}")
                ]
                
                # Get response from the model
                response = self.model_mini.invoke(messages)
                response_content = response.content.strip()
                
                # Extract the JSON part if there's any extra text
                json_match = re.search(r'({.*})', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_content
                
                # Parse the JSON response into the Pydantic model
                query_info_dict = json.loads(json_str)
                query_info = QueryInfo(**query_info_dict)
                
                return query_info
            except Exception as e:
                print(f"Error in fallback extraction method: {str(e)}")
                # Return a default QueryInfo if all extraction methods fail
                return QueryInfo()

# This Python script contains the enhanced system prompt as a multi-line string.
# The prompt provides detailed instructions for generating data analysis code.
# You can use this string as a reference or save it to a file if needed.

system_prompt3 = """
System Prompt: Expert Python Agent for Data Analysis, Visualization, and Machine Learning

You are an expert Python agent specializing in data analysis, visualization, and predictive modeling. Your task is to generate complete, well-structured, and fully executable Python scripts that help users analyze datasets, create insightful visualizations, and build simple machine learning models for tasks like prediction and forecasting.

Your code will run in a secure Docker container with the following libraries available:
✅ pandas – Data manipulation
✅ numpy – Numerical computing
✅ matplotlib – Data visualization
✅ seaborn – Statistical visualizations
✅ scikit-learn – Machine learning models
✅ statsmodels – Statistical modeling
✅ xlsxwriter – Excel file creation, and manipulation

--------------------------------------------------

Code Generation Guidelines

1. Data Loading & Cleaning
- **CRITICAL: If a dataset file is provided (filename or file_path), you MUST use that exact dataset. DO NOT generate fake or sample data.**
- Load the dataset from the provided filename/file_path into a pandas DataFrame using the exact path provided.
- **ONLY generate sample/fake data if NO file is provided AND the user explicitly asks for sample data.**
- Inspect data types, missing values, and potential inconsistencies.
- Apply necessary transformations such as:
  - Handling missing values
  - Converting data types
  - Encoding categorical variables
  - Removing duplicates or outliers
- Clearly document assumptions and preprocessing steps.

2. Exploratory Data Analysis (EDA) & Visualizations
- Carefully examine data types and values before subsetting data:
  • Check for mixed types within columns and convert as appropriate
  • Handle categorical data correctly (use appropriate encodings or groupings)
  • Be aware of null/NaN values and how they affect operations
  • For numeric data, consider the scale, range, and presence of outliers
  • For datetime data, ensure proper formatting and timezone consistency
- Generate key summary statistics and insights.
- Select appropriate visualizations based on the dataset and user request:
  - Histograms & boxplots – Data distribution
  - Scatter plots & correlation heatmaps – Relationships between variables
  - Line plots – Trends over time
  - Bar charts & pie charts – Comparisons between categories
- Use seaborn and matplotlib for high-quality, well-annotated plots.
- Ensure all visualizations include titles, axis labels, legends, and clear formatting.
- **IMPORTANT: Display plots inline in Jupyter notebook**
  • Use `plt.show()` or `display()` to display plots inline in the notebook
  • DO NOT save plots to files using `plt.savefig()` unless explicitly requested by the user
  • The code will be executed in a Jupyter notebook, and plots will be automatically captured in the notebook output
  • Only save plots if the user specifically asks to save them to a file
- Always use the Description column to create the visualizations if it exists

3. Machine Learning & Predictive Modeling (if applicable)
- Identify the correct ML approach for the task:
  - Regression for numerical prediction
  - Classification for categorical prediction
  - Clustering for grouping data
  - Time series forecasting for trend prediction Use the statsmodels library for time series forecasting ARIMA model, Create a Date Column from the dataset columns like SCENARIO, PERIOD.
- Perform feature selection and engineering where necessary.
- Split the dataset into training and testing sets.
- Train a suitable model using scikit-learn or statsmodels.
- Evaluate model performance with metrics like RMSE, accuracy, precision, recall, or R².
- Visualize model results (e.g., residual plots, feature importance charts, confusion matrix).
- **Display all model visualization plots inline using `plt.show()` - do not save to files unless explicitly requested**

4. Code Quality & Documentation
- Self-contained: The script must be executable without external dependencies.
- Well-commented: Every step should have clear explanations.
- Readable output: Use print() statements to summarize findings and model performance.
- No external API calls or internet access: The code must work entirely within the given environment.
- **Jupyter Notebook Execution**: The code will be converted to a Jupyter notebook and executed. All plots should be displayed inline using `plt.show()` so they appear in the notebook output cells.
- **Dataset Usage**: When file information is provided, you MUST use the exact dataset file specified. Never generate fake or synthetic data when a real dataset is available.

--------------------------------------------------

Expected Output
A fully functional Python script that enables the user to:
✅ Load and clean the dataset
✅ Perform insightful exploratory data analysis
✅ Create high-quality visualizations
✅ Train simple ML models for prediction and forecasting
✅ Interpret key results through comments and structured outputs
✅ Create Excel files with the results and formulas if needed

Your goal is to deliver expert-level code that is easy to understand, well-documented, and directly useful for data-driven decision-making.
"""



class CodeGenerationAgent:
    """Agent responsible for generating Python code based on user queries."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        # Try Gemini first (default)
        if GEMINI_AVAILABLE:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.0,
                    google_api_key=api_key
                )
                self.system_prompt = system_prompt3
                return
            else:
                print("Warning: GOOGLE_API_KEY not found. Trying OpenAI fallback...")
        
        # Fallback to OpenAI only if Gemini not available or no key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print("Using OpenAI as fallback (Gemini not available or no key)")
            self.model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=openai_key
            )
            self.system_prompt = system_prompt3
        else:
            raise ValueError(
                "No API key found. Please set either:\n"
                "  - GOOGLE_API_KEY (preferred) in .env file or run 'python scripts/setup_gemini.py'\n"
                "  - OPENAI_API_KEY in .env file\n"
                "Gemini is preferred. Install with: pip install langchain-google-genai"
            )
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Python code based on the user query and file information."""
        
        # Extract information from the state
        user_query = state.get("user_query", "")
        file_info = state.get("file_info", "")
        filename = state.get("filename", "")
        report = state.get("report", None)
        file_path = state.get("file_path", None)
        report_text = state.get("report_text", None)
        df_description = state.get("df_description", None)
        
        # Only process file if it exists and is a valid file path
        if file_path and os.path.exists(file_path) and os.path.isfile(file_path):
            print("file_path>>>>", file_path)
            try:
                df = pd.read_csv(file_path)
                df_description = generate_detailed_dataframe_description(df)
                summary = analyze_csv(file_path)
                file_info += f"\n------------------------\n{summary}\n------------------------\n"
                file_info += f"\n------------------------\n{df_description}\n------------------------\n"
                
                # Add column information with Excel-style indices
                column_info = "\nColumn Excel Indices:\n"
                for i, col in enumerate(df.columns):
                    excel_col = chr(65 + i) if i < 26 else chr(64 + i//26) + chr(65 + i%26)
                    column_info += f"{excel_col}: {col}\n"
                file_info += f"\n------------------------\n{column_info}\n------------------------\n"
                print("file_info>>>>", file_info)
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                file_info += f"\nWarning: Could not read file {file_path}: {str(e)}\n"
        elif file_path:
            print(f"Warning: File path '{file_path}' does not exist or is not a valid file")
            file_info += f"\nWarning: File '{file_path}' not found. Will proceed without file-specific information.\n"

        # Create messages for the model
        
        # Determine file instruction based on whether a file is provided
        file_instruction = ""
        if file_path and os.path.exists(file_path) and os.path.isfile(file_path):
            # File is provided and exists - MUST use it
            file_instruction = f"""
            
CRITICAL INSTRUCTIONS - DATASET PROVIDED:
- A dataset file has been provided: '{file_path}' (filename: '{filename}')
- You MUST use this exact dataset file - DO NOT generate fake or sample data
- Load the dataset using: df = pd.read_csv('{filename}') or the exact file path provided
- Use the actual columns and data from this file
- The file information above contains details about the dataset structure
- DO NOT create sample data - use the real dataset provided
"""
        elif filename and os.path.exists(file_path) if file_path else False:
            file_instruction = f" using the file at '/tmp/{filename}'. You MUST use this exact dataset - DO NOT generate fake data."
        elif filename:
            file_instruction = f" Note: A file '{filename}' was mentioned but may not be available. If the file exists, use it. Otherwise, generate code that can work with sample data or handle missing files gracefully."
        else:
            # Enhanced instruction for no-file scenario
            file_instruction = """
            
IMPORTANT: No specific data file was provided. You MUST:
1. Create realistic sample data that matches the user's query
2. The sample data should be representative and meaningful
3. For "customer age distribution" - create a dataset with customer ages (e.g., 100-1000 customers with ages 18-80)
4. Make the data realistic with appropriate distributions
5. Include all necessary columns to answer the question
6. Display visualizations inline in the Jupyter notebook using plt.show() - DO NOT save to files
7. The code will be executed in a Jupyter notebook, and plots will be automatically captured in the output

Example for "customer age distribution":
- Create a pandas DataFrame with 'age' column
- Generate realistic age data (e.g., normal distribution around age 35-45)
- Create a histogram showing the distribution using plt.show() to display it inline
- Include summary statistics
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            User question/task: {user_query}

            File information: {file_info}

            Generate a plan to solve or perform the task (pay attention to data types and values when selecting the subset data) then write the code to answer this question.{file_instruction}
            
            Make sure to:
            - **If a dataset file is provided (file_path or filename), you MUST use that exact dataset - DO NOT generate fake data**
            - **Only create sample data if NO file is provided**
            - Use the actual columns and data from the provided dataset file
            - Use appropriate data types and realistic values
            - Display all visualizations inline in the Jupyter notebook using plt.show() - DO NOT save plots to files unless explicitly requested
            - Include clear titles, labels, and legends
            - Print summary statistics and insights
            - Remember: The code will be executed in a Jupyter notebook, and plots will be automatically captured in the notebook output
            """)
        ]
        try:
            response = self.model.invoke(
                state['messages'] + messages
            )
        except Exception as e:
            # Handle rate limit and quota errors
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "quota" in error_str.lower():
                retry_delay = "50 seconds"
                if "retry in" in error_str.lower():
                    delay_match = re.search(r"retry in ([\d.]+)s", error_str, re.IGNORECASE)
                    if delay_match:
                        retry_delay = f"{delay_match.group(1)} seconds"
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
                raise ValueError(
                    f"⚠️ **Gemini API Error**\n\n"
                    f"An error occurred while calling the Gemini API:\n\n"
                    f"**Error:** {error_str[:500]}"
                )
            else:
                raise
        content = response.content
        
        # Simple code extraction - look for Python code blocks
        code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)
        
        if code_blocks:
            python_code = code_blocks[0].strip()
        else:
            # If no code block is found, assume the entire response is code
            python_code = content
        
        # Update the state
        return {
            **state,
            "python_code": python_code,
            "code_explanation": content
        }

class CodeExecutor:
    """Component responsible for executing Python code in the Docker container."""
    
    def __init__(self, max_attempts: int = 3):
        """Initialize the CodeExecutor with a maximum number of refinement attempts."""
        self.max_attempts = max_attempts
        # Try Gemini first (default)
        if GEMINI_AVAILABLE:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    temperature=0.0,
                    google_api_key=api_key
                )
                return
            else:
                print("Warning: GOOGLE_API_KEY not found. Trying OpenAI fallback...")
        
        # Fallback to OpenAI only if Gemini not available or no key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print("Using OpenAI as fallback (Gemini not available or no key)")
            self.model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=openai_key
            )
        else:
            raise ValueError(
                "No API key found. Please set either:\n"
                "  - GOOGLE_API_KEY (preferred) in .env file or run 'python scripts/setup_gemini.py'\n"
                "  - OPENAI_API_KEY in .env file\n"
                "Gemini is preferred. Install with: pip install langchain-google-genai"
            )
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Python code in the Docker container with automatic refinement on errors."""
        
        # Extract the Python code from the state
        python_code = state.get("python_code", "")
        
        if not python_code:
            return {
                **state,
                "execution_result": "No Python code to execute."
            }
        filename = state.get("filename", None)
        # Initialize variables for the refinement loop
        current_code = python_code
        attempts = 0
        execution_history = []
        
        images_list = []  # Track images across all attempts
        markdown_with_images = None  # Store markdown_with_images from last successful execution
        while attempts < self.max_attempts:
            # Execute the current version of the code
            execution_result, success, csv_file_list, attempt_images, attempt_markdown_with_images = self._execute_code(current_code, filename)
            
            # Collect images from this attempt
            if attempt_images:
                images_list.extend(attempt_images)
            
            # Store markdown_with_images from successful execution
            if success and attempt_markdown_with_images:
                markdown_with_images = attempt_markdown_with_images
            
            # Record this attempt
            execution_history.append({
                "attempt": attempts + 1,
                "code": current_code,
                "result": execution_result,
                "success": success
            })
            
            # If execution was successful, break out of the loop
            if success:
                break
            
            # Otherwise, try to refine the code
            attempts += 1
            if attempts < self.max_attempts:
                current_code = self._refine_code(current_code, execution_result)
        
        # Prepare the final execution result with history
        final_result = "# Code Execution Results\n\n"
        
        for i, attempt in enumerate(execution_history):
            final_result += f"## Attempt {attempt['attempt']}\n\n"
            if i > 0:
                final_result += "### Refined Code\n```python\n" + attempt['code'] + "\n```\n\n"
            final_result += "### Execution Output\n" + attempt['result'] + "\n\n"
            if attempt['success']:
                final_result += "✅ **Execution successful!**\n\n"
            else:
                final_result += "❌ **Execution failed.**\n\n"
        print("CodeExecutor csv_file_list", csv_file_list)
        print("CodeExecutor images_list", images_list)
        # Update the state with the final code and execution result
        state_update = {
            **state,
            "python_code": execution_history[-1]["code"],  # Use the last version of the code
            "csv_file_list": csv_file_list,
            "images": images_list,  # Add images to state
            "execution_result": final_result
        }
        # Add markdown_with_images if available (for ResultAdapter to use)
        if markdown_with_images:
            state_update["markdown_with_images"] = markdown_with_images
        return state_update
    
    def _execute_code(self, code: str, filename: str) -> Tuple[str, bool, List[str], List[str], Optional[str]]:
        """
        Execute the given Python code using codibox (Docker or Host backend).
        
        Args:
            code: The Python code to execute
            filename: The name of the input file
            
        Returns:
            Tuple of (execution_result, success_flag, csv_file_list, images_list)
        """
        # Import codibox package - use unified CodeExecutor
        from codibox import CodeExecutor
        
        # Use Host backend by default (works on Streamlit Cloud)
        # Only use Docker if explicitly requested via environment variable
        use_docker = os.environ.get("USE_DOCKER_BACKEND", "").lower() == "true"
        
        # Check if Docker is actually available (if requested)
        docker_available = False
        if use_docker:
            docker_available = (
                os.path.exists("/var/run/docker.sock") or 
                os.path.exists("/.dockerenv")
            )
        
        # Initialize codibox executor - Host backend by default
        if use_docker and docker_available:
            # Docker backend: Secure, isolated execution (only if explicitly requested)
            executor = CodeExecutor(
                backend="docker",
                container_name="sandbox",
                auto_setup=True  # Automatically set up container if needed
            )
        else:
            # Host backend: Fast execution, works on Streamlit Cloud (DEFAULT)
            executor = CodeExecutor(
                backend="host",
                auto_install_deps=True  # Automatically install dependencies
            )
        
        # Prepare input files
        input_files = []
        if filename:
            # Resolve the full path to the file
            file_path = Path(filename)
            if not file_path.is_absolute():
                # Try relative to datasets directory
                datasets_path = Path(__file__).parent.parent / "datasets" / filename
                if datasets_path.exists():
                    input_files = [str(datasets_path)]
                elif file_path.exists():
                    input_files = [str(file_path)]
            else:
                if file_path.exists():
                    input_files = [str(file_path)]
        
        # Execute code using codibox
        result = executor.execute(
            code=code,
            input_files=input_files
        )
        
        # Use processed markdown if available (has resolved image paths), otherwise fall back to raw markdown
        # This is better for AI processing as it has resolved image paths
        execution_result = result.markdown_processed if result.markdown_processed else result.markdown
        # Also store markdown_with_images for direct use (has base64-embedded images)
        # This will be stored in state for ResultAdapter to use
        markdown_with_images = result.markdown_with_images if result.markdown_with_images else None
        success = result.success
        
        # Process CSV files - convert to old format
        csv_filepath_list = []
        if result.csv_files:
            # The old format expected paths like "downloads/filename.csv"
            # Codibox saves CSV files to /tmp/downloads/ by default
            for csv_file in result.csv_files:
                csv_filename = os.path.basename(csv_file)
                # Keep the full path for now, but format as expected
                csv_filepath_list.append(csv_file)
        
        # Extract images from result
        images_list = result.images if result.images else []
        print(f"CodeExecutor: Found {len(images_list)} images: {images_list}")
        
        return execution_result, success, csv_filepath_list, images_list, markdown_with_images
    
    def _refine_code(self, code: str, error_output: str) -> str:
        """
        Use the LLM to refine the code based on the error output.
        
        Args:
            code: The original Python code
            error_output: The error output from the execution
            
        Returns:
            Refined Python code
        """
        try:
            # Create messages for the model
            messages = [
                SystemMessage(content="""
                You are an expert Python debugging assistant. Your task is to fix Python code that has execution errors.
                Analyze the error message carefully and provide a corrected version of the code.
                
                Guidelines:
                1. Focus on fixing the specific error mentioned in the error message.
                2. Make minimal changes to the code while fixing the issue.
                3. Maintain the original functionality and intent of the code.
                4. Return ONLY the complete fixed code, with no explanations or markdown formatting.
                """),
                HumanMessage(content=f"""
                The following Python code produced an error during execution:
                
                ```python
                {code}
                ```
                
                Here is the error output:
                
                ```
                {error_output}
                ```
                
                Please provide a fixed version of the code.
                """)
            ]
            
            # Get response from the model with error handling
            try:
                response = self.model.invoke(messages)
                refined_code = response.content.strip()
            except Exception as e:
                # Handle rate limit and quota errors
                error_str = str(e)
                if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "quota" in error_str.lower():
                    retry_delay = "50 seconds"
                    if "retry in" in error_str.lower():
                        delay_match = re.search(r"retry in ([\d.]+)s", error_str, re.IGNORECASE)
                        if delay_match:
                            retry_delay = f"{delay_match.group(1)} seconds"
                    print(f"⚠️ API Rate Limit Exceeded. Wait {retry_delay} before retrying.")
                    # Return original code if refinement fails due to rate limit
                    return code
                elif "ChatGoogleGenerativeAIError" in str(type(e).__name__):
                    print(f"⚠️ Gemini API Error during code refinement: {error_str[:200]}")
                    # Return original code if refinement fails
                    return code
                else:
                    # Re-raise other errors
                    raise
            
            # Extract code from markdown if necessary
            code_blocks = re.findall(r'```python(.*?)```', refined_code, re.DOTALL)
            if code_blocks:
                refined_code = code_blocks[0].strip()
            
            return refined_code
            
        except Exception as e:
            print(f"Error during code refinement: {str(e)}")
            # If refinement fails, return the original code
            return code 