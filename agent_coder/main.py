from .workflow import create_code_interpreter_graph, process_user_query
import os

def main():
    """Main application entry point."""
    
    print("Setting up the code interpreter...")
    
    # Create the workflow
    graph = create_code_interpreter_graph()
    
    print("Code interpreter is ready!")
    
    while True:
        print("\nEnter a file path and your question (or 'exit' to quit):")
        user_input = input("> ")
        
        if user_input.lower() == 'exit':
            break
        
        # Process the user query
        result = process_user_query(graph, user_input)
        
        # Display the results
        print("\n--- File Information ---")
        print(result.get("file_info", "No file information available."))
        
        print("\n--- Generated Code ---")
        print(result.get("python_code", "No code generated."))
        
        print("\n--- Execution Result ---")
        print(result.get("execution_result", "No execution result available."))

if __name__ == "__main__":
    main() 