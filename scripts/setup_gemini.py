"""
Setup script for Google Gemini API key configuration.
Run this script to set up your Gemini API key.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def setup_gemini_api_key():
    """Interactive setup for Gemini API key."""
    print("="*70)
    print("Google Gemini API Key Setup")
    print("="*70)
    print()
    
    # Check if .env file exists
    env_file = Path(".env")
    load_dotenv()  # Load existing .env if it exists
    
    # Check if key already exists
    existing_key = os.getenv("GOOGLE_API_KEY")
    if existing_key:
        print(f"✅ Found existing GOOGLE_API_KEY: {existing_key[:10]}...{existing_key[-4:]}")
        response = input("\nDo you want to update it? (y/n): ").strip().lower()
        if response != 'y':
            print("Keeping existing API key.")
            return
    
    print("Please enter your Google Gemini API key.")
    print("You can get one from: https://aistudio.google.com/app/apikey")
    print()
    
    api_key = input("Enter your GOOGLE_API_KEY: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Read existing .env file if it exists
    env_content = ""
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
    
    # Update or add GOOGLE_API_KEY
    lines = env_content.split('\n') if env_content else []
    updated = False
    new_lines = []
    
    for line in lines:
        if line.startswith('GOOGLE_API_KEY='):
            new_lines.append(f'GOOGLE_API_KEY={api_key}')
            updated = True
        else:
            new_lines.append(line)
    
    if not updated:
        new_lines.append(f'GOOGLE_API_KEY={api_key}')
    
    # Write back to .env file
    with open(env_file, 'w') as f:
        f.write('\n'.join(new_lines))
        if not new_lines[-1].endswith('\n'):
            f.write('\n')
    
    print(f"\n✅ API key saved to {env_file.absolute()}")
    print("\nYou can now use the LangChain chart generator with Gemini!")
    print("\nExample usage:")
    print("  from agent_coder.simple_workflow import process_query, simple_coder")
    print("  workflow = ChartGenerationWorkflow()")
    print("  result = workflow.generate_charts('Create a line chart')")


if __name__ == "__main__":
    setup_gemini_api_key()
