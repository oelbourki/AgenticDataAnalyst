#!/bin/bash
# Install all required dependencies

echo "=================================="
echo "Installing Dependencies"
echo "=================================="
echo ""

echo "Installing LangChain packages..."
pip install langgraph langchain langchain-google-genai langchain-core langchain-openai

echo ""
echo "Installing Streamlit..."
pip install streamlit

echo ""
echo "Installing other dependencies..."
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
pip install jupyter nbconvert jupytext ipynb-py-convert
pip install python-dotenv xlsxwriter tenacity pydantic

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "Verify installation:"
echo "  python -c 'import langgraph; print(\"✅ langgraph\")'"
echo "  python -c 'import streamlit; print(\"✅ streamlit\")'"
echo ""
