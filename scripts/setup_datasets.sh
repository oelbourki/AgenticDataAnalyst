#!/bin/bash
# Setup script to download datasets and create metadata

echo "=================================="
echo "Setting up datasets for testing"
echo "=================================="
echo ""

# Step 1: Download/Create datasets
echo "Step 1: Downloading/Creating datasets..."
python download_datasets.py

echo ""
echo "Step 2: Creating metadata file..."
python create_metadata.py

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run: streamlit run streamlit_app.py"
echo "2. Open browser to http://localhost:8501"
echo "3. Select a dataset and try the example prompts!"
echo ""
