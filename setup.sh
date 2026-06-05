#!/usr/bin/env bash
set -e

echo "=== Meeting Transcriber setup ==="

# Find python3
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "❌ python3 not found. Install from https://python.org or via: brew install python"
    exit 1
fi

echo "✓ Using $($PYTHON --version)"

# Create venv
$PYTHON -m venv .venv
echo "✓ Virtual environment created"

# Activate
source .venv/bin/activate

# Upgrade pip quietly
pip install --upgrade pip -q

# Install deps
pip install -r requirements.txt

echo ""
echo "✅ Setup complete."
echo ""
echo "To start the app:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
