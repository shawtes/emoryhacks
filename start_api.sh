#!/bin/bash
# Startup script for the API server

cd "$(dirname "$0")/emoryhacks"

# Activate virtual environment if it exists
if [ -d "../.venv311" ]; then
    source ../.venv311/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000



