#!/bin/bash
# Startup script for the frontend

cd "$(dirname "$0")/webapp"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the development server
npm run dev



