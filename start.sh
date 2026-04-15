#!/bin/bash
echo "Starting VoiceEye..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed. Download from https://nodejs.org"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo ""
echo "VoiceEye is starting at https://localhost:5173"
echo "Press Ctrl+C to stop."
echo ""

# Open browser (macOS or Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "https://localhost:5173" &
elif command -v xdg-open &> /dev/null; then
    xdg-open "https://localhost:5173" &
fi

npm run dev
