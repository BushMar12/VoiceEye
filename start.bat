@echo off
title VoiceEye
echo Starting VoiceEye...
echo.

:: Check Node.js
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed. Download from https://nodejs.org
    pause
    exit /b 1
)

:: Install dependencies if needed
if not exist node_modules (
    echo Installing dependencies...
    call npm install
)

:: Start dev server and open browser
echo.
echo VoiceEye is starting at https://localhost:5173
echo Press Ctrl+C to stop.
echo.
start "" "https://localhost:5173"
npm run dev
