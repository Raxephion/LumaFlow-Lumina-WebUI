@echo off
title Lumina App Installer

echo Checking for existing virtual environment (venv)...
IF NOT EXIST venv (
    echo Creating Python virtual environment (venv)...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment. Please ensure Python is installed and in PATH.
        pause
        exit /b 1
    )
) ELSE (
    echo Virtual environment (venv) already exists. Skipping creation.
)

echo Activating virtual environment...
call venv\Scripts\activate
IF ERRORLEVEL 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install --upgrade -r requirements.txt
IF ERRORLEVEL 1 (
    echo Failed to install dependencies. Please check requirements.txt and your internet connection.
    pause
    exit /b 1
)

echo.
echo Installation complete.
echo.
echo --------------------------------------------------------------------
echo To run the app:
echo 1. Open a new Command Prompt or PowerShell window.
echo 2. Navigate to this project directory (e.g., cd C:\path\to\your\LuminaAppFolder)
echo 3. Activate the virtual environment by running: venv\Scripts\activate
echo 4. Then run the application: python app.py
echo --------------------------------------------------------------------
echo.
pause
