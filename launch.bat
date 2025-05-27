@echo off
title Lumina App Launcher

echo Checking for virtual environment (venv)...
IF NOT EXIST venv\Scripts\activate.bat (
    echo Virtual environment not found!
    echo Please run the install.bat script first to set up the environment.
    pause
    exit /b 1
)

echo Checking for application script (app.py)...
IF NOT EXIST app.py (
    echo Application script (app.py) not found!
    echo Please ensure app.py is in the current directory.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate
IF ERRORLEVEL 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Launching Lumina Image App Web UI...
python app.py
IF ERRORLEVEL 1 (
    echo The application encountered an error. Please check the console output above.
) ELSE (
    echo The application has finished or the window was closed.
)

echo.
pause
