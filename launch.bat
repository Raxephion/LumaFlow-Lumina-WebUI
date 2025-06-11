@echo off
title Lumina App - DETAILED LAUNCH TEST

echo Script started.


echo Checking for venv activation script (venv\Scripts\activate.bat)...
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo    [ERROR] venv\Scripts\activate.bat was NOT found!
    echo    Please run install.bat first to create the virtual environment.
    
    GOTO :EOF
)
echo    [OK] venv\Scripts\activate.bat WAS found.


echo Attempting to CALL venv\Scripts\activate.bat...
call "venv\Scripts\activate.bat"
echo Returned from CALL venv\Scripts\activate.bat.


echo Checking which Python executable is now in PATH:
where python
echo    (The path above should point to your venv's Python, e.g., ...\venv\Scripts\python.exe)

echo Attempting to run 'python --version' using the venv's Python:
python --version
echo Finished 'python --version'.


echo Checking for application script (app.py)...
IF NOT EXIST "app.py" (
    echo    [ERROR] app.py not found in the current directory.
    echo    Please ensure app.py is present.
    
    GOTO :EOF
)
echo    [OK] app.py WAS found.


echo About to attempt launching 'python app.py' using 'cmd /k'.
echo    This command ('cmd /k python app.py') is designed to keep this window open
echo    after Python starts. If it closes instantly, there was an immediate error
echo    within Python or with how 'cmd /k' is being handled by your system.
echo    Any Python errors should be visible in this window.


cmd /k python app.py

echo           THIS LINE SHOULD NOT BE REACHED if cmd /k worked as expected,
echo           because 'cmd /k' keeps the command session active.
echo           If you see this, it means 'cmd /k python app.py' exited immediately
echo           and control returned to this script, which is unusual.
pause

GOTO :EOF
