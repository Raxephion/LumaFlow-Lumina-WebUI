@echo off
title Lumina Image App Installer

echo Checking for existing virtual environment (venv)...
IF EXIST "venv\Scripts\activate.bat" GOTO VENV_EXISTS
GOTO CREATE_VENV

:VENV_EXISTS
echo Virtual environment (venv) already exists. Skipping creation.
GOTO ACTIVATE_VENV

:CREATE_VENV
echo Creating Python virtual environment (venv)...
python -m venv venv
IF ERRORLEVEL 1 (
    echo Failed to create virtual environment. Please ensure Python is installed and in PATH.
    pause
    GOTO END_SCRIPT
)
echo Virtual environment created.

:ACTIVATE_VENV
echo Activating virtual environment...
call "venv\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    echo Failed to activate virtual environment.
    pause
    GOTO END_SCRIPT
)
echo Virtual environment activated.

echo Upgrading pip...
python -m pip install --upgrade pip
IF ERRORLEVEL 1 (
    echo Failed to upgrade pip. Continuing, but this might affect dependency installation.
    pause
    REM Decide if you want to GOTO END_SCRIPT here or try to continue
)
echo Pip upgraded.

echo.
echo Checking for existing PyTorch with CUDA support...
python -c "import torch; import sys; sys.exit(0) if torch.cuda.is_available() and torch.version.cuda is not None else sys.exit(1)"
IF ERRORLEVEL 1 GOTO INSTALL_PYTORCH
GOTO PYTORCH_EXISTS

:INSTALL_PYTORCH
echo PyTorch with CUDA support not found or not functional.
echo Attempting to install PyTorch with CUDA 12.1 support...
echo This is a common version for many modern NVIDIA GPUs.
echo If this fails, or if you have a different GPU/CUDA setup (e.g., older CUDA, AMD, or CPU only),
echo you may need to install PyTorch manually first by following instructions at:
echo https://pytorch.org/get-started/locally/
echo After manual PyTorch installation, you can re-run this script, and it should detect it.
echo.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
IF ERRORLEVEL 1 (
    echo FAILED to install PyTorch with CUDA 12.1. 
    echo Please try installing PyTorch manually from pytorch.org for your specific system setup.
    pause
    GOTO END_SCRIPT
)
echo PyTorch (CUDA 12.1) installation attempted. Please verify its functionality in the app.
GOTO INSTALL_DEPS

:PYTORCH_EXISTS
echo PyTorch with CUDA support already detected. Skipping PyTorch installation.

:INSTALL_DEPS
echo.
echo Installing other dependencies from requirements.txt...
echo (Ensure torch, torchvision, torchaudio are NOT in requirements.txt)
python -m pip install --upgrade -r requirements.txt
IF ERRORLEVEL 1 (
    echo Failed to install other dependencies from requirements.txt. 
    echo Please check your requirements.txt file and internet connection.
    pause
    GOTO END_SCRIPT
)
echo Other dependencies installed.

echo.
echo --------------------------------------------------------------------
echo Installation complete for Lumina Image App!
echo --------------------------------------------------------------------
echo.
echo To run the app:
echo 1. Ensure this command window (or a new one) has the venv active.
echo    (If you open a new window, navigate here and run: venv\Scripts\activate)
echo 2. Then run: python app.py
echo OR
echo    Simply use the launch.bat script (if available).
echo --------------------------------------------------------------------
echo.
pause
GOTO END_SCRIPT

:END_SCRIPT
echo Script finished.
