@echo off
title Internal Plagiarism Detector

:: Set the current working directory to the location of this batch file
pushd %~dp0

:: Check if Python is installed
where python >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not found in PATH. Please install Python and try again.
    timeout /t 3 >nul
    exit /b 1
)

:: Check if the nltk_data folder exists
if not exist "nltk_data\" (
    echo The 'nltk_data' folder is missing. Please ensure it is placed in the same directory as this script.
    timeout /t 3 >nul
    exit /b 1
)

:: Check if app.py exists
if not exist "app.py" (
    echo The 'app.py' file is missing. Please ensure it is placed in the same directory as this script.
    timeout /t 3 >nul
    exit /b 1
)

:: Run the Python script
echo Starting Internal Plagiarism Detector...
python app.py

:: If execution is successful, restart automatically
if %errorlevel% == 0 (
    echo Application exited successfully. Restarting...
    timeout /t 1 >nul
    goto :start
)

:: If an error occurs, display a message and exit
echo An error occurred while running the application. Please check the logs.
timeout /t 5 >nul
exit /b 1
