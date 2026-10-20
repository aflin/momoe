@echo off
rem Update Momoe without starting a model: the newest code (git pull, when this folder is a git clone), then what a
rem start would refresh - the engine (compiled again where its source changed), the Python packages, each installed
rem model's settings and draft subset. The model files are not touched. Start a model later with momoe.bat start NAME.
rem Options are passed on to setup.py.
setlocal
title Momoe - update
cd /d "%~dp0"
rem All of it in one block: cmd reads a .bat file while it runs it, and the git pull can change this very file.
(
  if exist ".git" (
    where git >nul 2>nul || (
      echo  This folder is a git clone, but git is not on PATH: install Git for Windows ^(winget install Git.Git^)
      echo  or run "git pull" here yourself, then run UPDATE.bat again.
      pause
      exit /b 1
    )
    echo  Getting the newest Momoe ^(git pull^) ...
    git pull --ff-only
    if errorlevel 1 (
      echo.
      echo  git pull did not succeed ^(the reason is above^): nothing was updated. Files you changed here can stop it:
      echo  "git status" lists them.
      pause
      exit /b 1
    )
  ) else (
    echo  This copy of Momoe was not made with git, so it cannot fetch new files itself. Download the newest one:
    echo    https://github.com/aflin/momoe/archive/refs/heads/main.zip
    echo  unzip it anywhere and run momoe.bat setup in it: it finds the model files this copy uses ^(its data folder
    echo  is recorded in your user settings^) and sets itself up the same way - nothing big is downloaded again.
    echo  Checking this copy's engine and settings meanwhile ...
  )
  call "%~dp0START-HERE.bat" --update %*
  if not errorlevel 1 pause
  exit /b
)
