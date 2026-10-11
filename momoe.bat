@echo off
rem Momoe (more MoEs, built on Strata): momoe setup ^| start CONFIG ^| stop ^| status ^| list ^| run TOOL   (momoe help)
setlocal
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\python.exe" (
  "%HERE%.venv\Scripts\python.exe" "%HERE%momoe.py" %*
  exit /b %errorlevel%
)
if /i not "%~1"=="setup" (
  echo Momoe is not set up yet: run  momoe setup
  exit /b 1
)
rem the first setup: START-HERE.bat makes .venv (and installs Python if needed); the arguments after "setup"
set "ARGS="
:collect
shift
if "%~1"=="" goto run
set ARGS=%ARGS% "%~1"
goto collect
:run
call "%HERE%START-HERE.bat" --setup --detach %ARGS%
