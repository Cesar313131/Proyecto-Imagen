@echo off
setlocal
cd /d "%~dp0"
echo =========================================
echo   NEON SMART HOME + YOLO - WINDOWS
 echo =========================================
if exist ".venv_neon\Scripts\python.exe" goto install
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    py -3.12 -m venv .venv_neon
    goto check
)
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    py -3.11 -m venv .venv_neon
    goto check
)
echo ERROR: instala Python 3.12 o 3.11 64-bit con el lanzador py.
set "EXITCODE=1"
goto end
:check
if not exist ".venv_neon\Scripts\python.exe" (
    echo ERROR al crear entorno virtual.
    set "EXITCODE=1"
    goto end
)
:install
".venv_neon\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv_neon\Scripts\python.exe" -m pip install -r requirements_yolo.txt
if errorlevel 1 goto fail
".venv_neon\Scripts\python.exe" -m pip check
if errorlevel 1 goto fail
echo.
echo INSTALACION COMPLETADA.
echo Siguiente: .\DESCARGAR_MODELOS.cmd
set "EXITCODE=0"
goto end
:fail
echo ERROR al instalar dependencias. Revisa Internet y la salida anterior.
set "EXITCODE=1"
:end
pause
exit /b %EXITCODE%
