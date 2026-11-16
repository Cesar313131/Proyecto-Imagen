@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv_neon\Scripts\python.exe" (
 echo Instala primero con INSTALAR_WINDOWS.cmd
 pause
 exit /b 1
)
".venv_neon\Scripts\python.exe" descargar_modelos.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo Error al descargar modelos. Revisa tu red.
pause
exit /b %RC%
