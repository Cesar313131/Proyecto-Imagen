@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv_neon\Scripts\python.exe" (
 echo No hay entorno. Ejecuta INSTALAR_WINDOWS.cmd
 pause
 exit /b 1
)
if not exist "modelos\doorway_seg_yolo26n.pt" goto getmodels
if not exist "modelos\yolov8n.pt" goto getmodels
goto run
:getmodels
echo Faltan modelos. Intentando descargarlos...
".venv_neon\Scripts\python.exe" descargar_modelos.py
if errorlevel 1 (
 echo ERROR al descargar modelos. Ejecuta DESCARGAR_MODELOS.cmd
 pause
 exit /b 1
)
:run
".venv_neon\Scripts\python.exe" neon_dashboard.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
