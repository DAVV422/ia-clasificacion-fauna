@echo off
title WWM Clasificador de especies - Escritorio
chcp 65001 > nul

echo.
echo  ==========================================
echo   WWM Clasificador de especies  --  Prototipo v1
echo  ==========================================
echo.

REM Definir donde esta el proyecto
set "PROYECTO=%~dp0"
set "LOG=%PROYECTO%app_error.log"

REM Intentar activar el entorno virtual si existe
if exist "%PROYECTO%venv\Scripts\activate.bat" (
    echo  [INFO] Activando entorno virtual venv...
    call "%PROYECTO%venv\Scripts\activate.bat"
    goto :run
)
if exist "%PROYECTO%.venv\Scripts\activate.bat" (
    echo  [INFO] Activando entorno virtual .venv...
    call "%PROYECTO%.venv\Scripts\activate.bat"
    goto :run
)

echo  [AVISO] No se encontro entorno virtual. Usando Python del sistema.
echo.

:run
echo  [INFO] Iniciando aplicacion...
echo  [INFO] Si hay errores, revisar: %LOG%
echo.

REM Lanzar la app suprimiendo FutureWarnings y capturando errores
python -W ignore "%PROYECTO%desktop_app.py" 2>"%LOG%"

REM Si Python termino con error, mostrar el log
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ============================================
    echo   ERROR: La aplicacion termino con errores.
    echo  ============================================
    echo.
    echo  Ultimas lineas del log:
    echo  -----------------------
    type "%LOG%"
    echo.
    echo  El log completo esta en: %LOG%
    echo.
    pause
)
