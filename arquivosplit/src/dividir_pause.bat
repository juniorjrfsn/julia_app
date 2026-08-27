@echo off
REM Executa o arquivosplit.jl e so fecha a janela apos o usuario
REM pressionar qualquer tecla.

cd /d "%~dp0"

julia "%~dp0arquivosplit.jl" %*

echo.
echo Processo finalizado.
pause
