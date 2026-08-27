@echo off
REM Executa o arquivosplit.jl e mantem a janela aberta ate o usuario fechar
REM (digitando "exit" ou clicando no X da janela).

cd /d "%~dp0"

start "arquivosplit" cmd /k julia "%~dp0arquivosplit.jl" %*
