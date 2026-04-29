@echo off
chcp 65001 >nul

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

echo =====================================================
echo     Docker Dev Full-Stack - Gerenciador
echo =====================================================
echo.

set ACTION=%1

:: Se nenhum parâmetro for passado, assume "start"
if "%ACTION%"=="" set ACTION=start

:: ====================== START ======================
if /i "%ACTION%"=="start" (
    call start.bat
    goto :EOF
)

:: ====================== STOP ======================
if /i "%ACTION%"=="stop" (
    call stop.bat
    goto :EOF
)

:: ====================== RESTART ======================
if /i "%ACTION%"=="restart" (
    call restart.bat
    goto :EOF
)

:: ====================== BUILD ======================
if /i "%ACTION%"=="build" (
    echo [AÇÃO] Iniciando com build da imagem...
    echo.
    call start.bat build
    goto :EOF
)

:: ====================== HELP ======================
echo [ERRO] Parâmetro inválido: %ACTION%
echo.
echo Uso correto:
echo   docker.bat                  - Inicia o ambiente
echo   docker.bat start            - Inicia o ambiente
echo   docker.bat stop             - Para o ambiente
echo   docker.bat restart          - Reinicia o ambiente
echo   docker.bat build            - Inicia com rebuild da imagem
echo.
pause