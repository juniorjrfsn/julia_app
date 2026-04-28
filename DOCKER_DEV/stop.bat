@echo off
chcp 65001 >nul

echo =====================================================
echo     Parando Ambiente Docker Dev Full-Stack
echo =====================================================
echo.

echo [1/3] Parando e removendo container...
docker rm -f docker-dev >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Container parado e removido com sucesso.
) else (
    echo [INFO] Nenhum container "docker-dev" encontrado em execução.
)

echo.
echo [2/3] Parando Docker Desktop...
taskkill /IM "Docker Desktop.exe" /F >nul 2>&1
taskkill /IM "com.docker.backend.exe" /F >nul 2>&1
taskkill /IM "com.docker.service.exe" /F >nul 2>&1
echo [OK] Docker Desktop finalizado.

echo.
echo [3/3] Desligando WSL...
wsl --shutdown >nul 2>&1
echo [OK] WSL desligado.

echo.
echo =====================================================
echo     Ambiente Docker parado com sucesso!
echo =====================================================
echo.
echo Dica: Para iniciar novamente, execute: start.bat
echo.

pause