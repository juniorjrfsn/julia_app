@echo off
chcp 65001 >nul

echo =====================================================
echo     Iniciando Ambiente Docker Dev Full-Stack
echo =====================================================
echo.

:: ====================== LIMPEZA DO CONTAINER ANTERIOR ======================
echo [1/4] Removendo container anterior (se existir)...
docker rm -f docker-dev >nul 2>&1
echo [OK] Pronto.
echo.

:: ====================== VERIFICAR SE DOCKER JÁ ESTÁ RODANDO ======================
echo [2/4] Verificando Docker Daemon...
docker version >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Docker já está rodando!
    goto CHECK_BUILD
)

:: Docker não está rodando — inicia o Docker Desktop
echo Docker não encontrado. Iniciando Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

echo Aguardando Docker Daemon subir (máx. 3 minutos)...
set MAX_WAIT=180
set count=0

:WAIT_DOCKER
docker version >nul 2>&1
if %errorlevel%==0 goto DOCKER_OK

timeout /t 6 /nobreak >nul
set /a count+=6
if %count% geq %MAX_WAIT% goto TIMEOUT

echo   ... aguardando (%count%s de %MAX_WAIT%s)
goto WAIT_DOCKER

:DOCKER_OK
echo [OK] Docker Daemon está rodando!
echo.
goto CHECK_BUILD

:TIMEOUT
echo.
echo [ERRO] Docker não subiu em %MAX_WAIT% segundos.
echo Verifique se o ícone do Docker ficou verde na bandeja do sistema.
pause
exit /b 1

:: ====================== BUILD ======================
:CHECK_BUILD
set BUILD=0
if /i "%1"=="build"    set BUILD=1
if /i "%1"=="BUILD"    set BUILD=1
if /i "%1"=="--build"  set BUILD=1
if /i "%1"=="-b"       set BUILD=1

if %BUILD%==1 (
    echo [3/4] Construindo imagem Docker (pode demorar na primeira vez)...
    docker build -t dev-fullstack:latest .
    if %errorlevel% neq 0 (
        echo [ERRO] Build falhou. Verifique o Dockerfile.
        pause
        exit /b 1
    )
    echo [OK] Build concluído!
    echo.
) else (
    echo [3/4] Pulando build ^(use "docker.bat build" para rebuildar^).
    echo.
)

:: ====================== RODAR CONTAINER ======================
echo [4/4] Iniciando container em background...

:: CORREÇÃO PRINCIPAL:
::   -d  = detached (background) — o container roda sem travar o terminal
::   Sem -it, pois o CMD do container já é o entrypoint.sh que mantém o processo vivo
docker run -d --privileged --name docker-dev ^
    -p 2222:22 ^
    -v /var/run/docker.sock:/var/run/docker.sock ^
    -v "%CD%":/app ^
    dev-fullstack:latest

if %errorlevel% neq 0 (
    echo.
    echo [ERRO] Falha ao iniciar o container.
    echo Dica: rode "docker.bat build" para rebuildar a imagem.
    pause
    exit /b 1
)

echo.
echo =====================================================
echo  Container "docker-dev" rodando em background!
echo =====================================================
echo.
echo  Para conectar via SSH:
echo    ssh root@localhost -p 2222
echo    Senha: _senha_
echo.
echo  Para entrar direto no container (sem SSH):
echo    docker exec -it docker-dev bash
echo.
echo  Para ver logs do container:
echo    docker logs -f docker-dev
echo.
echo =====================================================
echo.
pause