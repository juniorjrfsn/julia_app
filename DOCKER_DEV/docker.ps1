# docker.ps1 - Gerenciador do Ambiente Docker Dev Full-Stack

param(
    [string]$Action = "start"
)

$ContainerName = "docker-dev"
$ImageName = "dev-fullstack:latest"
$HostPort = 2222
$DockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
$DockerContext = "desktop-linux"
$DockerPipe = "\\.\pipe\dockerDesktopLinuxEngine"

function Write-Header($msg) {
    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Ok($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "[..]   $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERRO] $msg" -ForegroundColor Red }

# ==============================================================================
# STOP
# ==============================================================================
function Stop-Ambiente {
    Write-Header "Parando Ambiente Docker Dev"
    Write-Info "Removendo container '$ContainerName'..."
    docker rm -f $ContainerName 2>$null
    Write-Ok "Container removido."
}

# ==============================================================================
# Testa se o pipe do daemon esta disponivel
# ==============================================================================
function Test-DockerPipe {
    return (Test-Path $DockerPipe)
}

# ==============================================================================
# Testa se o daemon responde de verdade (apos o pipe existir)
# ==============================================================================
function Test-DockerReady {
    $result = docker info 2>&1
    return ($LASTEXITCODE -eq 0)
}

# ==============================================================================
# Garante que o contexto correto esta ativo
# ==============================================================================
function Set-DockerContext {
    $current = docker context show 2>$null
    if ($current -ne $DockerContext) {
        Write-Info "Ativando contexto '$DockerContext'..."
        docker context use $DockerContext | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Nao foi possivel ativar o contexto '$DockerContext'."
            Write-Err "Execute manualmente: docker context use $DockerContext"
            exit 1
        }
        Write-Ok "Contexto '$DockerContext' ativo."
    }
    else {
        Write-Ok "Contexto '$DockerContext' ja esta ativo."
    }
}

# ==============================================================================
# Aguarda o Docker Daemon ficar disponivel via pipe
# ==============================================================================
function Wait-Docker {
    Write-Info "Verificando se o Docker ja esta rodando..."

    if (Test-DockerPipe) {
        if (Test-DockerReady) {
            Write-Ok "Docker ja esta rodando."
            Set-DockerContext
            return
        }
        Write-Info "Pipe detectado. Aguardando daemon finalizar inicializacao..."
    }
    else {
        if (-Not (Test-Path $DockerDesktop)) {
            Write-Err "Docker Desktop nao encontrado em: $DockerDesktop"
            Write-Err "Ajuste a variavel DockerDesktop no topo do script."
            exit 1
        }

        $proc = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
        if (-Not $proc) {
            Write-Info "Iniciando Docker Desktop..."
            Start-Process -FilePath $DockerDesktop
            Write-Info "Aguardando processo subir (15s)..."
            Start-Sleep -Seconds 15
        }
        else {
            Write-Info "Docker Desktop ja esta aberto. Aguardando pipe do daemon..."
        }
    }

    # Fase 1: aguarda o pipe aparecer
    $maxPipe = 120
    $elapsedP = 0
    $interval = 5

    Write-Info "Fase 1/2 - Aguardando pipe do daemon (max ${maxPipe}s)..."

    while ($elapsedP -lt $maxPipe) {
        Start-Sleep -Seconds $interval
        $elapsedP += $interval

        if (Test-DockerPipe) {
            Write-Host ""
            Write-Ok "Pipe disponivel apos ${elapsedP}s."
            break
        }

        if ($elapsedP % 25 -eq 0) {
            Write-Info "  aguardando pipe... (${elapsedP}s / ${maxPipe}s)"
        }
        else {
            Write-Host "." -NoNewline -ForegroundColor DarkGray
        }
    }

    if (-Not (Test-DockerPipe)) {
        Write-Host ""
        Write-Err "Pipe do Docker nao apareceu em ${maxPipe}s."
        Write-Err "Verifique:"
        Write-Err "  1. Se o icone do Docker na bandeja ficou verde"
        Write-Err "  2. wsl --status  (no PowerShell)"
        Write-Err "  3. Aba Troubleshoot no Docker Desktop"
        exit 1
    }

    # Fase 2: aguarda o daemon responder
    $maxDaemon = 60
    $elapsedD = 0

    Write-Info "Fase 2/2 - Aguardando daemon responder (max ${maxDaemon}s)..."

    while ($elapsedD -lt $maxDaemon) {
        if (Test-DockerReady) {
            Write-Host ""
            Write-Ok "Docker Daemon pronto!"
            Set-DockerContext
            return
        }

        Start-Sleep -Seconds $interval
        $elapsedD += $interval

        if ($elapsedD % 20 -eq 0) {
            Write-Info "  aguardando daemon... (${elapsedD}s / ${maxDaemon}s)"
        }
        else {
            Write-Host "." -NoNewline -ForegroundColor DarkGray
        }
    }

    Write-Host ""
    Write-Err "Daemon nao respondeu em ${maxDaemon}s apos o pipe estar disponivel."
    Write-Err "Tente: docker context use $DockerContext  e rode o script novamente."
    exit 1
}

# ==============================================================================
# BUILD
# ==============================================================================
function Build-Imagem {
    Write-Info "Construindo imagem '$ImageName' (pode demorar na 1a vez)..."
    docker build -t $ImageName .
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Build falhou. Verifique o Dockerfile."
        exit 1
    }
    Write-Ok "Imagem construida com sucesso!"
}

# ==============================================================================
# START
# ==============================================================================
function Start-Ambiente([bool]$WithBuild) {
    Write-Header "Iniciando Ambiente Docker Dev"

    Write-Info "Removendo container anterior (se existir)..."
    docker rm -f $ContainerName 2>$null | Out-Null
    Write-Ok "Pronto."

    Wait-Docker

    if ($WithBuild) { Build-Imagem }

    Write-Info "Subindo container em background..."
    docker run -d --privileged --name $ContainerName `
        -p "${HostPort}:22" `
        -v /var/run/docker.sock:/var/run/docker.sock `
        -v "${PWD}:/app" `
        $ImageName

    if ($LASTEXITCODE -ne 0) {
        Write-Err "Falha ao iniciar container. Tente: .\docker.ps1 build"
        exit 1
    }

    Write-Info "Aguardando SSH iniciar no container..."
    Start-Sleep -Seconds 3

    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "  Container '$ContainerName' rodando!"                 -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  SSH:  ssh root@localhost -p $HostPort"               -ForegroundColor Yellow
    Write-Host "  Bash: docker exec -it $ContainerName bash"           -ForegroundColor Yellow
    Write-Host "  Logs: docker logs -f $ContainerName"                 -ForegroundColor Yellow
    Write-Host ""
}

# ==============================================================================
# ROTEADOR
# ==============================================================================
$act = $Action.ToLower()

if ($act -eq "start") {
    Start-Ambiente $false
}
elseif ($act -eq "build") {
    Start-Ambiente $true
}
elseif ($act -eq "stop") {
    Stop-Ambiente
}
elseif ($act -eq "restart") {
    Stop-Ambiente
    Write-Info "Aguardando 2 segundos..."
    Start-Sleep -Seconds 2
    Start-Ambiente $false
}
else {
    Write-Err "Acao invalida: '$Action'"
    Write-Host "Uso: .\docker.ps1 [start|stop|restart|build]" -ForegroundColor Cyan
    exit 1
}