# docker.ps1 - Gerenciador do Ambiente Docker Dev Full-Stack
param(
    [string]$Action = "start"
)

$ContainerName = "docker-dev"
$ImageName = "dev-fullstack:latest"
$HostPort = 2222
$DockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

function Write-Header($msg) {
    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Ok($msg) { Write-Host "[OK] $msg"    -ForegroundColor Green }
function Write-Info($msg) { Write-Host "[..] $msg"    -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERRO] $msg"  -ForegroundColor Red }

# ─── STOP ────────────────────────────────────────────────────────────────────
function Stop-Ambiente {
    Write-Header "Parando Ambiente Docker Dev"

    Write-Info "Removendo container '$ContainerName'..."
    docker rm -f $ContainerName 2>$null
    Write-Ok "Container removido."
}

# ─── DOCKER READY ─────────────────────────────────────────────────────────────
function Wait-Docker {
    # Testa se o daemon já responde
    docker version 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Docker já está rodando."
        return
    }

    Write-Info "Iniciando Docker Desktop..."
    Start-Process $DockerDesktop

    $max = 180
    $elapsed = 0
    while ($elapsed -lt $max) {
        Start-Sleep -Seconds 5
        $elapsed += 5
        docker version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Docker Daemon respondendo após ${elapsed}s."
            return
        }
        Write-Info "Aguardando Docker... (${elapsed}s / ${max}s)"
    }

    Write-Err "Docker não respondeu em ${max}s. Verifique o Docker Desktop."
    exit 1
}

# ─── BUILD ────────────────────────────────────────────────────────────────────
function Build-Imagem {
    Write-Info "Construindo imagem '$ImageName' (pode demorar na 1ª vez)..."
    docker build -t $ImageName .
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Build falhou. Verifique o Dockerfile."
        exit 1
    }
    Write-Ok "Imagem construída com sucesso!"
}

# ─── START ────────────────────────────────────────────────────────────────────
function Start-Ambiente([bool]$WithBuild) {
    Write-Header "Iniciando Ambiente Docker Dev"

    # Remove container antigo sem travar
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

    # Aguarda o SSH subir dentro do container
    Write-Info "Aguardando SSH iniciar no container..."
    Start-Sleep -Seconds 3

    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "  Container '$ContainerName' rodando!" -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Conectar via SSH:" -ForegroundColor White
    Write-Host "    ssh root@localhost -p $HostPort" -ForegroundColor Yellow
    Write-Host "    Senha: senha_forte_aqui" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Entrar direto (sem SSH):" -ForegroundColor White
    Write-Host "    docker exec -it $ContainerName bash" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Ver logs:" -ForegroundColor White
    Write-Host "    docker logs -f $ContainerName" -ForegroundColor Yellow
    Write-Host ""
}

# ─── ROTEADOR ─────────────────────────────────────────────────────────────────
switch ($Action.ToLower()) {
    "start" { Start-Ambiente $false }
    "build" { Start-Ambiente $true }
    "stop" { Stop-Ambiente }
    "restart" {
        Stop-Ambiente
        Write-Info "Aguardando 2 segundos..."
        Start-Sleep -Seconds 2
        Start-Ambiente $false
    }
    default {
        Write-Err "Ação inválida: '$Action'"
        Write-Host ""
        Write-Host "Uso:" -ForegroundColor Cyan
        Write-Host "  .\docker.ps1              # Inicia"
        Write-Host "  .\docker.ps1 start        # Inicia"
        Write-Host "  .\docker.ps1 stop         # Para"
        Write-Host "  .\docker.ps1 restart      # Reinicia"
        Write-Host "  .\docker.ps1 build        # Rebuilda e inicia"
        exit 1
    }
}