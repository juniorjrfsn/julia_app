Diagnóstico do problema
O erro got 1 SIGTERM/SIGINTs, forcing shutdown indica que o processo docker está sendo interrompido por um sinal antes de conseguir conectar ao daemon. Isso não é problema de contexto nem de pipe — o pipe existe (Test-Path retorna True), mas algo está matando o processo Docker CLI.
Causas mais prováveis:

Antivírus/EDR interceptando o processo docker.exe e enviando SIGTERM
Docker Desktop em estado inconsistente — o pipe existe mas o daemon não responde corretamente
Conflito entre contextos — o default aponta para npipe:////./pipe/docker_engine que pode estar sendo redirecionado incorretamente

Solução passo a passo
Passo 1 — Forçar contexto desktop-linux e testar diretamente
powershelldocker context use desktop-linux
docker version
Se docker version funcionar aqui, o problema é só no contexto default. Nesse caso, rode o build direto:
powershelldocker build -t dev-fullstack:latest .

Passo 2 — Se ainda falhar, reiniciar Docker Desktop completamente
powershell# Para tudo
taskkill /F /IM "Docker Desktop.exe"
taskkill /F /IM "com.docker.backend.exe"
taskkill /F /IM "com.docker.service.exe"
wsl --shutdown

# Aguarda 10 segundos

Start-Sleep -Seconds 10

# Reinicia Docker Desktop

Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Aguarda inicialização

Start-Sleep -Seconds 30

# Testa

docker context use desktop-linux
docker version

Passo 3 — Corrigir o docker.ps1 para sempre usar desktop-linux
O script atual tenta o contexto default que aponta para docker_engine — pipe que está com problema. Adicione isso no início do docker.ps1, antes de qualquer docker command:
powershell# Forçar contexto correto antes de qualquer operação
$env:DOCKER_HOST = "npipe:////./pipe/dockerDesktopLinuxEngine"
Ou execute via linha de comando por enquanto:
powershell$env:DOCKER_HOST = "npipe:////./pipe/dockerDesktopLinuxEngine"
docker build -t dev-fullstack:latest .

Passo 4 — Se o erro persistir (suspeita de antivírus)
Verifique se o Windows Defender ou outro AV está bloqueando:
powershellGet-WinEvent -LogName "Microsoft-Windows-Windows Defender/Operational" `
  -MaxEvents 20 | Where-Object { $_.Message -like "*docker*" } |
  Select-Object TimeCreated, Message
Se aparecer entradas, adicione exceção para:

C:\Program Files\Docker\
C:\Users\<seu_usuario>\.docker\

Resumo do que está acontecendo
SintomaCausaPipe existe mas daemon não respondeDocker Desktop iniciou mas o engine Linux não terminou de subirSIGTERM/SIGINT forcing shutdownCLI recebe sinal de kill antes de conectarContexto default falha, desktop-linux funcionadocker_engine pipe não está ativo, apenas o dockerDesktopLinuxEngine
Recomendação final: fixe o contexto para desktop-linux permanentemente:
powershelldocker context use desktop-linux
E atualize o docker.ps1 para não tentar mudar para default em nenhum momento.Você disse: PS D:\AGEPREV-PROJETOS\DOCKER_DEV> docker context use desktop-linuxPS D:\AGEPREV-PROJETOS\DOCKER_DEV> docker context use desktop-linux
desktop-linux
Current context is now "desktop-linux"
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> docker version
Client:
 Version:           29.4.1
 API version:       1.54
 Go version:        go1.26.2
 Git commit:        055a478
 Built:             Mon Apr 20 16:35:45 2026
 OS/Arch:           windows/amd64
 Context:           desktop-linux
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> docker build -t dev-fullstack:latest .
ERROR: error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping": got 1 SIGTERM/SIGINTs, forcing shutdown
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> # Para tudo
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> taskkill /F /IM "Docker Desktop.exe"
ÊXITO: o processo "Docker Desktop.exe" com PID 22396 foi finalizado.
ÊXITO: o processo "Docker Desktop.exe" com PID 20536 foi finalizado.
ÊXITO: o processo "Docker Desktop.exe" com PID 21308 foi finalizado.
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> taskkill /F /IM "com.docker.backend.exe"
ÊXITO: o processo "com.docker.backend.exe" com PID 20780 foi finalizado.
ÊXITO: o processo "com.docker.backend.exe" com PID 4620 foi finalizado.
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> taskkill /F /IM "com.docker.service.exe"
ERRO: o processo "com.docker.service.exe" não foi encontrado.
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> wsl --shutdown
PS D:\AGEPREV-PROJETOS\DOCKER_DEV>
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> # Aguarda 10 segundos
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> Start-Sleep -Seconds 10
PS D:\AGEPREV-PROJETOS\DOCKER_DEV>
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> # Reinicia Docker Desktop
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
PS D:\AGEPREV-PROJETOS\DOCKER_DEV>
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> # Aguarda inicialização
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> Start-Sleep -Seconds 30
PS D:\AGEPREV-PROJETOS\DOCKER_DEV>
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> # Testa
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> docker context use desktop-linux
desktop-linux
Current context is now "desktop-linux"
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> docker version
Client:
 Version:           29.4.1
 API version:       1.54
 Go version:        go1.26.2
 Git commit:        055a478
 Built:             Mon Apr 20 16:35:45 2026
 OS/Arch:           windows/amd64
 Context:           desktop-linux
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> $env:DOCKER_HOST = "npipe:////./pipe/dockerDesktopLinuxEngine"
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> wsl
Falha de logon: não foi concedido ao usuário o tipo de logon solicitado neste computador.
Código de erro: Wsl/Service/CreateInstance/CreateVm/HCS/0x80070569
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> $env:DOCKER_HOST = "npipe:////./pipe/dockerDesktopLinuxEngine"
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> docker build -t dev-fullstack:latest .
ERROR: error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping": got 1 SIGTERM/SIGINTs, forcing shutdown
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> Get-WinEvent -LogName "Microsoft-Windows-Windows Defender/Operational" `
>> -MaxEvents 20 | Where-Object { $_.Message -like "*docker*" } |
>> Select-Object TimeCreated, Message
PS D:\AGEPREV-PROJETOS\DOCKER_DEV> docker context use desktop-linux
desktop-linux
Current context is now "desktop-linux"
Warning: DOCKER_HOST environment variable overrides the active context. To use "desktop-linux", either set the global --context flag, or unset DOCKER_HOST environment variable.
PS D:\AGEPREV-PROJETOS\DOCKER_DEV>Mostrar mais15:40Claude respondeu: Problema identificado!Problema identificado!
O erro crítico está aqui:
Wsl/Service/CreateInstance/CreateVm/HCS/0x80070569
Esse código 0x80070569 significa "Logon failure: the user has not been granted the requested logon type" — o serviço do WSL/Hyper-V não tem permissão para criar máquinas virtuais. O Docker Desktop usa WSL2/Hyper-V para rodar o engine Linux, e sem isso o daemon nunca sobe de verdade (pipe existe mas não responde).

Solução — Passo a passo
Passo 1 — Verificar se Hyper-V está habilitado
powershell# Rodar como Administrador
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Se algum aparecer State: Disabled, habilite:
powershellEnable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -NoRestart
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
Restart-Computer

Passo 2 — Corrigir permissão do grupo Hyper-V (causa raiz do 0x80070569)
powershell# Rodar como Administrador

# Adiciona seu usuário ao grupo Hyper-V Administrators

$user = $env:USERNAME
Add-LocalGroupMember -Group "Hyper-V Administrators" -Member $user
Depois reinicie o computador — essa permissão só vale após relogin.

Passo 3 — Se estiver em domínio/ambiente corporativo (AGEPREV)
O erro 0x80070569 em ambientes corporativos geralmente vem de Group Policy bloqueando virtualização. Verifique:
powershell# Rodar como Administrador
gpresult /H C:\Temp\gpo_report.html
Start-Process "C:\Temp\gpo_report.html"
Procure por políticas relacionadas a Hyper-V, VirtualMachine ou Logon rights. Se houver restrição de GPO, será necessário solicitar liberação ao administrador de TI.

Passo 4 — Alternativa: mudar Docker para modo Hyper-V puro (sem WSL2)
Se o WSL2 estiver bloqueado por política, o Docker Desktop pode rodar via Hyper-V diretamente:

Abra Docker Desktop → Settings → General
Desmarque "Use the WSL 2 based engine"
Apply & Restart

Ou via linha de comando, edite o arquivo de configuração:
powershell$config = "$env:APPDATA\Docker\settings-store.json"

# Abre para editar

notepad $config
Mude "wslEngineEnabled": true para "wslEngineEnabled": false.

Passo 5 — Verificar logs detalhados do WSL/HCS
powershell# Ver eventos recentes de erro de virtualização
Get-WinEvent -LogName "Microsoft-Windows-Hyper-V-Worker-Admin" -MaxEvents 10 |
  Select-Object TimeCreated, Message

# Ou no log do Docker Desktop

Get-Content "$env:LOCALAPPDATA\Docker\log\vm\dockerd.log" -Tail 50

Resumo do diagnóstico
O que acontecePor quêPipe existe mas docker build falhaDaemon nunca termina de inicializardocker version mostra só o ClientServer não responde (engine Linux não subiu)wsl falha com 0x80070569Usuário sem permissão para criar VM (Hyper-V/WSL2 bloqueado)SIGTERM no docker buildCLI tenta conectar, timeout, recebe sinal de encerramento
A prioridade agora é resolver o 0x80070569 — todo o resto é consequência disso. Rode o Passo 2 como Administrador e reinicie. Se for ambiente corporativo, o Passo 3 vai revelar se há GPO bloqueando.Você disse: PS C:\Users\njunior> # Rodar como AdministradorPS D:\AGEPREV-PROJETOS\DOCKER_DEV> Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All

FeatureName      : Microsoft-Hyper-V-All
DisplayName      : Hyper-V
Description      : Fornece serviços e ferramentas de gerenciamento para a criação e a execução de máquinas virtuais e sepastedPS C:\Users\njunior> # Rodar como Administrador
PS C:\Users\njunior> Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
FeatureName      : Microsoft-Hyper-V-All
DisplayName      : Hyper-V
Description      : Fornece serviços e ferramentas de gerenciamento para a criação e a execução de máquinas virtuais e
                   seus recursos.
RestartRequired  : Possible
State            : Enabled
CustomProperties :
PS C:\Users\njunior> Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
FeatureName      : VirtualMachinePlatform
DisplayName      : Plataforma de Máquina Virtual
Description      : Habilita o suporte à plataforma para máquinas virtuais
RestartRequired  : Possible
State            : Enabled
CustomProperties :
PS C:\Users\njunior> Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
FeatureName      : Microsoft-Windows-Subsystem-Linux
DisplayName      : Subsistema do Windows para Linux
Description      : Fornece serviços e ambientes para a execução de shells e ferramentas Linux nativos no modo de
                   usuário no Windows.
RestartRequired  : Possible
State            : Enabled
CustomProperties :
                   ServerComponent\Description : Fornece serviços e ambientes para a execução de shells e ferramentas
                   Linux nativos no modo de usuário no Windows.
                   ServerComponent\DisplayName : Subsistema do Windows para Linux
                   ServerComponent\Id : 1033
                   ServerComponent\Type : Feature
                   ServerComponent\UniqueName : Microsoft-Windows-Subsystem-Linux
                   ServerComponent\Deploys\Update\Name : Microsoft-Windows-Subsystem-Linux
PS C:\Users\njunior>o carregamento do docker continua travadoMostrar mais15:48Claude respondeu: Diagnóstico atualizadoDiagnóstico atualizado
Os logs do Hyper-V mostram algo importante — a VM subiu e depois foi desligada em menos de 1 minuto (15:44:51 → 15:45:51). Isso confirma que o engine Linux do Docker Desktop inicia mas cai imediatamente.
Vamos encontrar o log real do problema:
Passo 1 — Localizar os logs do Docker Desktop
powershell# Encontrar todos os logs disponíveis
Get-ChildItem "$env:LOCALAPPDATA\Docker\log" -Recurse -Filter "*.log" |
  Select-Object FullName, LastWriteTime, Length |
  Sort-Object LastWriteTime -Descending
powershell# Também verificar aqui
Get-ChildItem "$env:APPDATA\Docker" -Recurse -Filter "*.log" |
  Select-Object FullName, LastWriteTime |
  Sort-Object LastWriteTime -Descending

Passo 2 — Ver o log principal do Docker Desktop
powershell# Log principal (caminho mais comum)
Get-Content "$env:LOCALAPPDATA\Docker\log\host\com.docker.backend.exe.log" -Tail 80
powershell# Alternativa
Get-Content "$env:APPDATA\Docker\log\vm\dockerd.log" -Tail 80 -ErrorAction SilentlyContinue

Passo 3 — Ver o log do WSL/VM
powershell# Log de boot da VM Linux
Get-ChildItem "$env:LOCALAPPDATA\Docker" -Recurse |
  Where-Object { $_.Name -match "wsl|vm|boot|kernel" } |
  Select-Object FullName, LastWriteTime

Passo 4 — Verificar o settings-store.json que você abriu no notepad
Cole aqui o conteúdo (pode omitir dados pessoais). Especialmente quero ver:

"wslEngineEnabled"
"engineType"
"hyperVEnabled"

powershellGet-Content "$env:APPDATA\Docker\settings-store.json" | ConvertFrom-Json |
  Select-Object wslEngineEnabled, engineType, hyperVEnabled, useVirtualizationFrameworkVirtioFS

Passo 5 — Verificar o erro do WSL com mais detalhes
powershell# Criar pasta Temp se não existir e gerar relatório GPO
New-Item -ItemType Directory -Force -Path C:\Temp
gpresult /H C:\Temp\gpo_report.html /F
Start-Process "C:\Temp\gpo_report.html"
powershell# Ver política de direitos de logon
secedit /export /cfg C:\Temp\secpol.cfg /areas USER_RIGHTS
Select-String "SeCreateGlobalPrivilege|SeBatchLogonRight|SeServiceLogonRight|SeInteractiveLogonRight" C:\Temp\secpol.cfg
Cole os resultados aqui — especialmente o conteúdo do settings-store.json e os logs encontrados no Passo 1.Você disse: os logsPS C:\Users\njunior> Get-ChildItem "$env:LOCALAPPDATA\Docker\log" -Recurse -Filter "*.log" |
>> Select-Object FullName, LastWriteTime, Length |
>> Sort-Object LastWriteTime -Descending

FullName                                                                          LastWriteTime       Lengpastedos logs15:55Claude respondeu: Causa raiz encontrada!Causa raiz encontrada!
Olha o init.log da VM — esse é o log mais importante e ainda não foi lido. Mas antes, já consigo ver dois problemas graves nas políticas:
Problema 1 — SeServiceLogonRight não inclui seu usuário
O serviço do Docker/WSL precisa fazer logon como serviço, mas a política só permite:

S-1-5-83-0 = Virtual Machine group
Um SID específico de domínio

Problema 2 — settings-store.json retornou vazio
Isso indica que o Docker Desktop pode estar usando configurações padrão corrompidas ou o arquivo está em outro lugar.

Execute agora (como Administrador)
Passo 1 — Ler o log de boot da VM (causa raiz real)
powershellGet-Content "$env:LOCALAPPDATA\Docker\log\vm\init.log" -Tail 100
Passo 2 — Ver o settings real
powershell# Verificar onde está o settings
Get-ChildItem "$env:APPDATA\Docker" -Recurse | Select-Object FullName
Get-Content "$env:APPDATA\Docker\settings-store.json"
Passo 3 — Verificar seu SID e se está nas políticas
powershell# Ver seu SID
whoami /user

# Ver todos os grupos do seu usuário

whoami /groups | Select-String "S-1-5"
Passo 4 — Verificar o WSL data folder
powershell# Ver o que tem na pasta WSL do Docker
Get-ChildItem "$env:LOCALAPPDATA\Docker\wsl" -Recurse |
  Select-Object FullName, LastWriteTime, Length
Cole o resultado do init.log — ele vai mostrar exatamente onde a VM trava durante o boot.PS C:\Users\njunior> Get-Content "$env:LOCALAPPDATA\Docker\log\vm\init.log" -Tail 100
{"component":"apiproxy","level":"info","msg":">> GET /networks/8036cb20bc5b901f0e03813f3744fa089541cdab715bac845ee65751172aa992","time":"2026-04-28T20:36:27.873946593Z","user_agent":"DockerDesktopUI"}
{"componentpasted15:56Claude respondeu: Causa raiz encontrada!Causa raiz encontrada!
O settings-store.json revela tudo:
json{
  "UseContainerdSnapshotter": true
  // wslEngineEnabled: AUSENTE = usa WSL2 por padrão
  // engineType: AUSENTE
}
Não há wslEngineEnabled: false nem engineType — o Docker está tentando usar WSL2, mas o WSL falha com 0x80070569 porque o ambiente de domínio (ADMINISTRACAO\njunior) tem restrições de logon que impedem o serviço WSL de criar a VM.
Também confirmei: você está no grupo docker-users ✅, é Administrador ✅ — não é problema de permissão de usuário. É o WSL2 bloqueado por política de domínio.

Solução — Mudar Docker para modo Hyper-V (sem WSL2)
Passo 1 — Fechar Docker Desktop completamente
powershelltaskkill /F /IM "Docker Desktop.exe"
taskkill /F /IM "com.docker.backend.exe"
Start-Sleep -Seconds 3
Passo 2 — Editar o settings para desabilitar WSL2
powershell$settings = "$env:APPDATA\Docker\settings-store.json"
$json = Get-Content $settings | ConvertFrom-Json

# Desabilita WSL2, força Hyper-V

$json | Add-Member -Force -NotePropertyName "wslEngineEnabled" -NotePropertyValue $false
$json | Add-Member -Force -NotePropertyName "engineType" -NotePropertyValue "hyper-v"

$json | ConvertTo-Json -Depth 10 | Set-Content $settings
Confirme o resultado:
powershellGet-Content $settings
Deve aparecer "wslEngineEnabled": false e "engineType": "hyper-v".
Passo 3 — Reiniciar Docker Desktop
powershellStart-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Start-Sleep -Seconds 40
docker version
Passo 4 — Se docker version mostrar Server, fazer o build
powershelldocker context use desktop-linux
docker build -t dev-fullstack:latest .

Por que isso vai funcionar
ModoDepende deStatus no seu ambienteWSL2 (atual)WSL service + logon de VM via domínio❌ Bloqueado por GPO de domínioHyper-V diretoHyper-V habilitado + grupo Administradores✅ Você tem ambos
O Hyper-V está habilitado e você é Administrador local — o modo Hyper-V vai subir o engine Linux sem passar pelo WSL, contornando completamente o 0x80070569.Você disse: PS C:\Users\njunior> taskkill /F /IM "Docker Desktop.PS C:\Users\njunior> taskkill /F /IM "Docker Desktop.exe"
ÊXITO: o processo "Docker Desktop.exe" com PID 5328 foi finalizado.
ÊXITO: o processo "Docker Desktop.exe" com PID 6404 foi finalizado.
ÊXITO: o processo "Docker Desktop.exe" com PID 5204 foi finalizado.
ÊXITO: o processo "Docker Desktop.exe" com PID 18228 foi finalizado.
PS C:\Users\njunior> taskkill /F /IM "com.docker.backend.exe"
ÊXITO: o processo "com.docker.backend.exe" com PID 7104 foi finalizado.
ÊXITO: o processo "com.docker.backend.exe" com PID 5600 foi finalizado.
PS C:\Users\njunior> Start-Sleep -Seconds 3
PS C:\Users\njunior> $settings = "$env:APPDATA\Docker\settings-store.json"
PS C:\Users\njunior> $json = Get-Content $settings | ConvertFrom-Json
PS C:\Users\njunior>
PS C:\Users\njunior> # Desabilita WSL2, força Hyper-V
PS C:\Users\njunior> $json | Add-Member -Force -NotePropertyName "wslEngineEnabled" -NotePropertyValue $false
PS C:\Users\njunior> $json | Add-Member -Force -NotePropertyName "engineType" -NotePropertyValue "hyper-v"
PS C:\Users\njunior>
PS C:\Users\njunior> $json | ConvertTo-Json -Depth 10 | Set-Content $settings
PS C:\Users\njunior> Get-Content $settings
{
    "AutoStart":  false,
    "DisplayedOnboarding":  true,
    "EnableDockerAI":  true,
    "LastContainerdSnapshotterEnable":  1777474834,
    "LicenseTermsVersion":  2,
    "SettingsVersion":  43,
    "UseContainerdSnapshotter":  true,
    "wslEngineEnabled":  false,
    "engineType":  "hyper-v"
}
PS C:\Users\njunior> Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
PS C:\Users\njunior> Start-Sleep -Seconds 40
PS C:\Users\njunior> docker version
Client:
 Version:           29.4.1
 API version:       1.54
 Go version:        go1.26.2
 Git commit:        055a478
 Built:             Mon Apr 20 16:35:45 2026
 OS/Arch:           windows/amd64
 Context:           desktop-linux
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: O sistema não pode encontrar o arquivo especificado.
PS C:\Users\njunior> docker context use desktop-linux
desktop-linux
Current context is now "desktop-linux"
PS C:\Users\njunior> docker build -t dev-fullstack:latest .
ERROR: failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: O sistema não pode encontrar o arquivo especificado.
PS C:\Users\njunior>Mostrar mais15:58Claude respondeu: O erro mudou!O erro mudou! Agora é "arquivo não encontrado" — o pipe dockerDesktopLinuxEngine não existe mais porque mudamos para Hyper-V, que usa um pipe diferente. O Docker ainda não terminou de subir ou precisa de mais tempo.

Passo 1 — Aguardar mais e verificar qual pipe está disponível
powershell# Aguardar mais 30s
Start-Sleep -Seconds 30

# Ver quais pipes do Docker existem agora

Get-ChildItem "\\.\pipe\" | Where-Object { $_.Name -like "*docker*" } | Select-Object Name

Passo 2 — Usar o contexto correto para Hyper-V
Com Hyper-V, o Docker geralmente usa o pipe docker_engine (contexto default):
powershelldocker context use default
docker version
Se o Server aparecer, faça o build:
powershelldocker build -t dev-fullstack:latest .

Passo 3 — Se ainda não conectar, ver o log atual
powershellGet-Content "$env:LOCALAPPDATA\Docker\log\host\com.docker.backend.exe.log" -Tail 30
O Docker Desktop com Hyper-V pode demorar até 2-3 minutos para inicializar a VM — é mais lento que WSL2. Rode os comandos acima e cole o resultado.Você disse: deu isso aíPS C:\Users\njunior> Start-Sleep -Seconds 30
PS C:\Users\njunior>
PS C:\Users\njunior> # Ver quais pipes do Docker existem agora
PS C:\Users\njunior> Get-ChildItem "\\.\pipe\" | Where-Object { $_.Name -like "*docker*" } | Select-Object Name

Name
----

dockerBackendV2

PS C:\Users\njunior>pasteddeu isso aí16:00Claude respondeu: Problema claro agora!Problema claro agora!
O log mostra que o Docker Desktop se fechou sozinho antes de terminar de inicializar:
terminating process 2920 (Docker Desktop.exe)
kill frontend processes: Acesso negado.     ← tentou matar o processo mas falhou
shutting down vital services                ← encerrou mesmo assim
closing \\.\pipe\dockerDesktopLinuxEngine   ← fechou os pipes todos
O Docker Desktop está sendo forçado a fechar por algo externo — muito provavelmente uma GPO de domínio que mata o processo, ou o próprio Docker detectando que não consegue subir o engine e se encerrando.

Investigação definitiva
Passo 1 — Ver o log do Docker Desktop (electron) que mostra o motivo do shutdown
powershellGet-Content "$env:LOCALAPPDATA\Docker\log\host\docker-desktop.exe.log" -Tail 50
Passo 2 — Ver se há erro de Hyper-V ao tentar criar a VM
powershellGet-WinEvent -LogName "Microsoft-Windows-Hyper-V-Worker-Admin" -MaxEvents 5 |
  Select-Object TimeCreated, Message | Format-List
powershellGet-WinEvent -LogName "Microsoft-Windows-Hyper-V-VMMS-Admin" -MaxEvents 10 |
  Select-Object TimeCreated, Message | Format-List
Passo 3 — Tentar subir o Docker e monitorar o log em tempo real
Abra dois terminais PowerShell:

Terminal 1 — monitorar log:
powershell

```
Get-Content "$env:LOCALAPPDATA\Docker\log\host\com.docker.backend.exe.log" -Wait -Tail 5
```

Terminal 2 — subir o Docker:
powershellStart-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Observe o Terminal 1 e cole aqui o que aparecer nos primeiros 60 segundos — especialmente qualquer linha com [E], error, failed, HCS, ou shutdown.Você disse: está dando erro
PS C:\Users\njunior> Get-Content "$env:LOCALAPPDATA\Docker\log\host\docker-desktop.exe.log" -Tail 50
[2026-04-29T19:26:49.925545000Z][docker-desktop.exe.ipc] a1462323-docker-desktop-cli -> \\.\pipe\dockerBackendApiServer BackendAPI
[2026-04-29T19:26:49.928792500Z][docker-desktop.exe.ipc] 1762f87pastedestá dando erro16:03Claude respondeu: Causa raiz definitivamente encontrada!Causa raiz definitivamente encontrada!
A linha que mata tudo, repetida duas vezes:
status code not OK but 500: Unhandled exception:
O acesso ao caminho 'C:\Windows\System32\drivers\etc\hosts' foi negado.
O Docker Desktop tenta atualizar o arquivo hosts ao iniciar o engine, e não tem permissão para escrever nesse arquivo. Isso é bloqueio de GPO corporativa — comum em ambientes de domínio como AGEPREV.

Solução imediata — Dar permissão ao arquivo hosts
Execute como Administrador:
powershell# Ver permissões atuais
icacls "C:\Windows\System32\drivers\etc\hosts"

# Dar permissão de escrita ao seu usuário

icacls "C:\Windows\System32\drivers\etc\hosts" /grant "ADMINISTRACAO\njunior:(M)"

# Ou para o grupo Administradores

icacls "C:\Windows\System32\drivers\etc\hosts" /grant "BUILTIN\Administradores:(M)"
Depois reinicie o Docker Desktop:
powershelltaskkill /F /IM "Docker Desktop.exe"
taskkill /F /IM "com.docker.backend.exe"
Start-Sleep -Seconds 5
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Start-Sleep -Seconds 40
docker version

Solução alternativa — Desabilitar a atualização do hosts no Docker
Se a permissão for bloqueada por GPO e não puder ser alterada, desabilite essa função no Docker. Edite o settings-store.json:
powershell$settings = "$env:APPDATA\Docker\settings-store.json"
$json = Get-Content $settings | ConvertFrom-Json
$json | Add-Member -Force -NotePropertyName "updateHostsFile" -NotePropertyValue $false
$json | ConvertTo-Json -Depth 10 | Set-Content $settings

Aqui está o diagnóstico preciso do que está acontecendo com o seu ambiente Docker Dev Full-Stack e as correções prioritárias.
Causa raiz atual (confirmada pelos logs)
O Docker Desktop em modo Hyper-V (que você configurou para contornar o erro WSL 0x80070569) falha ao iniciar o engine Linux com esta mensagem exata:
starting engine: engine linux/hyperv failed to start: refreshing DNS hosts: status code not OK but 500: Unhandled exception: O acesso ao caminho 'C:\Windows\System32\drivers\etc\hosts' foi negado.
Isso é um problema clássico em ambientes corporativos no domínio (como o da AGEPREV). O Docker Desktop tenta atualizar o arquivo hosts do Windows (adicionando entradas como host.docker.internal e gateway.docker.internal) durante a inicialização do engine, mas não tem permissão de escrita devido a restrições de GPO ou permissões do arquivo.
O daemon fica preso em loop de "still waiting for the engine to respond to _ping" (HTTP 503), o que explica por que o script docker.ps1 e os .bat dão timeout.
Soluções (na ordem recomendada)

1. Dar permissão explícita no arquivo hosts (solução mais direta)
Abra o PowerShell como Administrador e execute:
PowerShell# Ver permissões atuais
icacls "C:\Windows\System32\drivers\etc\hosts"

# Dar permissão de modificação (M) para seu usuário

icacls "C:\Windows\System32\drivers\etc\hosts" /grant "ADMINISTRACAO\njunior:(M)"

# Ou para o grupo Administradores (mais amplo)

icacls "C:\Windows\System32\drivers\etc\hosts" /grant "BUILTIN\Administradores:(M)"
Depois feche completamente o Docker Desktop e teste:
PowerShelltaskkill /F /IM "Docker Desktop.exe" 2>$null
taskkill /F /IM "com.docker.backend.exe" 2>$null
Start-Sleep -Seconds 5

Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Start-Sleep -Seconds 40

docker version
Se o docker version mostrar o Server (não só o Client), rode o build:
PowerShelldocker context use desktop-linux
docker build -t dev-fullstack:latest .
2. Desabilitar a atualização do arquivo hosts (se a permissão acima for bloqueada por GPO)
No PowerShell (pode ser como usuário normal):
PowerShell$settings = "$env:APPDATA\Docker\settings-store.json"
$json = Get-Content $settings -Raw | ConvertFrom-Json

# Força desabilitação da atualização do hosts

$json | Add-Member -Force -NotePropertyName "updateHostsFile" -NotePropertyValue $false -ErrorAction SilentlyContinue

$json | ConvertTo-Json -Depth 10 | Set-Content $settings

Write-Host "Configuração atualizada. Reinicie o Docker Desktop."
Depois reinicie o Docker Desktop e teste novamente.
3. Melhorias no seu docker.ps1 (recomendado)
Adicione no início da função Wait-Docker (logo após Write-Header):
PowerShell# Forçar contexto correto para Hyper-V
docker context use default 2>$null | Out-Null
$env:DOCKER_HOST = "npipe:////./pipe/docker_engine"
E aumente um pouco os timeouts se necessário (o Hyper-V é mais lento que WSL2).
Próximos passos depois de o Docker subir
Uma vez que o docker build funcione:

Corrija as senhas no Dockerfile (elas estão expostas):
Mude @Dirgin@202604 para senhas fortes.
Troque também no entrypoint.sh e nos .bat.

Rode:PowerShell.\docker.ps1 build
Conecte via:
docker exec -it docker-dev bash (mais simples)
ou ssh root@localhost -p 2222

Dicas extras para ambiente corporativo (domínio AGEPREV)

Se as permissões no hosts continuarem bloqueadas, a solução mais estável costuma ser voltar para WSL2 resolvendo o erro 0x80070569 (adicionar usuário ao grupo Hyper-V Administrators + possível liberação de GPO com o TI).
Evite rodar Docker Desktop como Administrador o tempo todo — isso causa conflitos de privilégios.

Me cole o resultado dos comandos da Solução 1 (icacls + docker version) depois de executar. Se ainda der erro, cole também a saída de:
PowerShellGet-Content "$env:LOCALAPPDATA\Docker\log\host\com.docker.backend.exe.log" -Tail 30

✅ Entendi. O problema ainda persiste no mesmo ponto: o Docker Desktop (modo Hyper-V) fica preso esperando o engine Linux subir, e o erro principal continua sendo a falta de permissão para escrever no arquivo C:\Windows\System32\drivers\etc\hosts.
O log que você enviou mostra o backend aguardando por mais de 6 minutos (still waiting for the engine... HTTP 500), e você cancelou porque não avançava.
Passo a passo urgente (faça exatamente nesta ordem)

1. Rode como Administrador (PowerShell como Admin):
PowerShell# Dar permissão explícita no arquivo hosts
icacls "C:\Windows\System32\drivers\etc\hosts" /grant "ADMINISTRACAO\njunior:(M)" /T

# Ou, mais amplo (grupo Administradores)

icacls "C:\Windows\System32\drivers\etc\hosts" /grant "BUILTIN\Administradores:(F)" /T

# Verificar o resultado

icacls "C:\Windows\System32\drivers\etc\hosts"
2. Forçar a configuração updateHostsFile: false novamente (para garantir):
PowerShell$settings = "$env:APPDATA\Docker\settings-store.json"
$json = Get-Content $settings | ConvertFrom-Json

$json | Add-Member -Force -NotePropertyName "updateHostsFile" -NotePropertyValue $false

$json | ConvertTo-Json -Depth 10 | Set-Content $settings

# Verificar

Get-Content $settings | Select-String -Pattern "updateHostsFile|wslEngineEnabled|engineType"
3. Reiniciar tudo completamente:
PowerShell# Parar tudo
taskkill /F /IM "Docker Desktop.exe" 2>$null
taskkill /F /IM "com.docker.backend.exe" 2>$null
wsl --shutdown 2>$null

Start-Sleep -Seconds 10

# Iniciar Docker Desktop

Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Aguardar mais tempo (Hyper-V é lento)

Start-Sleep -Seconds 60

# Testar

docker version
Se docker version mostrar a seção Server, então rode o build:
PowerShelldocker context use default
docker build -t dev-fullstack:latest .
Se ainda falhar após o passo 1 e 2
Tente estas alternativas:
A. Rodar Docker Desktop como Administrador (temporariamente):

Clique direito no atalho do Docker Desktop → "Executar como administrador"

B. Voltar para WSL2 (pode ser mais estável no longo prazo)
Como o Hyper-V está dando muito trabalho, vamos tentar voltar:

Edite o settings-store.json novamente:

PowerShell$json = Get-Content "$env:APPDATA\Docker\settings-store.json" | ConvertFrom-Json
$json.wslEngineEnabled = $true
$json | ConvertTo-Json -Depth 10 | Set-Content "$env:APPDATA\Docker\settings-store.json"

Reinicie o Docker Desktop e teste.

Se der o erro 0x80070569 novamente, precisamos resolver a permissão de logon do serviço WSL/Hyper-V (adicionar usuário ao grupo Hyper-V Administrators + possível abertura de GPO com o TI da AGEPREV).
Me envie o resultado dos comandos acima, especialmente a saída do icacls e do docker version depois de tentar.
Quer que eu prepare uma versão atualizada do seu docker.ps1 com mais robustez para esse ambiente corporativo? Posso fazer isso também.

Ótimo progresso! As permissões no arquivo hosts foram aplicadas corretamente (você viu (F) para Administradores e (M) para seu usuário). O updateHostsFile também está definido como false.
No entanto, o Docker ainda não consegue subir o engine (Hyper-V). O erro de pipe (O sistema não pode encontrar o arquivo especificado) indica que o backend do Docker Desktop não terminou de inicializar a VM Hyper-V.
Próximos passos (faça na ordem):

1. Reinicie o Docker Desktop como Administrador (muito importante em ambiente corporativo)

Feche completamente o Docker (use o taskkill novamente se precisar).
Clique com botão direito no ícone do Docker Desktop no menu Iniciar → "Executar como administrador".
Aguarde pelo menos 2 minutos (Hyper-V é mais lento).
Abra outro PowerShell (normal) e teste:

PowerShelldocker version
Se aparecer a seção Server, ótimo! Caso contrário, cole aqui a saída completa.
2. Limpeza forte das configurações do Docker (recomendado agora)
Feche o Docker Desktop completamente e rode como Administrador:
PowerShell# Parar tudo
taskkill /F /IM "Docker Desktop.exe" 2>$null
taskkill /F /IM "com.docker.backend.exe" 2>$null
wsl --shutdown 2>$null

# Remover configurações problemáticas

Remove-Item -Path "$env:APPDATA\Docker" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$env:LOCALAPPDATA\Docker" -Recurse -Force -ErrorAction SilentlyContinue

# Reiniciar Docker Desktop como Administrador (faça manualmente)

Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -Verb RunAs
Aguarde 90–120 segundos e teste docker version novamente.
3. Se ainda não funcionar → Voltar para WSL2 de forma limpa
Como o Hyper-V está dando muito trabalho (comum em domínio corporativo), vamos tentar forçar WSL2 novamente:
PowerShell# Definir WSL2 como engine
$settings = "$env:APPDATA\Docker\settings-store.json"
$json = Get-Content $settings | ConvertFrom-Json
$json.wslEngineEnabled = $true
$json | Add-Member -Force -NotePropertyName "engineType" -NotePropertyValue "wsl2" -ErrorAction SilentlyContinue
$json | ConvertTo-Json -Depth 10 | Set-Content $settings

# Reiniciar Docker Desktop (como Admin)

Depois teste docker version.
4. Verificar status do Hyper-V (importante)
Rode como Administrador:
PowerShellGet-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
bcdedit /enum | findstr hypervisorlaunchtype
Cole a saída aqui.
Dica importante para o seu Dockerfile
Enquanto isso, troque as senhas no Dockerfile (elas estão fracas e expostas):
dockerfile# No final do Dockerfile, mude estas linhas:
echo 'root:SuaSenhaForteAqui2026!' | chpasswd
echo 'dev:SuaSenhaForteAqui2026!' | chpasswd
E atualize também no entrypoint.sh a mensagem de senha.

Me envie agora:

Se você rodou o Docker como Administrador → resultado de docker version
Saída do comando bcdedit acima
Se preferir, posso preparar uma versão mais robusta do docker.ps1 com espera maior e forçar contexto.

Qual caminho você quer seguir primeiro? (Rodar como Admin + limpeza, ou forçar WSL2?)

✅ Entendi o status atual.
Você fez uma limpeza boa das pastas do Docker, mas o engine ainda não está subindo (pipe não encontrado). Além disso, o arquivo settings-store.json foi recriado sem a propriedade wslEngineEnabled (por isso deu erro ao tentar setar diretamente).
Passos para tentar agora (faça na ordem):

1. Corrija o hypervisor (recomendado antes de qualquer coisa)
Como o hypervisorlaunchtype está Auto, mas em ambientes corporativos isso às vezes não pega corretamente, rode como Administrador:
PowerShellbcdedit /set hypervisorlaunchtype auto
Depois reinicie o computador completamente.
Após reiniciar, abra o Docker Desktop como Administrador (clique direito → Executar como administrador) e aguarde 2 minutos completos.
Teste:
PowerShelldocker version
2. Se ainda não funcionar após o reboot, force WSL2 novamente (corrija o JSON)
Após o Docker Desktop estar aberto (mesmo que o engine não suba), rode:
PowerShell# Recriar configurações mínimas para WSL2
$settingsPath = "$env:APPDATA\Docker\settings-store.json"
$json = Get-Content $settingsPath | ConvertFrom-Json

# Adicionar as propriedades corretamente

$json | Add-Member -Force -NotePropertyName "wslEngineEnabled" -NotePropertyValue $true
$json | Add-Member -Force -NotePropertyName "engineType" -NotePropertyValue "wsl2" -ErrorAction SilentlyContinue
$json | Add-Member -Force -NotePropertyName "updateHostsFile" -NotePropertyValue $false

$json | ConvertTo-Json -Depth 10 | Set-Content $settingsPath

Write-Host "Configurações atualizadas. Reinicie o Docker Desktop."
Depois feche e abra novamente o Docker Desktop como Administrador e teste docker version.
3. Verifique o status do WSL
PowerShellwsl --status
wsl -l -v
Se não mostrar nada ou der erro, rode:
PowerShellwsl --install
4. Sobre as senhas no Dockerfile
Os comandos que você tentou (chpasswd) são para dentro do container Linux, não no PowerShell do Windows.
Para corrigir as senhas, edite o Dockerfile (no VS Code ou vim) e mude as linhas:
dockerfile# ==================== SSH SERVER ====================
...
echo 'root:SuaSenhaForteAqui2026!' | chpasswd
echo 'dev:SuaSenhaForteAqui2026!' | chpasswd
E também no final, na criação do usuário dev.
Depois rode o build quando o Docker estiver funcionando.

Qual é o próximo passo que você quer tentar primeiro?

A) Rodar o bcdedit + reiniciar o PC agora
B) Forçar o JSON para WSL2 sem reiniciar ainda

Cole a saída dos comandos que você rodar (especialmente docker version após tentar).
Se nada disso resolver, a próxima etapa mais agressiva seria desinstalar completamente o Docker Desktop, remover pastas residuais e reinstalar (com WSL2 recomendado).
