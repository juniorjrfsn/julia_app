# Docker Dev Full-Stack

## Como usar (PowerShell — recomendado)

```powershell
.\docker.ps1              # Inicia o ambiente
.\docker.ps1 start        # Inicia explicitamente
.\docker.ps1 stop         # Para o container
.\docker.ps1 restart      # Reinicia (stop + start)
.\docker.ps1 build        # Rebuilda a imagem e inicia
```

## Conectar via SSH (após o start)

```powershell
ssh root@localhost -p 2222
# Senha: senha_forte_aqui
```

Ou usando o usuário dev:

```powershell
ssh dev@localhost -p 2222
# Senha: senha_forte_dev
```

Na primeira conexão ele vai perguntar se quer continuar — digite `yes`.

## Alternativas mais práticas (sem SSH)

Entrar direto no container:

```powershell
docker exec -it docker-dev bash
```

Ver logs do container:

```powershell
docker logs -f docker-dev
```

VS Code + Dev Containers (melhor experiência):

- Instale a extensão **Dev Containers** no VS Code
- Command Palette → "Dev Containers: Attach to Running Container" → selecione `docker-dev`

## Resumo das configurações SSH

| Campo  | Valor              |
|--------|--------------------|
| Host   | localhost          |
| Porta  | 2222               |
| Usuário| root **ou** dev    |
| Senha  | definida no Dockerfile |

## Primeira vez (build obrigatório)

```powershell
.\docker.ps1 build
```

> ⚠️ Troque as senhas `senha_forte_aqui` e `senha_forte_dev` no `Dockerfile` antes de buildar.

Passo 3 — Execute o build (obrigatório na primeira vez)
powershellcd D:\AGEPREV-PROJETOS\DOCKER_DEV
.\docker.ps1 build
O build vai demorar alguns minutos na primeira vez (baixa Python, Node, Rust, Java, .NET, Elixir...).

Passo 4 — Acompanhe os logs após subir
powershelldocker logs -f docker-dev
Você deve ver as mensagens do entrypoint.sh. Se aparecer algum erro, cole aqui.

Passo 5 — Teste o SSH
powershell

ssh root@localhost -p 2222

# Senha: senha_forte_aqui

Se pedir yes/no na primeira conexão, digite yes.

Resumo do que estava errado
ProblemaCausaCorreçãoContainer morria instantaneamenteImagem nunca foi buildada
.\docker.ps1 buildSSH recusava conexãosshd rodava sem -D e saíaexec /usr/sbin/sshd -D no entrypointdocker logs vazioContainer morria antes de imprimirCorrigido pelo build correto
