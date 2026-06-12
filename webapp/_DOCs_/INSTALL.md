# 🛠️ Guia de Instalação — Julia MVC WebApp

## Pré-requisitos

| Requisito | Versão mínima | Link |
|-----------|--------------|------|
| Julia     | 1.10+        | https://julialang.org/downloads/ |
| Git       | qualquer     | https://git-scm.com/ |

---

## 1. Instalar Julia

### Windows
```powershell
# Via winget (recomendado)
winget install julia -s msstore

# Ou baixe o instalador em:
# https://julialang.org/downloads/
```

### Linux / macOS (via juliaup — recomendado)
```bash
curl -fsSL https://install.julialang.org | sh
```

Verifique a instalação:
```bash
julia --version
# julia version 1.12.x
```

---

## 2. Clonar o repositório

```bash
git clone https://github.com/juniorjrfsn/julia_app.git
cd julia_app
```

---

## 3. Ativar o ambiente e instalar dependências

Entre no diretório `webapp` e ative o ambiente do projeto:

```bash
# Inicie o REPL com o ambiente do projeto
julia --project=webapp

# No REPL, pressione ] para entrar no modo Pkg:
(@webapp) pkg> instantiate
```

O comando `instantiate` lê o `Manifest.toml` e instala exatamente as versões travadas:

| Pacote | Versão | Função |
|--------|--------|--------|
| HTTP.jl | 2.0.0 | Servidor HTTP e roteamento |
| JSON3.jl | 1.14.3 | Serialização/deserialização JSON |
| Dates | stdlib | Timestamps das tarefas |

> **Alternativa manual** (sem Manifest.toml):
> ```julia
> # No modo Pkg:
> add HTTP JSON3
> ```

---

## 4. Rodar a aplicação

```bash
# A partir da raiz do repositório:
julia --project=webapp webapp/src/main.jl
```

Ou, dentro da pasta `webapp`:
```bash
julia --project=. src/main.jl
```

Saída esperada no terminal:
```
[ Info: 🚀 Servidor iniciado em http://127.0.0.1:9090
```

Abra o navegador em: **http://127.0.0.1:9090**

---

## 5. Verificar a instalação

```bash
# Listar tarefas via API
curl http://localhost:9090/api/tasks
```

Resposta esperada (JSON com 3 tarefas de demonstração):
```json
[
  {"id":1,"title":"Estudar Julia MVC","done":false,"created_at":"..."},
  {"id":2,"title":"Construir uma API REST","done":false,"created_at":"..."},
  {"id":3,"title":"Deploy no servidor","done":false,"created_at":"..."}
]
```

---

## Solução de Problemas

### `ERROR: LoadError: ArgumentError: Package HTTP not found`
O ambiente não foi ativado corretamente. Certifique-se de usar `--project=webapp` ou `--project=.` ao iniciar o Julia.

### `ERROR: AddressInUse`
A porta 9090 já está em uso. Edite `src/server.jl` e altere o `port`:
```julia
function start(; host = "127.0.0.1", port = 9091)
```

### Primeira inicialização lenta
O Julia precisa compilar os pacotes na primeira vez (precompilação). Execuções subsequentes são mais rápidas.
