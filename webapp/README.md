# Julia MVC Web App

Serviço web simples em Julia seguindo o padrão **Model → View → Controller**, usando apenas `HTTP.jl` e `JSON3.jl`.

## Estrutura (em um único arquivo)



```
julia> ]
(@v1.12) pkg> add HTTP
(@v1.12) pkg> add JSON3 
```

No seu projeto Julia, ative o ambiente e adicione os pacotes:
```bash
# 1. Entre no REPL com o ambiente do projeto ativado
julia --project=.

# 2. Pressione ] para entrar no modo Pkg, depois:
add HTTP JSON3

# 3. Saia do REPL
exit()

# 4. Agora rode o app
julia --project=. webapp/src/main.jl
```

```
# Linux/macOS — via juliaup (recomendado)
curl -fsSL https://install.julialang.org | sh

# Ou baixe direto em: https://julialang.org/downloads/
```


```
# listar tarefas
curl http://localhost:8080/api/tasks

# criar tarefa
curl -X POST http://localhost:8080/api/tasks \
     -H "Content-Type: application/json" \
     -d '{"title":"Minha nova tarefa"}'

# deletar tarefa (id=1)
curl -X DELETE http://localhost:8080/api/tasks/1
```




```
main.jl
├── MODEL       — struct Task + funções CRUD (in-memory dict)
├── VIEW        — templates HTML como strings Julia
├── CONTROLLER  — parse de request, montagem de response
└── ROUTER      — despacha método+path para o controller certo
```

## Dependências

```julia
# No REPL Julia:
using Pkg
Pkg.add(["HTTP", "JSON3"])
```

## Como rodar

```bash
julia webapp/src/main.jl
# → http://localhost:8080
```

## Rotas disponíveis

### Interface Web (HTML)
| Método | Rota | Ação |
|--------|------|------|
| GET | `/tasks` | Lista todas as tarefas |
| POST | `/tasks` | Cria nova tarefa (form) |
| POST | `/tasks/:id/toggle` | Marca como feita/pendente |
| POST | `/tasks/:id/delete` | Remove a tarefa |

### API JSON
| Método | Rota | Ação |
|--------|------|------|
| GET | `/api/tasks` | Lista (JSON) |
| GET | `/api/tasks/:id` | Busca por ID |
| POST | `/api/tasks` | Cria `{"title":"..."}` |
| DELETE | `/api/tasks/:id` | Remove |

## Exemplo API

```bash
# Criar tarefa
curl -X POST http://localhost:8080/api/tasks \
     -H "Content-Type: application/json" \
     -d '{"title":"Aprender Julia"}'

# Listar
curl http://localhost:8080/api/tasks
```

### execução rápida
```
julia .\webapp\src\main.jl
```