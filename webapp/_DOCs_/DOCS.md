# 📚 Documentação do Projeto — Julia MVC WebApp

## Visão Geral

Um servidor web **MVC (Model-View-Controller)** escrito em [Julia](https://julialang.org/), usando apenas as bibliotecas `HTTP.jl` e `JSON3.jl`. O projeto demonstra como estruturar uma aplicação web completa — com interface HTML e API REST — sem frameworks de terceiros.

- **Versão:** 0.1.0  
- **Julia:** 1.12.6  
- **Autor:** juniorjrfsn &lt;junior.jrfsn@gmail.com&gt;  
- **Porta padrão:** `9090`

---

## Arquitetura

```
webapp/
├── Project.toml          ← Dependências do projeto
├── Manifest.toml         ← Versões travadas (lock file)
├── README.md
├── _DOCs_/               ← Documentação
│   ├── INSTALL.md        ← Guia de instalação
│   ├── DOCS.md           ← Este arquivo
│   ├── API.md            ← Referência da API REST
│   └── ARCHITECTURE.md   ← Decisões de arquitetura
└── src/
    ├── main.jl           ← Ponto de entrada
    ├── webapp.jl         ← Módulo principal (WebApp)
    ├── model.jl          ← Camada de dados (Model)
    ├── controller.jl     ← Lógica de rotas (Controller)
    ├── server.jl         ← Inicialização do servidor
    └── views/
        ├── layout.jl     ← Motor de templates
        ├── layout.html   ← Template HTML base
        ├── tasks.jl      ← View de tarefas
        └── tasks.html    ← Template parcial de tarefas
```

---

## Camadas MVC

### Model (`model.jl`)

Gerencia os dados das tarefas em memória. **Os dados são perdidos ao reiniciar o servidor** (sem persistência em disco).

#### Estrutura de dados

```julia
mutable struct Task
    id         :: Int
    title      :: String
    done       :: Bool
    created_at :: DateTime
end
```

#### Armazenamento

```julia
const DB      = Dict{Int, Task}()   # Banco em memória
const NEXT_ID = Ref(1)              # Contador auto-incremento
```

#### Funções disponíveis

| Função | Assinatura | Retorno | Descrição |
|--------|-----------|---------|-----------|
| `all_tasks` | `()` | `Vector{Task}` | Retorna todas as tarefas ordenadas por ID |
| `find_task` | `(id::Int)` | `Union{Task, Nothing}` | Busca por ID |
| `create_task` | `(title::String)` | `Task` | Cria e persiste nova tarefa |
| `update_task` | `(id; title, done)` | `Union{Task, Nothing}` | Atualiza campos opcionais |
| `delete_task` | `(id::Int)` | `Bool` | Remove; retorna `true` se encontrou |

---

### View (`views/`)

Sistema de templates simples baseado em substituição de `{{variável}}`.

#### `layout.jl` — Motor de Templates

```julia
# Lê um arquivo HTML da pasta views/
read_template(name::String) → String

# Substitui {{chave}} → valor em um template
render_template(template, replacements::Dict) → String

# Renderiza o layout base com título e corpo
render_layout(title::String, body::String) → String
```

#### `tasks.jl` — View de Tarefas

```julia
# Escapa caracteres HTML para segurança (XSS prevention)
escape_html(s::String) → String

# Gera a página completa de listagem de tarefas
render_tasks_index(tasks::Vector{Task}) → String
```

A view `render_tasks_index` exibe:
- **Stats bar:** total, concluídas, pendentes
- **Formulário** de criação de nova tarefa
- **Lista de itens** com checkbox de toggle e botão de exclusão

---

### Controller (`controller.jl`)

Recebe `HTTP.Request`, executa a lógica e devolve `HTTP.Response`.

#### Funções auxiliares de resposta

```julia
html_response(body; status=200) → HTTP.Response
json_response(data; status=200) → HTTP.Response
redirect(location::String)      → HTTP.Response  # 302
```

#### Parser de formulários

```julia
parse_form(req::HTTP.Request) → Dict{String, String}
```

Faz o URL-decode do corpo `application/x-www-form-urlencoded`.

#### Roteador principal

```julia
route(req::HTTP.Request) → HTTP.Response
```

Tabela de despacho de rotas:

| Método | Padrão | Ação |
|--------|--------|------|
| `GET` | `/` | Redireciona para `/tasks` |
| `GET` | `/tasks` | Renderiza lista HTML |
| `POST` | `/tasks` | Cria tarefa via form, redireciona |
| `POST` | `/tasks/:id/toggle` | Inverte `done`, redireciona |
| `POST` | `/tasks/:id/delete` | Remove, redireciona |
| `GET` | `/api/tasks` | Lista em JSON |
| `GET` | `/api/tasks/:id` | Busca por ID em JSON |
| `POST` | `/api/tasks` | Cria via JSON, retorna 201 |
| `DELETE` | `/api/tasks/:id` | Remove, retorna JSON |
| `*` | `*` | 404 HTML |

---

### Server (`server.jl`)

```julia
# Popula 3 tarefas de demonstração se o banco estiver vazio
seed_demo_tasks!()

# Inicia o servidor HTTP
start(; host="127.0.0.1", port=9090)
```

---

### Módulo principal (`webapp.jl`)

```julia
module WebApp
    using HTTP, JSON3, Dates

    include("model.jl")
    include("views/layout.jl")
    include("views/tasks.jl")
    include("controller.jl")
    include("server.jl")

    export start
end
```

A ordem dos `include` é importante: `model.jl` deve vir antes das views e do controller, pois define a struct `Task`.

---

## Dependências

| Pacote | Versão | Uso no projeto |
|--------|--------|---------------|
| `HTTP.jl` | 2.0.0 | Servidor TCP, parsing de requests, `HTTP.serve` |
| `JSON3.jl` | 1.14.3 | `JSON3.write` (serialização) e `JSON3.read` (parse do body) |
| `Dates` | stdlib | Campo `created_at` na struct `Task` |

### Dependências transitivas relevantes
- `CodecZlib` — compressão HTTP
- `URIs` — `HTTP.URIs.unescapeuri` para form parsing
- `Reseau` — backend de rede do HTTP.jl 2.x

---

## Fluxo de uma Requisição

```
Cliente HTTP
    │
    ▼
HTTP.serve(route, host, port)     ← server.jl
    │
    ▼
route(req::HTTP.Request)          ← controller.jl
    │
    ├─ parse path + method
    │
    ├─ chama função do Model       ← model.jl
    │     (all_tasks, create_task, …)
    │
    ├─ chama View (se HTML)        ← views/tasks.jl + layout.jl
    │     render_tasks_index(tasks)
    │
    └─ retorna HTTP.Response
         (html_response / json_response / redirect)
```

---

## Executar

```bash
# Raiz do repositório
julia --project=webapp webapp/src/main.jl

# Dentro de webapp/
julia --project=. src/main.jl
```

Veja o [Guia de Instalação](./INSTALL.md) para detalhes de setup.

---

## Limitações conhecidas

- **Sem persistência:** dados em memória (`Dict`) são perdidos ao reiniciar
- **Sem autenticação:** todas as rotas são públicas
- **Single-threaded:** `HTTP.serve` processa uma requisição por vez por padrão
- **Sem validação avançada:** apenas verifica campo `title` não vazio
