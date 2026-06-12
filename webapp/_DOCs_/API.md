# 🔌 Referência da API REST — Julia MVC WebApp

**Base URL:** `http://127.0.0.1:9090`  
**Formato:** `application/json`

---

## Endpoints

### `GET /api/tasks`
Lista todas as tarefas ordenadas por ID.

**Request:**
```http
GET /api/tasks HTTP/1.1
Host: 127.0.0.1:9090
```

**Response `200 OK`:**
```json
[
  {
    "id": 1,
    "title": "Estudar Julia MVC",
    "done": false,
    "created_at": "2026-05-29T09:00:00"
  },
  {
    "id": 2,
    "title": "Construir uma API REST",
    "done": true,
    "created_at": "2026-05-29T09:01:00"
  }
]
```

**Exemplo curl:**
```bash
curl http://localhost:9090/api/tasks
```

---

### `GET /api/tasks/:id`
Busca uma tarefa pelo ID.

**Parâmetros de rota:**
| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `id` | `Int` | ID da tarefa |

**Response `200 OK`:**
```json
{
  "id": 1,
  "title": "Estudar Julia MVC",
  "done": false,
  "created_at": "2026-05-29T09:00:00"
}
```

**Response `404 Not Found`:**
```json
{ "error": "not found" }
```

**Exemplo curl:**
```bash
curl http://localhost:9090/api/tasks/1
```

---

### `POST /api/tasks`
Cria uma nova tarefa.

**Request Body:**
```json
{ "title": "Minha nova tarefa" }
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `title` | `String` | ✅ | Título da tarefa (não pode ser vazio) |

**Response `201 Created`:**
```json
{
  "id": 4,
  "title": "Minha nova tarefa",
  "done": false,
  "created_at": "2026-05-29T10:00:00"
}
```

**Response `400 Bad Request`** (título vazio):
```json
{ "error": "title required" }
```

**Exemplo curl:**
```bash
curl -X POST http://localhost:9090/api/tasks \
     -H "Content-Type: application/json" \
     -d '{"title":"Aprender Julia"}'
```

---

### `DELETE /api/tasks/:id`
Remove uma tarefa pelo ID.

**Parâmetros de rota:**
| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `id` | `Int` | ID da tarefa a remover |

**Response `200 OK`:**
```json
{ "ok": true }
```

**Response `404 Not Found`:**
```json
{ "error": "not found" }
```

**Exemplo curl:**
```bash
curl -X DELETE http://localhost:9090/api/tasks/1
```

---

## Rotas da Interface Web (HTML)

> Estas rotas retornam HTML e são projetadas para uso pelo navegador via formulários.

| Método | Rota | Ação |
|--------|------|------|
| `GET` | `/` | Redireciona para `/tasks` (302) |
| `GET` | `/tasks` | Exibe a lista de tarefas |
| `POST` | `/tasks` | Cria tarefa via `<form>`, redireciona para `/tasks` |
| `POST` | `/tasks/:id/toggle` | Inverte o campo `done`, redireciona para `/tasks` |
| `POST` | `/tasks/:id/delete` | Remove a tarefa, redireciona para `/tasks` |

**Corpo do form (POST /tasks):**
```
Content-Type: application/x-www-form-urlencoded

title=Minha+tarefa
```

---

## Resumo de Status Codes

| Código | Significado | Quando ocorre |
|--------|-------------|---------------|
| `200` | OK | Leitura bem-sucedida |
| `201` | Created | Tarefa criada via API |
| `302` | Found | Redirecionamento após ação HTML |
| `400` | Bad Request | `title` ausente ou vazio |
| `404` | Not Found | ID não existe ou rota inválida |

---

## Exemplos completos (fluxo CRUD)

```bash
# 1. Criar duas tarefas
curl -X POST http://localhost:9090/api/tasks \
     -H "Content-Type: application/json" \
     -d '{"title":"Comprar leite"}'

curl -X POST http://localhost:9090/api/tasks \
     -H "Content-Type: application/json" \
     -d '{"title":"Estudar Julia"}'

# 2. Listar todas
curl http://localhost:9090/api/tasks

# 3. Buscar a tarefa de ID 4
curl http://localhost:9090/api/tasks/4

# 4. Deletar a tarefa de ID 4
curl -X DELETE http://localhost:9090/api/tasks/4

# 5. Confirmar exclusão (deve retornar lista sem ID 4)
curl http://localhost:9090/api/tasks
```
