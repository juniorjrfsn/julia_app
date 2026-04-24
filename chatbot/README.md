# Chatbot Julia

Chatbot desenvolvido em **Julia** com aprendizado de máquina baseado em similaridade de texto.
O bot lê um arquivo TOML contendo pares de perguntas e respostas e utiliza o algoritmo de
**similaridade de Jaccard** para encontrar a resposta mais adequada à entrada do usuário.

---

## Funcionalidades

- Lê perguntas e respostas de um arquivo `.toml`
- Cada resposta pode ter **várias perguntas** associadas
- Normalização de texto (minúsculas, remoção de pontuação)
- Tokenização da entrada do usuário
- Cálculo de **similaridade de Jaccard** entre tokens
- Resposta padrão configurável para entradas não reconhecidas
- Loop interativo no terminal

---

## Estrutura do Projeto

```
chatbot/
├── src/
│   └── chatbot.jl       # Módulo principal do chatbot
├── data/
│   └── respostas.toml   # Base de perguntas e respostas
├── Project.toml          # Dependências do projeto Julia
└── README.md             # Este arquivo
```

---

## Formato do arquivo TOML

O arquivo `data/respostas.toml` define os pares de perguntas e respostas.
Cada bloco `[[pares]]` contém uma resposta e uma lista de perguntas que a disparam:

```toml
[[pares]]
resposta = "Olá! Como posso te ajudar?"
perguntas = ["olá", "oi", "bom dia"]

[[pares]]
resposta = "Tchau! Até mais!"
perguntas = ["tchau", "até logo", "bye"]

# Resposta padrão para entradas não reconhecidas
[[pares]]
resposta = "Não entendi. Pode reformular?"
perguntas = ["__padrao__"]
```

> A pergunta especial `"__padrao__"` define a resposta usada quando nenhuma outra
> pergunta atinge o limiar mínimo de similaridade.

---

## Como usar

### 1. Instalar dependências

```julia
julia> ]          # abre o modo Pkg
pkg> activate .   # ativa o ambiente do projeto
pkg> instantiate  # instala as dependências
```

### 2. Executar o chatbot

```julia
include("src/chatbot.jl")
using .chatbot
chatbot.iniciar("data/respostas.toml")
```

### 3. Conversar

```
==================================================
  Chatbot Julia — Aprendizado de Máquina
==================================================
Carregando modelo de: data/respostas.toml
Modelo carregado com 7 pares de perguntas/respostas.
Digite 'sair' para encerrar.

Você: oi
Bot: Olá! Eu sou um chatbot. Como posso te ajudar hoje?

Você: o que você faz
Bot: Você pode me fazer perguntas e eu tentarei responder com base no meu treinamento!

Você: sair
Bot: Até logo!
```

---

## Como funciona o aprendizado de máquina

1. **Normalização** — A entrada é convertida para minúsculas e a pontuação é removida.
2. **Tokenização** — O texto é dividido em palavras (tokens).
3. **Similaridade de Jaccard** — Para cada pergunta cadastrada, calcula-se:

   ```
   J(A, B) = |A ∩ B| / |A ∪ B|
   ```

   onde `A` são os tokens da entrada e `B` os tokens da pergunta.

4. **Seleção** — A resposta com maior score acima do limiar (padrão: `0.15`) é retornada.
5. **Fallback** — Se nenhuma pergunta atinge o limiar, retorna a resposta padrão.

---

## Dependências

| Pacote | Versão mínima | Descrição                        |
|--------|--------------|----------------------------------|
| TOML   | stdlib       | Leitura do arquivo de dados      |

Julia `>= 1.6` é necessário. O pacote `TOML` já faz parte da biblioteca padrão.

---

## Autor

**juniorjrfsn** — [junior.jrfsn@gmail.com](mailto:junior.jrfsn@gmail.com)
