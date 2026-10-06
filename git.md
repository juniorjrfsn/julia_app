# Git — Guia Prático para VS Code + GitHub

Este arquivo reúne os procedimentos usados neste projeto para configurar o Git, autenticar no GitHub, fazer commit, sincronizar e enviar alterações (`push`), evitando informar nome e e-mail manualmente quando essas informações puderem ser obtidas da conta do GitHub.

---

## 1. Verificar se o Git está instalado

No terminal do VS Code:

```bash
git --version
```

Se estiver instalado, será exibida a versão do Git.

---

## 2. Verificar a configuração atual do Git

Ver nome e e-mail configurados:

```bash
git config --global user.name
git config --global user.email
```

Ver todas as configurações:

```bash
git config --global --list
```

Ver também as configurações locais do projeto:

```bash
git config --list --show-origin
```

> A configuração `--global` vale para seu usuário. A configuração local do repositório pode sobrescrever a global.

---

## 3. Configurar automaticamente nome e e-mail usando o GitHub

Se o GitHub CLI (`gh`) estiver instalado e autenticado, não é necessário digitar manualmente nome e e-mail.

Primeiro confirme a autenticação:

```bash
gh auth status
```

Depois:

```bash
GH_LOGIN=$(gh api user --jq '.login')
GH_ID=$(gh api user --jq '.id')

git config --global user.name "$GH_LOGIN"
git config --global user.email "${GH_ID}+${GH_LOGIN}@users.noreply.github.com"

echo "Nome : $(git config --global user.name)"
echo "Email: $(git config --global user.email)"
```

O e-mail `ID+USUARIO@users.noreply.github.com` é um formato de e-mail privado/noreply do GitHub e evita colocar seu e-mail pessoal no histórico dos commits.

---

## 4. Autenticar no GitHub

Se ainda não estiver autenticado:

```bash
gh auth login
```

Durante o processo, escolha o GitHub.com e o método solicitado pelo GitHub CLI.

Depois confirme:

```bash
gh auth status
```

Também é possível verificar o usuário:

```bash
gh api user --jq '.login'
```

---

## 5. Conferir o repositório atual

Ver o estado:

```bash
git status
```

Ver o remote:

```bash
git remote -v
```

Ver detalhes do remote `origin`:

```bash
git remote show origin
```

Exemplo esperado:

```text
origin  https://github.com/USUARIO/REPOSITORIO.git (fetch)
origin  https://github.com/USUARIO/REPOSITORIO.git (push)
```

---

## 6. Conferir a branch atual

```bash
git branch --show-current
```

Para listar branches:

```bash
git branch -a
```

Para ver a relação entre branches locais e remotas:

```bash
git branch -vv
```

---

## 7. Fazer um commit

Primeiro confira o que mudou:

```bash
git status
```

Adicionar todos os arquivos alterados:

```bash
git add .
```

Criar o commit:

```bash
git commit -m "Descrição da alteração"
```

Exemplo:

```bash
git commit -m "Atualiza documentação do projeto"
```

> `commit` salva a alteração no histórico LOCAL. Ele não envia automaticamente para o GitHub.

---

## 8. Enviar o commit para o GitHub

Depois do commit:

```bash
git push origin main
```

Se a branch atual estiver configurada para acompanhar o remoto, pode ser suficiente:

```bash
git push
```

Depois confira:

```bash
git status
```

---

## 9. Entender `commit` x `push`

É importante não confundir:

```text
Arquivo alterado
      ↓
git add .
      ↓
git commit
      ↓
Histórico LOCAL
      ↓
git push
      ↓
GitHub
```

### `git commit`

Cria um ponto no histórico do computador.

### `git push`

Envia os commits locais para o GitHub.

Por isso, é possível o VS Code mostrar:

```text
Outgoing Changes
```

mesmo depois de o commit ter sido realizado.

---

## 10. Quando aparecem alterações no GitHub que ainda não estão no computador

Se o VS Code indicar algo parecido com:

```text
Sync Changes 2↓ 1↑
```

isso significa que há commits para baixar (`↓`) e commits locais para enviar (`↑`).

Uma maneira segura de sincronizar:

```bash
git pull --rebase origin main
```

Depois:

```bash
git push origin main
```

O `rebase` coloca seus commits locais depois dos commits que chegaram do GitHub, mantendo o histórico mais organizado.

---

## 11. Sequência recomendada para atualizar o projeto

Antes de começar uma nova alteração:

```bash
git pull --rebase origin main
```

Depois de alterar os arquivos:

```bash
git status
git add .
git commit -m "Descrição da alteração"
git push origin main
```

Fluxo completo:

```bash
git pull --rebase origin main
git status
git add .
git commit -m "Atualização do projeto"
git push origin main
```

---

## 12. Fazer tudo em uma sequência automática

Para uso rápido:

```bash
git add .
git commit -m "Atualização automática" 2>/dev/null || true
git pull --rebase origin main
git push origin main
```

### Atenção

Esse comando é conveniente, mas não substitui a verificação de conflitos.

Se o `pull --rebase` encontrar conflitos, pare e resolva os conflitos antes de continuar.

Não use `git push --force` sem entender exatamente o que será sobrescrito.

---

## 13. Se aparecer "user.name" e "user.email" não configurados

Erro típico no VS Code:

```text
Make sure you configure your "user.name" and "user.email" in git.
```

Se o GitHub CLI já estiver autenticado:

```bash
GH_LOGIN=$(gh api user --jq '.login')
GH_ID=$(gh api user --jq '.id')

git config --global user.name "$GH_LOGIN"
git config --global user.email "${GH_ID}+${GH_LOGIN}@users.noreply.github.com"
```

Depois tente novamente:

```bash
git commit -m "Atualização do projeto"
```

---

## 14. Se o `push` for rejeitado com `non-fast-forward`

Exemplo:

```text
rejected
non-fast-forward
```

Isso normalmente significa que o GitHub possui commits que seu computador ainda não possui.

Faça:

```bash
git pull --rebase origin main
```

Se terminar sem conflitos:

```bash
git push origin main
```

### NÃO fazer imediatamente:

```bash
git push --force
```

O `--force` pode sobrescrever histórico remoto e causar perda de trabalho de outras pessoas.

---

## 15. Se houver conflito durante o `pull --rebase`

Confira:

```bash
git status
```

O Git indicará os arquivos em conflito.

Abra os arquivos no VS Code e resolva os trechos marcados por:

```text
<<<<<<<
=======
>>>>>>>
```

Depois adicione os arquivos resolvidos:

```bash
git add .
```

Continue o rebase:

```bash
git rebase --continue
```

Quando terminar:

```bash
git push origin main
```

Se precisar desistir do rebase:

```bash
git rebase --abort
```

---

## 16. Ver o histórico

Histórico resumido:

```bash
git log --oneline --decorate --graph --all
```

Últimos commits:

```bash
git log --oneline -10
```

---

## 17. Ver diferenças antes do commit

Alterações ainda não adicionadas:

```bash
git diff
```

Alterações já adicionadas ao stage:

```bash
git diff --cached
```

---

## 18. Desfazer `git add`

Se adicionou um arquivo por engano:

```bash
git restore --staged arquivo.ext
```

Para retirar todos os arquivos do stage:

```bash
git restore --staged .
```

Isso não apaga as alterações dos arquivos.

---

## 19. Descartar alterações locais de um arquivo

CUIDADO: isso apaga as alterações não commitadas daquele arquivo.

```bash
git restore arquivo.ext
```

Para todos os arquivos:

```bash
git restore .
```

Use somente quando tiver certeza de que não precisa das alterações.

---

## 20. Ver arquivos que estão sendo ignorados

```bash
git status --ignored
```

O projeto deve ter um `.gitignore` adequado para evitar enviar arquivos temporários, ambientes virtuais, credenciais, caches e outros arquivos que não devem entrar no repositório.

Nunca coloque senhas, tokens ou chaves privadas no Git.

---

## 21. Verificar se o GitHub está funcionando

Autenticação:

```bash
gh auth status
```

Usuário:

```bash
gh api user --jq '.login'
```

Remote:

```bash
git remote -v
```

Branch:

```bash
git branch --show-current
```

Status:

```bash
git status
```

---

## 22. Diagnóstico rápido

Execute:

```bash
echo "=== Git ==="
git --version

echo "=== Usuário Git ==="
git config --global user.name
git config --global user.email

echo "=== GitHub CLI ==="
gh auth status

echo "=== Usuário GitHub ==="
gh api user --jq '.login'

echo "=== Remote ==="
git remote -v

echo "=== Branch ==="
git branch --show-current

echo "=== Status ==="
git status
```

---

## 23. Script para configurar automaticamente o Git

Crie um arquivo chamado:

```text
git-config-auto.sh
```

Conteúdo:

```bash
#!/usr/bin/env bash

set -e

echo "======================================"
echo " Configuração automática do Git"
echo "======================================"

if ! command -v git >/dev/null 2>&1; then
    echo "ERRO: Git não está instalado."
    exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "ERRO: GitHub CLI (gh) não está instalado."
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "GitHub não está autenticado."
    echo "Iniciando login..."
    gh auth login
fi

GH_LOGIN=$(gh api user --jq '.login')
GH_ID=$(gh api user --jq '.id')

if [ -z "$GH_LOGIN" ] || [ -z "$GH_ID" ]; then
    echo "ERRO: não foi possível obter os dados do GitHub."
    exit 1
fi

git config --global user.name "$GH_LOGIN"
git config --global user.email "${GH_ID}+${GH_LOGIN}@users.noreply.github.com"

echo ""
echo "Git configurado:"
echo "Nome : $(git config --global user.name)"
echo "Email: $(git config --global user.email)"
echo "GitHub: $GH_LOGIN"
```

Dar permissão de execução:

```bash
chmod +x git-config-auto.sh
```

Executar:

```bash
./git-config-auto.sh
```

---

## 24. Script para sincronizar, commit e push

Pode ser criado um segundo arquivo:

```text
git-sync.sh
```

Conteúdo:

```bash
#!/usr/bin/env bash

set -e

echo "======================================"
echo " Git Sync"
echo "======================================"

BRANCH=$(git branch --show-current)

if [ -z "$BRANCH" ]; then
    echo "ERRO: não foi possível identificar a branch."
    exit 1
fi

echo "Branch atual: $BRANCH"

echo ""
echo "1. Atualizando do GitHub..."
git pull --rebase origin "$BRANCH"

echo ""
echo "2. Verificando alterações..."
git status

echo ""
echo "3. Adicionando alterações..."
git add .

if git diff --cached --quiet; then
    echo "Nenhuma alteração para commit."
else
    MESSAGE="${1:-Atualização do projeto}"
    echo ""
    echo "4. Criando commit: $MESSAGE"
    git commit -m "$MESSAGE"
fi

echo ""
echo "5. Enviando para o GitHub..."
git push origin "$BRANCH"

echo ""
echo "======================================"
echo " Sincronização concluída"
echo "======================================"

git status
```

Dar permissão:

```bash
chmod +x git-sync.sh
```

Usar:

```bash
./git-sync.sh "Atualiza projeto"
```

---

## 25. Comandos essenciais — resumo

### Ver situação

```bash
git status
```

### Baixar atualizações

```bash
git pull --rebase origin main
```

### Adicionar alterações

```bash
git add .
```

### Criar commit

```bash
git commit -m "Descrição"
```

### Enviar para GitHub

```bash
git push origin main
```

### Ver remote

```bash
git remote -v
```

### Ver branch

```bash
git branch --show-current
```

### Ver histórico

```bash
git log --oneline --graph --all
```

### Ver autenticação do GitHub

```bash
gh auth status
```

### Ver usuário do GitHub

```bash
gh api user --jq '.login'
```

---

## 26. Fluxo recomendado para este projeto

Sempre que terminar uma alteração:

```bash
git status
git add .
git commit -m "Descrição da alteração"
git pull --rebase origin main
git push origin main
```

Ou, antes de começar a trabalhar:

```bash
git pull --rebase origin main
```

E no final:

```bash
git add .
git commit -m "Descrição da alteração"
git push origin main
```

---

## 27. Regra principal

Lembre-se:

```text
COMMIT = salva localmente
PUSH   = envia para o GitHub
PULL   = baixa do GitHub
REBASE = reorganiza seus commits locais sobre as alterações remotas
```

Se o VS Code mostrar:

```text
Outgoing Changes
```

há commits locais que ainda precisam de `push`.

Se mostrar:

```text
Incoming Changes
```

há alterações remotas que precisam ser baixadas.

Se mostrar algo como:

```text
2↓ 1↑
```

há alterações nos dois lados. Nesse caso, prefira:

```bash
git pull --rebase origin main
git push origin main
```

e resolva qualquer conflito antes de tentar forçar o envio.

---

## 28. Segurança

Nunca envie para o GitHub:

- senhas;
- tokens de acesso;
- chaves privadas SSH;
- arquivos `.env` com credenciais;
- credenciais de bancos de dados;
- certificados/chaves privadas;
- arquivos de configuração contendo segredos.

Use `.gitignore` e variáveis de ambiente para informações sensíveis.

Em caso de dúvida sobre um `push` rejeitado ou conflito, verifique o estado com:

```bash
git status
```

antes de usar comandos destrutivos.

---

**Fim do guia.**
