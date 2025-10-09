#!/bin/bash
# QUICK FIX - Correção rápida do problema de reconhecimento
# Execute: bash QUICK_FIX.sh

set -e

echo "⚡ CORREÇÃO RÁPIDA DO RECONHECIMENTO"
echo "======================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Show the problem
echo -e "${RED}❌ PROBLEMA DETECTADO:${NC}"
echo "   O modelo identifica objeto-3.jpeg como 'junior'"
echo "   Mas deveria identificar como 'cachorro'"
echo ""

# Step 2: Ask user to verify the test image
echo -e "${YELLOW}🔍 VERIFICAÇÃO OBRIGATÓRIA:${NC}"
echo ""
echo "   Vamos abrir a imagem objeto-3.jpeg para você verificar"
echo "   PERGUNTA: Esta imagem é de um cachorro ou de uma pessoa?"
echo ""
read -p "Pressione ENTER para abrir a imagem..." 

if [ -f "../../../dados/fotos_auth/objeto-3.jpeg" ]; then
    xdg-open ../../../dados/fotos_auth/objeto-3.jpeg 2>/dev/null || \
    open ../../../dados/fotos_auth/objeto-3.jpeg 2>/dev/null || \
    echo "   (Abra manualmente: ../../../dados/fotos_auth/objeto-3.jpeg)"
fi

echo ""
echo "O que você vê em objeto-3.jpeg?"
echo "1) Cachorro"
echo "2) Pessoa (Junior)"
echo "3) Outra coisa"
read -p "Escolha (1-3): " IMAGE_TYPE

if [ "$IMAGE_TYPE" = "2" ]; then
    echo ""
    echo -e "${GREEN}✅ OK! A imagem É do Junior${NC}"
    echo "   Neste caso o modelo está CORRETO!"
    echo "   O problema é que o arquivo está mal nomeado."
    echo ""
    echo "Deseja renomear objeto-3.jpeg para junior-X.jpeg? (s/n)"
    read -p "> " RENAME
    if [ "$RENAME" = "s" ]; then
        NEW_NAME="junior-teste-$(date +%s).jpeg"
        mv ../../../dados/fotos_auth/objeto-3.jpeg "../../../dados/fotos_auth/$NEW_NAME"
        echo "✅ Renomeado para: $NEW_NAME"
    fi
    exit 0
    
elif [ "$IMAGE_TYPE" = "3" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  A imagem não é nem cachorro nem Junior${NC}"
    echo "   O modelo não foi treinado para reconhecer este objeto"
    echo "   Isso é normal e esperado!"
    exit 0
fi

# If it's a dog, continue with the fix
echo ""
echo -e "${BLUE}📋 OK, é um cachorro. Vamos corrigir o modelo.${NC}"
echo ""

# Step 3: Backup
echo "📦 Passo 1/5: Criando backup..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

[ -f "face_recognition_model.jld2" ] && cp face_recognition_model.jld2 "$BACKUP_DIR/"
[ -f "face_recognition_config.toml" ] && cp face_recognition_config.toml "$BACKUP_DIR/"

echo -e "${GREEN}✅ Backup criado em: $BACKUP_DIR${NC}"
echo ""

# Step 4: Show training data
echo "📁 Passo 2/5: Verificando imagens de treino..."
echo ""
echo "Imagens de JUNIOR encontradas:"
find ../../../dados/fotos_train -name "junior-*" -type f 2>/dev/null | head -5 | while read file; do
    echo "   - $(basename $file)"
done

echo ""
echo "Imagens de CACHORRO encontradas:"
find ../../../dados/fotos_train -name "cachorro-*" -type f 2>/dev/null | head -5 | while read file; do
    echo "   - $(basename $file)"
done

echo ""
echo -e "${YELLOW}⚠️  ATENÇÃO: Você precisa verificar estas imagens!${NC}"
echo ""
echo "Vamos abrir as primeiras imagens de cada categoria"
echo "Pressione ENTER para continuar..."
read

# Open first junior image
FIRST_JUNIOR=$(find ../../../dados/fotos_train -name "junior-*" -type f 2>/dev/null | head -1)
if [ ! -z "$FIRST_JUNIOR" ]; then
    echo "Abrindo: $FIRST_JUNIOR"
    xdg-open "$FIRST_JUNIOR" 2>/dev/null || open "$FIRST_JUNIOR" 2>/dev/null || true
    echo ""
    echo "Esta imagem é REALMENTE do Junior (pessoa)? (s/n)"
    read -p "> " IS_JUNIOR
    
    if [ "$IS_JUNIOR" != "s" ]; then
        echo ""
        echo -e "${RED}❌ PROBLEMA ENCONTRADO!${NC}"
        echo "   As imagens de 'junior' contêm fotos erradas!"
        echo ""
        echo "SOLUÇÃO:"
        echo "   1. Vá para: ../../../dados/fotos_train/"
        echo "   2. Verifique TODOS os arquivos junior-*.jpg"
        echo "   3. Se houver cachorros, renomeie:"
        echo "      mv junior-X.jpg cachorro-X.jpg"
        echo "   4. Execute novamente este script"
        echo ""
        exit 1
    fi
fi

# Open first cachorro image
FIRST_CACHORRO=$(find ../../../dados/fotos_train -name "cachorro-*" -type f 2>/dev/null | head -1)
if [ ! -z "$FIRST_CACHORRO" ]; then
    echo ""
    echo "Abrindo: $FIRST_CACHORRO"
    xdg-open "$FIRST_CACHORRO" 2>/dev/null || open "$FIRST_CACHORRO" 2>/dev/null || true
    echo ""
    echo "Esta imagem é REALMENTE de um cachorro? (s/n)"
    read -p "> " IS_DOG
    
    if [ "$IS_DOG" != "s" ]; then
        echo ""
        echo -e "${RED}❌ PROBLEMA ENCONTRADO!${NC}"
        echo "   As imagens de 'cachorro' contêm fotos erradas!"
        echo ""
        echo "SOLUÇÃO:"
        echo "   1. Vá para: ../../../dados/fotos_train/"
        echo "   2. Verifique TODOS os arquivos cachorro-*.jpg"
        echo "   3. Se houver pessoas, renomeie para o nome correto"
        echo "   4. Execute novamente este script"
        echo ""
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✅ Imagens parecem estar corretas${NC}"
echo ""

# Step 5: Ask confirmation to retrain
echo "🚀 Passo 3/5: Re-treino necessário"
echo ""
echo "O modelo precisa ser re-treinado com os dados corretos."
echo "Isso vai:"
echo "   - Remover o modelo atual"
echo "   - Treinar um novo modelo do zero"
echo "   - Levar aproximadamente 5-10 minutos"
echo ""
echo -e "${YELLOW}Confirma o re-treino? (s/n)${NC}"
read -p "> " CONFIRM

if [ "$CONFIRM" != "s" ]; then
    echo "❌ Re-treino cancelado"
    echo "Backup mantido em: $BACKUP_DIR"
    exit 1
fi

# Step 6: Remove old model
echo ""
echo "🧹 Passo 4/5: Removendo modelo antigo..."
rm -f face_recognition_model.jld2
rm -f face_recognition_config.toml
rm -f face_recognition_model_data.toml
echo -e "${GREEN}✅ Modelo antigo removido${NC}"
echo ""

# Step 7: Retrain
echo "🧠 Passo 5/5: Treinando novo modelo..."
echo ""
julia cnncheckin_pretrain.jl

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo -e "${GREEN}✅ RE-TREINO CONCLUÍDO COM SUCESSO!${NC}"
    echo "======================================================================"
    echo ""
    
    # Test with the problematic image
    echo "🧪 Testando com objeto-3.jpeg..."
    echo ""
    julia cnncheckin_identify.jl ../../../dados/fotos_auth/objeto-3.jpeg
    
    echo ""
    echo "======================================================================"
    echo "📋 PRÓXIMOS PASSOS:"
    echo "======================================================================"
    echo ""
    echo "1. Verifique o resultado acima"
    echo "2. Se objeto-3.jpeg AINDA é identificado como 'junior':"
    echo "   → O problema está nos DADOS DE TREINO"
    echo "   → Você precisa corrigir as imagens manualmente"
    echo ""
    echo "3. Como corrigir manualmente:"
    echo "   a) cd ../../../dados/fotos_train"
    echo "   b) Abra CADA arquivo junior-*.jpg"
    echo "   c) Se houver cachorro, renomeie:"
    echo "      mv junior-X.jpg cachorro-X.jpg"
    echo "   d) Execute novamente: bash QUICK_FIX.sh"
    echo ""
    echo "4. Se agora identifica corretamente como 'cachorro':"
    echo "   → ${GREEN}🎉 PROBLEMA RESOLVIDO!${NC}"
    echo ""
    
else
    echo ""
    echo "======================================================================"
    echo -e "${RED}❌ ERRO NO RE-TREINO${NC}"
    echo "======================================================================"
    echo ""
    echo "Restaurando backup..."
    cp "$BACKUP_DIR/face_recognition_model.jld2" . 2>/dev/null || true
    cp "$BACKUP_DIR/face_recognition_config.toml" . 2>/dev/null || true
    echo ""
    echo "Para investigar o erro:"
    echo "   julia diagnose_and_fix.jl"
    echo ""
    exit 1
fi