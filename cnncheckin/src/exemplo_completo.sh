#!/bin/bash
# projeto: cnncheckin
# file: exemplo_completo.sh
# descrição: Script de exemplo mostrando o fluxo completo do sistema

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                       ║"
echo "║           🎥 CNNCheckin - Exemplo de Uso Completo                    ║"
echo "║           Sistema de Reconhecimento Facial com Webcam                ║"
echo "║                                                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para pausa
pause() {
    echo ""
    echo -e "${YELLOW}⏸️  Pressione ENTER para continuar...${NC}"
    read
}

# Navegar para diretório src
cd src 2>/dev/null || {
    echo -e "${RED}❌ Erro: Diretório 'src' não encontrado${NC}"
    echo "Execute este script da raiz do projeto cnncheckin"
    exit 1
}

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  CENÁRIO: Sistema de Controle de Acesso em uma Empresa"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Este exemplo demonstra:"
echo "  1. Cadastro inicial de funcionários"
echo "  2. Treinamento do modelo"
echo "  3. Adição de novo funcionário"
echo "  4. Identificação de pessoas"
echo "  5. Sistema de check-in/check-out"
echo ""
pause

# ═══════════════════════════════════════════════════════════════════════
# FASE 1: SETUP INICIAL E VERIFICAÇÃO
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  FASE 1: Verificação do Sistema                                      ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${BLUE}🔍 Verificando câmeras disponíveis...${NC}"
julia cnncheckin_capture.jl --cameras

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Erro: Nenhuma câmera detectada ou erro ao executar${NC}"
    echo ""
    echo "Possíveis soluções:"
    echo "  1. Conecte uma webcam ao computador"
    echo "  2. Verifique se a webcam funciona em outros programas"
    echo "  3. Verifique permissões (Linux/macOS: sudo usermod -a -G video \$USER)"
    echo "  4. Windows: verifique drivers da webcam"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Câmera detectada!${NC}"
pause

echo ""
echo -e "${BLUE}📸 Testando preview da câmera (5 segundos)...${NC}"
echo "Você deverá ver uma janela com a imagem da câmera"
echo ""
julia cnncheckin_capture.jl --preview 0 5

pause

# ═══════════════════════════════════════════════════════════════════════
# FASE 2: CADASTRO INICIAL DE FUNCIONÁRIOS
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  FASE 2: Cadastro Inicial de Funcionários                            ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Vamos cadastrar 3 funcionários iniciais:"
echo "  1. João Silva (Gerente)"
echo "  2. Maria Santos (Desenvolvedora)"
echo "  3. Pedro Costa (Designer)"
echo ""
echo "⚠️  IMPORTANTE: Você precisará capturar fotos reais via webcam"
echo "   Dicas:"
echo "   - Use boa iluminação"
echo "   - Mantenha-se centralizado na câmera"
echo "   - Varie levemente a posição entre as fotos"
echo "   - Cada pessoa: ~15 fotos (~2 minutos)"
echo ""
pause

echo -e "${BLUE}🎯 Iniciando cadastro no modo rápido...${NC}"
echo ""
echo "OPÇÃO 1: Modo Rápido (Recomendado)"
echo "  julia cnncheckin_pretrain_webcam.jl --quick \"João Silva\" \"Maria Santos\" \"Pedro Costa\" --num 15"
echo ""
echo "OPÇÃO 2: Modo Interativo"
echo "  julia cnncheckin_pretrain_webcam.jl"
echo ""
echo -e "${YELLOW}Para este exemplo, vamos usar o modo INTERATIVO para melhor controle${NC}"
pause

# Executar treinamento inicial
julia cnncheckin_pretrain_webcam.jl

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Erro no treinamento inicial${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Treinamento inicial concluído!${NC}"
echo ""
echo "Arquivos gerados:"
echo "  📄 face_recognition_model.jld2 (modelo treinado)"
echo "  📄 face_recognition_config.toml (configurações)"
echo "  📄 face_recognition_model_data.toml (metadados)"
pause

# ═══════════════════════════════════════════════════════════════════════
# FASE 3: TESTE DE IDENTIFICAÇÃO
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  FASE 3: Teste de Identificação                                      ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${BLUE}🎯 Vamos testar a identificação com as pessoas cadastradas${NC}"
echo ""
echo "Teste 1: Identificação Única"
echo "  - Captura uma foto"
echo "  - Identifica quem é a pessoa"
echo ""
pause

echo -e "${YELLOW}📸 Posicione uma das pessoas cadastradas na câmera${NC}"
julia cnncheckin_identify_webcam.jl --identify

pause

echo ""
echo "Teste 2: Autenticação"
echo "  - Verifica se a pessoa é quem diz ser"
echo "  - Útil para controle de acesso"
echo ""
echo -e "${YELLOW}Digite o nome da pessoa para autenticar (ex: João Silva):${NC}"
read pessoa_nome

if [ -n "$pessoa_nome" ]; then
    echo ""
    echo -e "${BLUE}🔐 Autenticando: $pessoa_nome${NC}"
    julia cnncheckin_identify_webcam.jl --auth "$pessoa_nome" 0.7
fi

pause

# ═══════════════════════════════════════════════════════════════════════
# FASE 4: ADICIONAR NOVO FUNCIONÁRIO (APRENDIZADO INCREMENTAL)
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  FASE 4: Adicionar Novo Funcionário                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Cenário: Chegou um novo funcionário!"
echo "  4. Carlos Alberto (Analista de Dados)"
echo ""
echo "Vamos usar o APRENDIZADO INCREMENTAL:"
echo "  ✅ Mais rápido (não retreina tudo)"
echo "  ✅ Mantém conhecimento anterior"
echo "  ✅ Adiciona apenas a nova pessoa"
echo ""
pause

echo -e "${BLUE}➕ Adicionando Carlos Alberto...${NC}"
echo ""
echo "OPÇÃO 1: Modo Rápido"
echo "  julia cnncheckin_incremental_webcam.jl --quick \"Carlos Alberto\" --num 10"
echo ""
echo "OPÇÃO 2: Modo Interativo"
echo "  julia cnncheckin_incremental_webcam.jl"
echo ""
pause

julia cnncheckin_incremental_webcam.jl

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Erro no aprendizado incremental${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Novo funcionário adicionado!${NC}"
echo ""
echo "O modelo agora reconhece 4 pessoas:"
echo "  1. João Silva"
echo "  2. Maria Santos"
echo "  3. Pedro Costa"
echo "  4. Carlos Alberto"
pause

# ═══════════════════════════════════════════════════════════════════════
# FASE 5: SISTEMA DE CHECK-IN/CHECK-OUT
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  FASE 5: Sistema de Check-in/Check-out                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${BLUE}📋 Iniciando sistema de registro de presença${NC}"
echo ""
echo "Este sistema:"
echo "  ✅ Registra entrada (check-in) e saída (check-out)"
echo "  ✅ Mantém lista de pessoas presentes"
echo "  ✅ Gera log em arquivo CSV"
echo "  ✅ Timestamp automático"
echo ""
echo "Funcionamento:"
echo "  - Primeira foto de uma pessoa = CHECK-IN"
echo "  - Segunda foto da mesma pessoa = CHECK-OUT"
echo "  - E assim sucessivamente..."
echo ""
echo "Para este exemplo, faremos alguns registros de teste"
echo ""
pause

echo -e "${YELLOW}Sistema de check-in iniciando...${NC}"
echo ""
echo "💡 Durante o teste:"
echo "   1. Pressione ENTER para registrar"
echo "   2. Posicione uma pessoa na câmera"
echo "   3. Sistema captura e registra"
echo "   4. Digite 'sair' para encerrar"
echo ""
echo "Arquivo de log: presenca_exemplo.csv"
echo ""
pause

julia cnncheckin_identify_webcam.jl --checkin presenca_exemplo.csv

# ═══════════════════════════════════════════════════════════════════════
# FASE 6: RELATÓRIOS E ANÁLISE
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  FASE 6: Relatórios e Análise                                        ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "presenca_exemplo.csv" ]; then
    echo -e "${BLUE}📊 Analisando log de presença...${NC}"
    echo ""
    echo "Conteúdo do arquivo presenca_exemplo.csv:"
    echo "─────────────────────────────────────────────────────────────────"
    cat presenca_exemplo.csv
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    
    # Contar registros
    total_registros=$(wc -l < presenca_exemplo.csv)
    echo "📈 Estatísticas:"
    echo "   Total de registros: $total_registros"
    echo ""
    
    # Contar por ação
    checkins=$(grep "CHECK-IN" presenca_exemplo.csv | wc -l)
    checkouts=$(grep "CHECK-OUT" presenca_exemplo.csv | wc -l)
    echo "   Check-ins: $checkins"
    echo "   Check-outs: $checkouts"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 EXEMPLO COMPLETO FINALIZADO!                                     ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${GREEN}✅ Você completou todas as fases do sistema!${NC}"
echo ""
echo "Resumo do que fizemos:"
echo "  1. ✅ Verificação de câmera"
echo "  2. ✅ Cadastro de 3 funcionários iniciais"
echo "  3. ✅ Treinamento do modelo CNN"
echo "  4. ✅ Testes de identificação e autenticação"
echo "  5. ✅ Adição de novo funcionário (incremental)"
echo "  6. ✅ Sistema de check-in/check-out"
echo "  7. ✅ Análise de relatórios"
echo ""

echo "📁 Arquivos gerados:"
echo "   • face_recognition_model.jld2"
echo "   • face_recognition_config.toml"
echo "   • face_recognition_model_data.toml"
echo "   • presenca_exemplo.csv"
echo ""

echo "🚀 Próximos passos:"
echo ""
echo "  Para uso contínuo:"
echo "    julia cnncheckin_identify_webcam.jl --continuous 10"
echo ""
echo "  Para adicionar mais pessoas:"
echo "    julia cnncheckin_incremental_webcam.jl --quick \"Nome Pessoa\""
echo ""
echo "  Para sistema de check-in permanente:"
echo "    julia cnncheckin_identify_webcam.jl --checkin presenca_diaria.csv"
echo ""
echo "  Ver documentação completa:"
echo "    cat ../README_WEBCAM.md"
echo ""

echo "💡 Dicas importantes:"
echo ""
echo "  1. BACKUP: Faça backup regular dos arquivos .jld2 e .toml"
echo "     tar -czf backup_modelo_\$(date +%Y%m%d).tar.gz face_recognition_*"
echo ""
echo "  2. SEGURANÇA: Proteja o modelo treinado"
echo "     chmod 600 face_recognition_*.jld2"
echo ""
echo "  3. MANUTENÇÃO: Retreine periodicamente (a cada 2-3 meses)"
echo "     julia cnncheckin_pretrain_webcam.jl --no-capture"
echo ""
echo "  4. QUALIDADE: Mais fotos = melhor acurácia"
echo "     Mínimo: 10 fotos/pessoa"
echo "     Recomendado: 15-20 fotos/pessoa"
echo ""
echo "  5. ILUMINAÇÃO: Mantenha condições consistentes"
echo "     Capture em iluminações similares ao uso real"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}🎊 Obrigado por testar o CNNCheckin!${NC}"
echo ""
echo "Para mais informações:"
echo "  📖 Documentação: README_WEBCAM.md"
echo "  💬 Julia Discourse: https://discourse.julialang.org/"
echo "  🐛 Issues: Reporte problemas no repositório"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"