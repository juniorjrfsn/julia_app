#!/usr/bin/env bash
# projeto: cnncheckin
# file: exemplo_completo.sh
# Fluxo completo: câmeras → captura → treino → identificação → incremental → check-in
# Execute da raiz do projeto.

set -euo pipefail

JL="julia --project=."
MAIN="src/cnncheckin.jl"

pause() { read -r -p $'\n⏸  ENTER para continuar... '; }
step()  { printf '\n══════════ %s ══════════\n' "$1"; }

[ -f "$MAIN" ] || { echo "Execute este script na raiz do projeto cnncheckin."; exit 1; }

mkdir -p dados/fotos_train dados/fotos_new dados/fotos_auth

step "0. Dependências"
$JL -e 'using Pkg; Pkg.instantiate()'

step "1. Câmeras disponíveis"
$JL "$MAIN" --list-cameras
read -r -p "Índice da câmera a usar [0]: " CAM
CAM=${CAM:-0}
pause

step "2. Cadastro inicial (15 fotos por pessoa)"
for pessoa in "João" "Maria"; do
    echo "Posicione: $pessoa"
    pause
    $JL "$MAIN" --capture "$pessoa" dados/fotos_train 15 --camera "$CAM"
done

step "3. Treinamento"
$JL "$MAIN" --train dados/fotos_train

step "4. Identificação por foto"
FOTO=dados/fotos_auth/teste.jpg
$JL src/capture_iriun.jl --single "$FOTO" --camera "$CAM"
$JL "$MAIN" --identify "$FOTO"
pause

step "5. Nova pessoa (aprendizado incremental)"
read -r -p "Nome da nova pessoa: " NOVA
$JL "$MAIN" --capture "$NOVA" dados/fotos_new 15 --camera "$CAM"
$JL "$MAIN" --add dados/fotos_new

step "6. Check-in / check-out"
echo "Primeira leitura de uma pessoa = CHECK-IN; a seguinte = CHECK-OUT."
$JL "$MAIN" --checkin presenca_exemplo.csv --camera "$CAM"

[ -f presenca_exemplo.csv ] && { echo; cat presenca_exemplo.csv; }

echo
echo "Pronto. Dicas: faça backup de src/face_recognition_*, mantenha iluminação"
echo "parecida com a do uso real e use 15–20 fotos por pessoa."
