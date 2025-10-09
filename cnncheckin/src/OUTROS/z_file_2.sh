#!/bin/bash
# Script para instalar dependências do sistema para CNN Check-In

# chmod +x setup_system.sh
# ./setup_system.sh

# Execute com: bash z_file_2.sh

echo "🔧 Instalando dependências do sistema para CNN Check-In..."

# Atualizar repositórios
echo "📦 Atualizando repositórios..."
sudo apt update

# Instalar dependências GTK e desenvolvimento
echo "📦 Instalando bibliotecas GTK..."
sudo apt install -y \
    libgtk-3-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libglib2.0-dev \
    libgdk-pixbuf2.0-dev \
    libatk1.0-dev

# Instalar dependências para vídeo/webcam
echo "📦 Instalando suporte para webcam..."
sudo apt install -y \
    v4l-utils \
    libv4l-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Instalar dependências para imagens
echo "📦 Instalando bibliotecas de imagem..."
sudo apt install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev

# Instalar visualizadores de imagem
echo "📦 Instalando visualizadores de imagem..."
sudo apt install -y \
    eog \
    feh \
    imagemagick

# Verificar instalação da webcam
echo "📹 Verificando dispositivos de vídeo..."
ls -la /dev/video* 2>/dev/null && echo "✅ Dispositivos de vídeo encontrados" || echo "⚠️ Nenhum dispositivo de vídeo encontrado"

# Verificar se usuário está no grupo video
if groups $USER | grep -q video; then
    echo "✅ Usuário já está no grupo 'video'"
else
    echo "🔧 Adicionando usuário ao grupo 'video'..."
    sudo usermod -a -G video $USER
    echo "⚠️ IMPORTANTE: Faça logout e login novamente para aplicar as mudanças"
fi

echo "✅ Instalação das dependências do sistema concluída!"
echo "💡 Próximo passo: Execute o script Julia para instalar os pacotes"