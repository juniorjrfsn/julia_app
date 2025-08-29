#!/bin/bash

# Script para instalar dependências GTK no Ubuntu/Debian
# Para uso com CNN Check-In Julia Application

echo "🔧 Instalando dependências GTK para CNN Check-In..."

# Atualizar repositórios
echo "📦 Atualizando repositórios..."
sudo apt update

# Instalar bibliotecas GTK essenciais
echo "📚 Instalando bibliotecas GTK..."
sudo apt install -y \
    libgtk-3-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libglib2.0-dev \
    libgdk-pixbuf2.0-dev \
    libatk1.0-dev \
    gtk2-engines-pixbuf

# Instalar módulos GTK extras (para resolver warnings)
echo "🔌 Instalando módulos GTK extras..."
sudo apt install -y \
    appmenu-gtk3-module \
    libcanberra-gtk3-module \
    gir1.2-gtk-3.0

# Instalar suporte para webcam
echo "📹 Instalando suporte para webcam..."
sudo apt install -y \
    v4l-utils \
    libv4l-dev \
    uvcdynctrl

# Instalar visualizadores de imagem
echo "🖼️ Instalando visualizadores de imagem..."
sudo apt install -y \
    eog \
    feh \
    imagemagick

# Verificar instalações
echo "✅ Verificando instalações..."

# Verificar GTK
if pkg-config --exists gtk+-3.0; then
    GTK_VERSION=$(pkg-config --modversion gtk+-3.0)
    echo "✅ GTK+ 3.0 instalado: versão $GTK_VERSION"
else
    echo "❌ GTK+ 3.0 não encontrado"
fi

# Verificar Cairo
if pkg-config --exists cairo; then
    CAIRO_VERSION=$(pkg-config --modversion cairo)
    echo "✅ Cairo instalado: versão $CAIRO_VERSION"
else
    echo "❌ Cairo não encontrado"
fi

# Verificar webcam
echo "📹 Verificando dispositivos de vídeo..."
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "✅ Dispositivos de vídeo encontrados:"
    ls -la /dev/video*
else
    echo "⚠️ Nenhum dispositivo de vídeo encontrado"
fi

# Instalar pacotes Julia
echo "📦 Reconstruindo pacotes Julia..."
julia -e "
using Pkg
println(\"🔄 Removendo pacotes existentes...\")
try
    Pkg.rm([\"Gtk\", \"Cairo\", \"VideoIO\", \"Images\", \"FileIO\"])
catch
    println(\"⚠️ Alguns pacotes não estavam instalados\")
end

println(\"📦 Reinstalando pacotes...\")
Pkg.add([\"Gtk\", \"Cairo\", \"VideoIO\", \"Images\", \"FileIO\", \"Dates\"])

println(\"🔨 Reconstruindo pacotes...\")
Pkg.build([\"Gtk\", \"Cairo\"])

println(\"✅ Testando imports...\")
try
    using Gtk, Cairo, VideoIO, Images, FileIO, Dates
    println(\"✅ Todos os pacotes importados com sucesso\")
catch e
    println(\"❌ Erro ao importar: \$e\")
end
"

echo ""
echo "🎉 Instalação concluída!"
echo ""
echo "📋 Próximos passos:"
echo "1. Reinicie o terminal ou faça logout/login"
echo "2. Execute: julia cnncheckin_acount_fixed.jl"
echo ""
echo "🔧 Se ainda houver problemas:"
echo "- Verifique se sua webcam está conectada"
echo "- Execute: v4l2-ctl --list-devices"
echo "- Reinicie o computador se necessário"