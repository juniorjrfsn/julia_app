
## ***Instalação de Júlia pelo Dockerfile***

```
# ============================= JULIA ===========================
FROM julia:latest
# Instalar dependências do sistema necessárias para pacotes gráficos (Gtk, VideoIO, ImageMagick)
RUN apt-get update && apt-get install -y \
    libgtk-3-dev \
    imagemagick \
    ffmpeg \
    gettext \
    && rm -rf /var/lib/apt/lists/*
# Instalação das bibliotecas Julia
 RUN julia -e 'using Pkg; Pkg.add([ \
     "JSON", "Pkg", "SQLite", "DataFrames", "Flux", "StatsBase", \
     "CSV", "MLJ", "MLJFlux", "Images", "FileIO", "Serialization", \
     "Distributions", "ImageDraw", "ImageCore", "JSON3", "Luxor", \
     "ImageMagick", "QuartzImageIO", "JLD2", "MLDatasets", "CUDA", \
     "Statistics", "Random", "ImageTransformations", "LinearAlgebra", \
     "Plots", "VideoIO", "ImageView", "Dates", "Gtk", "PlotlyJS", \
     "GtkDrawingArea", \
     "ColorSchemes", "Optimisers", "Colors", "Glob", "TOML" \
     ]); Pkg.precompile()'
```

  [336ed68f] CSV v0.10.16
⌃ [052768ef] CUDA v5.11.1
  [a93c6f00] DataFrames v1.8.2
  [31c24e10] Distributions v0.25.125
  [5789e2e9] FileIO v1.18.0
  [587475ba] Flux v0.16.10
  [4c0ca9eb] Gtk v1.3.1
  [a09fc81d] ImageCore v0.10.5
  [4381153b] ImageDraw v0.2.6
  [6218d12a] ImageMagick v1.4.2
  [02fcd773] ImageTransformations v0.10.2
  [86fae568] ImageView v0.13.1
  [916415d5] Images v0.26.2
  [033835bb] JLD2 v0.6.4
⌅ [682c06a0] JSON v0.21.4
  [0f8b85d8] JSON3 v1.14.3
  [ae8d54c2] Luxor v4.5.0
  [eb30cadb] MLDatasets v0.7.21
  [add582a8] MLJ v0.23.2
  [094fc8d1] MLJFlux v0.6.7
  [f0f68f2c] PlotlyJS v0.18.18
  [91a5bcdd] Plots v1.41.6
  [dca85d43] QuartzImageIO v0.7.5
  [0aa819cd] SQLite v1.8.0
  [2913bbd2] StatsBase v0.34.10
  [d6d074c3] VideoIO v1.6.1
  [02a925ec] cuDNN v6.0.0
  [ade2ca70] Dates
  [37e2e46d] LinearAlgebra
  [44cfe95a] Pkg v1.10.0
  [9a3f8284] Random
  [9e88b42a] Serialization
  [10745b16] Statistics v1.10.0

```
julia
]

add JSON
add Pkg
add SQLite
add DataFrames
add Flux
add StatsBase
add CSV
add MLJ
add MLJFlux
add Images
add FileIO
add Serialization
add Distributions
add ImageDraw
add ImageCore
add JSON3
add Luxor
add ImageMagick
add QuartzImageIO
add JLD2
add MLDatasets
add CUDA
add Statistics
add Random
add ImageTransformations
add LinearAlgebra
add Plots
add VideoIO
add ImageView
add Dates
add Gtk
add PlotlyJS
add cuDNN
add GLib
add GNNlib
add PGLib
add GenLib
add GtkDrawingArea
add ColorSchemes
add Optimisers
add Colors
add Glob
add TOML
```

# S:\DGI-Diretoria de Gestão da Informação\Bkp - Backup - Ex-servidores\Breno\Installers\UTIL\pgAdmin 4\v4\docs\en_US\html\_sources

Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V
whoami
net localgroup "Hyper-V Administrators" $env:USERNAME /add
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
docker logs docker-dev
docker exec docker-dev ps aux | findstr sshd
docker exec docker-dev /usr/sbin/sshd -t
docker images

Solução rápida (sessão atual)
Execute isso no PowerShell como Administrador antes de rodar o script:
powershellSet-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
Depois rode normalmente:
powershell.\docker.ps1 start

Esse modo é temporário — vale só para a sessão atual do PowerShell.

Solução permanente para seu usuário
powershellSet-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
RemoteSigned permite rodar scripts locais sem assinatura, mas exige assinatura em scripts baixados da internet. É o nível recomendado para desenvolvimento.

Alternativa sem mudar a política
powershellPowerShell -ExecutionPolicy Bypass -File .\docker.ps1 start
