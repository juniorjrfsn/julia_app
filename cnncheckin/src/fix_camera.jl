#!/usr/bin/env julia
"""
Script de diagnóstico e correção para problemas de câmera
Soluciona: CUDA errors, VideoIO issues, Iriun detection
"""

using Pkg

println("🔧 DIAGNÓSTICO E CORREÇÃO - CNNCheckin")
println("="^70)

# ============================================================================
# 1. CORRIGIR ERRO CUDA
# ============================================================================

println("\n1️⃣ Verificando/Corrigindo CUDA...")

if haskey(Pkg.project().dependencies, "CUDA")
    println("   ⚠️  CUDA detectado (não necessário sem GPU NVIDIA)")
    print("   Deseja remover CUDA? (S/n): ")
    response = readline()
    
    if lowercase(strip(response)) != "n"
        println("   Removendo CUDA...")
        try
            Pkg.rm("CUDA")
            println("   ✅ CUDA removido com sucesso!")
            println("   ⚠️  Reinicie Julia após este script")
        catch e
            println("   ⚠️  Erro ao remover CUDA: $e")
        end
    end
else
    println("   ✅ CUDA não instalado (correto para CPU)")
end

# ============================================================================
# 2. VERIFICAR E REINSTALAR VIDEOIO
# ============================================================================

println("\n2️⃣ Verificando VideoIO...")

try
    # Tentar carregar VideoIO
    @eval using VideoIO
    println("   ✅ VideoIO carregado")
    
    # Verificar se tem problemas
    print("   Testando funcionalidade... ")
    
    # Verificar métodos disponíveis
    if isdefined(VideoIO, :opencamera)
        println("✅")
        println("   ✅ Método opencamera disponível")
    else
        println("❌")
        println("   ⚠️  Método opencamera não encontrado")
        
        print("   Deseja reinstalar VideoIO? (S/n): ")
        if lowercase(strip(readline())) != "n"
            println("   Reinstalando VideoIO...")
            Pkg.rm("VideoIO")
            Pkg.add("VideoIO")
            Pkg.build("VideoIO")
            println("   ✅ VideoIO reinstalado")
        end
    end
    
catch e
    println("   ❌ Erro ao carregar VideoIO: $e")
    
    print("   Deseja reinstalar VideoIO? (S/n): ")
    if lowercase(strip(readline())) != "n"
        println("   Instalando VideoIO...")
        try
            Pkg.add("VideoIO")
            Pkg.build("VideoIO")
            println("   ✅ VideoIO instalado")
        catch install_error
            println("   ❌ Erro na instalação: $install_error")
        end
    end
end

# ============================================================================
# 3. VERIFICAR SISTEMA
# ============================================================================

println("\n3️⃣ Verificando sistema...")

# Sistema operacional
if Sys.islinux()
    println("   ✅ Sistema: Linux")
    
    # Verificar dispositivos de vídeo
    println("\n   📹 Dispositivos de vídeo:")
    try
        run(pipeline(`ls /dev/video*`, stdout=devnull, stderr=devnull))
        run(`ls -l /dev/video*`)
    catch
        println("   ⚠️  Nenhum dispositivo /dev/video* encontrado")
        println("\n   💡 Possíveis soluções:")
        println("      1. Conecte uma webcam USB")
        println("      2. Se usar Iriun, inicie o app no celular")
        println("      3. Verifique permissões: ls -l /dev/video*")
        println("      4. Adicione usuário ao grupo video:")
        println("         sudo usermod -a -G video \$USER")
        println("         (faça logout/login após)")
    end
    
    # Verificar v4l2
    println("\n   🔍 Verificando v4l-utils...")
    try
        run(pipeline(`which v4l2-ctl`, stdout=devnull))
        println("   ✅ v4l2-ctl instalado")
        
        println("\n   📋 Listando câmeras com v4l2-ctl:")
        try
            run(`v4l2-ctl --list-devices`)
        catch
            println("   ⚠️  Erro ao listar dispositivos")
        end
    catch
        println("   ⚠️  v4l2-ctl não instalado")
        println("   💡 Instale com: sudo apt-get install v4l-utils")
    end
    
    # Verificar FFmpeg
    println("\n   🎬 Verificando FFmpeg...")
    try
        run(pipeline(`which ffmpeg`, stdout=devnull))
        println("   ✅ FFmpeg instalado")
    catch
        println("   ⚠️  FFmpeg não instalado")
        println("   💡 Instale com: sudo apt-get install ffmpeg")
    end
    
    # Verificar Iriun
    println("\n   📱 Verificando Iriun Webcam...")
    try
        run(pipeline(`which iriunwebcam`, stdout=devnull))
        println("   ✅ Iriun instalado")
        
        # Verificar se está rodando
        try
            run(pipeline(`pgrep -f iriun`, stdout=devnull))
            println("   ✅ Serviço Iriun rodando")
        catch
            println("   ⚠️  Serviço Iriun não está rodando")
            println("   💡 Inicie com: sudo systemctl start iriunwebcam")
        end
    catch
        println("   ℹ️  Iriun não instalado (opcional)")
    end
    
elseif Sys.iswindows()
    println("   ✅ Sistema: Windows")
    println("   💡 Verifique no Gerenciador de Dispositivos se a webcam aparece")
    
elseif Sys.isapple()
    println("   ✅ Sistema: macOS")
    println("   💡 Verifique as permissões de câmera em Preferências do Sistema")
end

# ============================================================================
# 4. TESTAR CAPTURA COM VIDEOIO
# ============================================================================

println("\n4️⃣ Testando captura com VideoIO...")

try
    using VideoIO
    
    println("   Testando índices de câmera 0-10...")
    cameras_found = []
    
    for i in 0:10
        try
            print("   Câmera $i: ")
            
            # Método 1: Tentar com VideoIO.opencamera
            cam = VideoIO.opencamera(i)
            
            if cam !== nothing
                try
                    frame = read(cam)
                    if frame !== nothing
                        h, w = size(frame)[1:2]
                        println("✅ Funcional ($(w)x$(h))")
                        push!(cameras_found, i)
                    else
                        println("⚠️  Abriu mas não captura")
                    end
                catch read_error
                    println("⚠️  Abriu mas erro ao ler: $(typeof(read_error).name)")
                end
                
                try
                    close(cam)
                catch
                end
            else
                println("❌ Não abriu")
            end
            
        catch e
            error_type = typeof(e).name
            if error_type == "MethodError"
                println("❌ MethodError (problema VideoIO)")
            elseif error_type == "ArgumentError"
                println("❌ Não existe")
            else
                println("❌ Erro: $error_type")
            end
        end
    end
    
    if isempty(cameras_found)
        println("\n   ❌ Nenhuma câmera funcional encontrada com VideoIO!")
        println("\n   💡 Soluções:")
        println("      1. Use o script Python (mais confiável):")
        println("         python3 capture_opencv.py --list")
        println("      2. Use capture_iriun.jl (alternativo)")
        println("      3. Verifique se outro programa está usando a câmera")
        println("      4. Reinicie o computador")
    else
        println("\n   ✅ Câmeras funcionais: $(join(cameras_found, ", "))")
        println("\n   💡 Use estas câmeras nos comandos:")
        println("      julia cnncheckin_capture.jl --train \"Nome\" 15 --camera $(cameras_found[1])")
    end
    
catch videoio_error
    println("\n   ❌ Erro ao usar VideoIO: $videoio_error")
    println("\n   💡 Alternativas:")
    println("      1. Use Python + OpenCV:")
    println("         pip3 install opencv-python")
    println("         python3 capture_opencv.py --list")
    println("      2. Reinstale VideoIO:")
    println("         julia -e 'using Pkg; Pkg.rm(\"VideoIO\"); Pkg.add(\"VideoIO\"); Pkg.build(\"VideoIO\"))'")
end

# ============================================================================
# 5. SCRIPT PYTHON ALTERNATIVO
# ============================================================================

println("\n5️⃣ Verificando alternativa Python...")

# Verificar se Python está instalado
try
    run(pipeline(`which python3`, stdout=devnull))
    println("   ✅ Python3 instalado")
    
    # Verificar OpenCV
    try
        run(pipeline(`python3 -c "import cv2"`, stdout=devnull, stderr=devnull))
        println("   ✅ OpenCV instalado")
        
        # Verificar se script existe
        if isfile("capture_opencv.py")
            println("   ✅ Script capture_opencv.py encontrado")
            println("\n   🚀 RECOMENDAÇÃO: Use o script Python para captura:")
            println("      python3 capture_opencv.py --list")
            println("      python3 capture_opencv.py --test --camera 0")
            println("      python3 capture_opencv.py --multiple \"Nome\" ../dados/fotos_train 15 --camera 0")
        else
            println("   ⚠️  Script capture_opencv.py não encontrado")
            println("   💡 Copie o script Python fornecido anteriormente")
        end
        
    catch
        println("   ⚠️  OpenCV não instalado")
        println("   💡 Instale com: pip3 install opencv-python")
    end
    
catch
    println("   ⚠️  Python3 não instalado")
    println("   💡 Instale Python3 primeiro")
end

# ============================================================================
# RESUMO E RECOMENDAÇÕES
# ============================================================================

println("\n" * "="^70)
println("📋 RESUMO E RECOMENDAÇÕES")
println("="^70)

println("\n🔧 Ações imediatas:")
println("   1. Remova CUDA se não tiver GPU NVIDIA")
println("   2. Verifique se webcam está conectada")
println("   3. Use Python + OpenCV (mais estável):")
println("      pip3 install opencv-python")
println("      python3 capture_opencv.py --list")
println("")
println("   4. Para Linux + Iriun:")
println("      a) Inicie app Iriun no celular")
println("      b) Conecte via USB ou WiFi")
println("      c) Verifique: ls -l /dev/video*")
println("      d) Teste: python3 capture_opencv.py --test --camera 2")

println("\n📚 Workflow recomendado:")
println("   # 1. Capturar imagens (Python)")
println("   python3 capture_opencv.py --multiple \"Pessoa1\" ../dados/fotos_train 15 --camera 0")
println("   python3 capture_opencv.py --multiple \"Pessoa2\" ../dados/fotos_train 15 --camera 0")
println("")
println("   # 2. Treinar modelo (Julia)")
println("   julia cnncheckin_pretrain_webcam.jl --no-capture")
println("")
println("   # 3. Identificar (Python para captura)")
println("   python3 capture_opencv.py --single foto_teste.jpg --camera 0")
println("   julia cnncheckin_identify.jl foto_teste.jpg")

println("\n⚠️  Se problemas persistirem:")
println("   1. Reinicie o computador")
println("   2. Teste webcam em outro programa (Cheese, VLC)")
println("   3. Verifique permissões: sudo usermod -a -G video \$USER")
println("   4. Reinstale drivers da webcam")

println("\n✅ Próximos passos:")
println("   1. Execute: python3 capture_opencv.py --list")
println("   2. Teste uma câmera: python3 capture_opencv.py --test --camera N")
println("   3. Capture imagens: python3 capture_opencv.py --multiple \"Nome\" dir 15 --camera N")
println("   4. Treine: julia cnncheckin_pretrain_webcam.jl --no-capture")

println("\n" * "="^70)
println("Script finalizado!")
println("="^70)