#!/usr/bin/env julia
"""
Teste simples e direto de câmera
"""

println("🔍 TESTE SIMPLES DE CÂMERA")
println("="^60)

# 1. Verificar dispositivos Linux
println("\n1️⃣ Dispositivos de vídeo:")
try
    video_devices = readdir("/dev")
    video_list = filter(f -> startswith(f, "video"), video_devices)
    
    if !isempty(video_list)
        for device in video_list
            device_path = "/dev/$device"
            println("   ✔ $device_path")
        end
        println("   Total: $(length(video_list)) dispositivos")
    else
        println("   ⚠️  NENHUM dispositivo /dev/video* encontrado!")
        println("\n   💡 Soluções:")
        println("      - Conecte uma webcam USB")
        println("      - Se usar Iriun, inicie o app no celular")
        println("      - Verifique: sudo usermod -a -G video \$USER")
    end
catch e
    println("   ⚠️  Erro ao listar dispositivos: $e")
end

# 2. Testar com VideoIO
println("\n2️⃣ Testando VideoIO:")
try
    using VideoIO
    println("   ✅ VideoIO carregado")
    
    cameras_found = Int[]
    
    for i in 0:10
        try
            print("   Câmera $i: ")
            cam = VideoIO.opencamera(i)
            
            if cam !== nothing
                frame = read(cam)
                close(cam)
                
                if frame !== nothing
                    h, w = size(frame)[1:2]
                    println("✅ $(w)x$(h)")
                    push!(cameras_found, i)
                else
                    println("⚠️  Abriu mas não captura")
                end
            else
                println("❌")
            end
        catch e
            println("❌ $(typeof(e).name)")
        end
    end
    
    if isempty(cameras_found)
        println("\n   ❌ NENHUMA câmera funcional!")
    else
        println("\n   ✅ Câmeras funcionais: $(join(cameras_found, ", "))")
        println("\n   🎯 Use nos comandos:")
        println("      julia cnncheckin_capture.jl --cameras")
        println("      julia cnncheckin_capture.jl --preview $(cameras_found[1]) 5")
    end
    
catch e
    println("   ❌ Erro ao usar VideoIO: $e")
end

# 3. Verificar Python + OpenCV
println("\n3️⃣ Verificando Python + OpenCV:")
try
    run(pipeline(`python3 -c "import cv2; print('OpenCV', cv2.__version__)"`, stderr=devnull))
    println("   ✅ Python + OpenCV funcionando!")
    println("\n   🚀 RECOMENDAÇÃO:")
    println("      python3 capture_opencv.py --list")
    println("      python3 capture_opencv.py --test --camera 0")
catch
    println("   ⚠️  OpenCV não instalado")
    println("   💡 Instale: pip3 install opencv-python")
end

# 4. Resumo
println("\n" * "="^60)
println("📋 PRÓXIMOS PASSOS:")
println("="^60)

video_count = try
    video_devices = readdir("/dev")
    length(filter(f -> startswith(f, "video"), video_devices))
catch
    0
end

if video_count == 0
    println("\n❌ PROBLEMA: Nenhum dispositivo de vídeo!")
    println("\n🔧 Soluções:")
    println("   1. Conecte uma webcam USB")
    println("   2. Para Iriun:")
    println("      - Baixe app no celular")
    println("      - Instale driver: wget http://iriun.com/downloads/iriun-webcam-linux-2.8.2.deb")
    println("      - sudo dpkg -i iriun-webcam-linux-2.8.2.deb")
    println("      - Inicie app no celular e conecte")
    println("   3. Verifique permissões:")
    println("      - sudo usermod -a -G video \$USER")
    println("      - Faça logout/login")
else
    println("\n✅ Dispositivos encontrados!")
    println("\n📸 Para capturar imagens:")
    println("\n   OPÇÃO A - Python (mais confiável):")
    println("      pip3 install opencv-python")
    println("      python3 capture_opencv.py --list")
    println("      python3 capture_opencv.py --multiple \"Nome\" ../dados/fotos_train 15 --camera 0")
    println("\n   OPÇÃO B - Julia:")
    println("      julia cnncheckin_capture.jl --cameras")
    println("      julia cnncheckin_capture.jl --train \"Nome\" 15")
    println("\n🎓 Para treinar:")
    println("      julia cnncheckin_pretrain_webcam.jl --no-capture")
end

println("\n" * "="^60)