
using Pkg
Pkg.activate(".")
using Test
using Images
using FileIO

include("cnncheckin_core.jl")
using .CNNCheckinCore

@testset "Face Detection Integration" begin
    # Create a dummy image
    img = Gray.(rand(100, 100))
    img_path = "test_image_dummy.jpg"
    save(img_path, img)

    println("Testing preprocess_image with fallback (assuming no opencv installed or mock)...")
    
    # 1. Test standard call (should likely fallback if no opencv)
    processed = CNNCheckinCore.preprocess_image(img_path)
    @test processed !== nothing
    @test length(processed) == 1
    @test size(processed[1]) == (128, 128, 3) # Normalized size
    
    println("Processed image size: $(size(processed[1]))")
    
    # Clean up
    rm(img_path, force=true)
end
