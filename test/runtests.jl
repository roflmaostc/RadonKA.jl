using RadonKA
using Test
using FiniteDifferences
using ChainRulesTestUtils
using Zygote

@testset "RadonKA.jl" begin

    @testset "Simple iradon test" begin
        sinogram = zeros(Float32, (9, 2, 1))
	    sinogram[5, :, 1] .= 1
        @test iradon(sinogram, [0.0f0, π * 0.5]) ≈ [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 1.0 1.0 1.0 1.0 2.0 1.0 1.0 1.0 0.5; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0;;;] 

        @test iradon(sinogram[:, 1, :], Float32[π / 4 + π]) ≈ [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.46446568 6.7434956f-7 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.4142132 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.4142135 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.4142132 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 1.4142137 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.4142133 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.4142135 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;;;] 

    end

    @testset "Exponential iradon" begin
	    sinogram = zeros(Float64, (9, 1))
	    sinogram[5, :] .= 1


        angles = [0]
        arr = iradon(sinogram, angles, 0.1)

        exp.(-(8.5:-1:1.5) .* 0.1) ≈ arr[6, begin+1:end-1]
    end


    @testset "Compare with theoretical radon" begin

        array = zeros((16, 16,1))
	    array[12,9] = 1
   
        angles = range(0, 2pi, 100)
        sg = radon(array, angles)
        sg2 = radon(array[:, :, 1], angles)
        @test sg[:,:, 1] ≈ sg2
    	theory = zeros(size(sg))

    	i = 1
    	for θ in angles
    		cc = size(sg, 1) ÷ 2 + 1
    		x,y = (11, 8) .- (cc)
    	
    		y,x = [cos(θ) sin(θ); -sin(θ) cos(θ)] * [x, y]
    
    		c =  cc + x + 1f-8
    
    		c1 = floor(Int, c)
    		c2 = ceil(Int, c)
    
    		theory[c1, i] += (c2 - c)
    		theory[c2, i] += (c - c1)
    		i += 1
    	end
    
        @test ≈(sg, theory, rtol=0.3)


    end


    @testset "Filtered Backprojection" begin
    	array3 = zeros((32, 32));
	    
	    array3[10:15, 10:11] .= 1
	
	    array3[10:12, 20:26] .= 1
        angles2 = range(0, π, 200);

        sinogram2 = radon(array3, angles2)
        array_filtered = filtered_backprojection(sinogram2, angles2)
        @test ≈(array_filtered / sum(array_filtered) .+ 0.1, array3 / sum(array3) .+ 0.1, rtol=0.05)
    end

    @testset "Test gradients" begin
        x = randn((10, 10))
        test_rrule(radon, x, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), nothing ⊢ ChainRulesTestUtils.NoTangent())
        y = radon(x, [0, π/4, π/2, 2π, 0.1, 0.00001])
        test_rrule(iradon, y, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), nothing ⊢ ChainRulesTestUtils.NoTangent())
        
        x = randn((10, 10))
        test_rrule(radon, x, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), 0.01⊢ ChainRulesTestUtils.NoTangent())
        y = radon(x, [0, π/4, π/2, 2π, 0.1, 0.00001])
        test_rrule(iradon, y, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), 0.01 ⊢ ChainRulesTestUtils.NoTangent())
    end

end
