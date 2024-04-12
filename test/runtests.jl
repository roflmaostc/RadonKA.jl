using RadonKA
using Test
using FiniteDifferences
using ChainRulesTestUtils
using Zygote

@testset "RadonKA.jl" begin
    @testset "Simple radon test" begin
        x = zeros((6,6)); x[4,3] = 1
        @test radon(x, [0, π / 2, π, (3 / 2) * π, 2π]) ≈ [0.0 0.0 0.0 0.0 0.0; 1.0 0.0 0.0 0.0 1.0; 0.0 1.0 0.0 1.0 0.0; 0.0 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0] 
        x = zeros((6,6)); x[4,4] = 1
        @test radon(x, [0, π / 2, π, (3 / 2) * π, 2π]) ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 1.0 1.0 1.0 1.0 1.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]
        x = zeros((6,6)); x[3,3] = 1
        @test radon(x, [0, π / 2, π, (3 / 2) * π, 2π]) ≈ [0.0 0.0 0.0 0.0 0.0; 1.0 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 0.0 0.0; 0.0 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]

        x = zeros((7,7)); x[4,4] = 1
        @test radon(x, [0, π / 2, π, (3 / 2) * π, 2π]) ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 1.0 1.0 1.0 1.0 1.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]
        x = zeros((7,7)); x[4,3] = 1
        @test radon(x, [0, π / 2, π, (3 / 2) * π, 2π]) ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 1.0 0.0 0.0 0.0 1.0; 0.0 1.0 0.0 1.0 0.0; 0.0 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0] 

        x = zeros((5,5));x[3,3] = 1
        @test radon(x, [0], geometry = RadonParallelCircle(5, (-1:0.25:1) .- 0.1)) ≈ [0.0; 0.0; 0.0; 1.0; 1.0; 1.0; 1.0; 0.0; 0.0;;]
        @test radon(x, [0], geometry = RadonParallelCircle(5, (-1:0.25:1) .+ 0.1)) ≈ [0.0; 0.0; 1.0; 1.0; 1.0; 1.0; 0.0; 0.0; 0.0;;]
        @test radon(x, [0], geometry = RadonParallelCircle(5, [0.499])) == [1.0;;]
        @test radon(x, [0], geometry = RadonParallelCircle(5, [-0.499])) == [1.0;;]
        @test radon(x, [0], geometry = RadonParallelCircle(5, [-0.5001])) == [0.0;;]
        @test radon(x, [0], geometry = RadonParallelCircle(5, [+0.5001])) == [0.0;;]
    end

    @testset "Simple backproject test" begin
        sinogram = zeros(Float32, (9, 2, 1))
	    sinogram[5, :, 1] .= 1
        @test backproject(sinogram, [0.0f0, π * 0.5]) ≈ Float32[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 1.0 1.0 1.0 1.0 2.0 1.0 1.0 1.0 1.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;;;] 


        @test backproject(sinogram[:, 1, :], Float32[π / 4 + π]) == Float32[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.96446574 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.4142138 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.4142135 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.4142132 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 1.4142137 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.4142133 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.3486991f-6 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0] 
    end
    
    @testset "Simple API test" begin
        array = ones((6,6))
        @test radon(array, [0]) ≈ [1.0; 3.7320508075688767; 5.0; 3.7320508075688767; 1.0;;]
        @test radon(array, [0], geometry = RadonParallelCircle(6, -2:2:2)) ≈ [1.0; 5.0; 1.0;;]
        @test radon(array, [π], geometry = RadonParallelCircle(6, -2:2:2)) ≈ [1.0; 5.0; 1.0;;]
        @test radon(array, [0], geometry=RadonParallelCircle(6, -2:2, [12, 10, 8, 7, -3])) ≈ radon(array, [0], geometry=RadonParallelCircle(6, -2:2)) .* [12, 10, 8, 7, -3]
        @test radon(array, [0], geometry=RadonFlexibleCircle(6, -2:2, -2:2)) ≈ radon(array, [0], geometry=RadonParallelCircle(6, -2:2))
        @test radon(array, [-π/8], geometry=RadonParallelCircle(6, -2:2)) ≈ radon(array, [π/8], geometry=RadonParallelCircle(6, -2:2))[end:-1:begin, :]
        @test radon(array, [π/8], geometry=RadonFlexibleCircle(6, -2:2, [0,0,0,0,0])) ≈ radon(array, [-π/8], geometry=RadonFlexibleCircle(6, -2:2, [0,0,0,0,0]))[end:-1:begin, :]
    
        @test radon(array, [0], geometry = RadonFlexibleCircle(6, -2:2, [0, 0, 0, 0, 0], [2, 1, 0, 1, 2])) ≈ [5.122499389946278; 3.8348232886767164; 0.0; 3.834823288676716; 5.122499389946278;;]


        sinogram = ones((3, 1))
        @test backproject(sinogram, [0], geometry = RadonFlexibleCircle(20, [0, 0, 0], [0, 0, 0])) ≈ [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]

        @test backproject(sinogram, [0], geometry = RadonFlexibleCircle(20, [-5, 0, 5], [0, 0, 0])) ≈ [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0848743343786964 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.8356890637711426 1.4134962068364105 0.8356890637711426 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0424371671893484 1.0 1.0424371671893484 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0424371671893484 1.0 1.0424371671893484 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.42219285693473696 0.620244310254611 1.0 0.620244310254611 0.42219285693473696 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.042437167189348 0.0 1.0 0.0 1.042437167189348 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0424371671893484 0.0 1.0 0.0 1.0424371671893484 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.008696650098325685 1.0337405170910223 0.0 1.0 0.0 1.0337405170910223 0.008696650098325685 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0424371671893484 0.0 0.0 1.0 0.0 0.0 1.0424371671893484 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.042437167189348 0.0 0.0 1.0 0.0 0.0 1.042437167189348 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0424371671893484 0.0 0.0 1.0 0.0 0.0 1.0424371671893484 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.6376376104512703 0.40479955673807827 0.0 0.0 1.0 0.0 0.0 0.40479955673807827 0.6376376104512703 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 1.0424371671893482 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0424371671893484 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 1.0424371671893482 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0424371671893484 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.22414140361486262 0.8182957635744856 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.8182957635744816 0.22414140361486629 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0424371671893482 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.042437167189348 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.5038252833980154 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.5038252833980154 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]

        @test ≈(sum(backproject(sinogram, [0], geometry=RadonFlexibleCircle(20, 0.01 .+[-1,0,1], 0.01 .+[-1,0,1]))) + 1, sum(backproject(sinogram, [0], geometry=RadonFlexibleCircle(20, 0.00 .+[-1,0,1], 0.00 .+[-1,0,1]))), rtol=0.001)
    end

    @testset "Exponential backproject" begin
	    sinogram = zeros(Float64, (9, 1))
	    sinogram[5, :] .= 1


        angles = [0]
        arr = backproject(sinogram, angles, μ=0.1)
        @test exp.(-(9:-1:1) .* 0.1) ≈ arr[begin+1:end, 6][:]
        
        angles = [π]
        arr = backproject(sinogram, angles, μ=0.1)
        @test exp.(-(1:1:8).* 0.1) ≈ arr[begin+1:end-1, 6][:]
    end

    @testset "Compare with array absorption" begin
	    sinogram = zeros(Float64, (9, 10))
	    sinogram[5, :] .= 1
        angles = range(0, 2π, 10)
        arr1 = backproject(sinogram, angles, μ=0.1)
        arr2 = backproject(sinogram, angles, μ=0.1 * ones((10, 10)))
        @test all(.≈(arr1, arr2, rtol=0.08))

        
        array = zeros((16, 16,1))
        arr1 = radon(array, angles, μ=0.1)
        arr2 = radon(array, angles, μ=0.1 * ones((10, 10)))
        @test all(.≈(arr1, arr2, rtol=0.09))

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
        array_filtered = backproject_filtered(sinogram2, angles2)
        @test ≈(array_filtered / sum(array_filtered) .+ 0.1, array3 / sum(array3) .+ 0.1, rtol=0.06)

        geometry = RadonParallelCircle(32, -15:0.1:15)
        sinogram2 = radon(array3, angles2; geometry)
        array_filtered = backproject_filtered(sinogram2, angles2; geometry)
        @test ≈(array_filtered[5:28, 5:28] / sum(array_filtered[5:28, 5:28]) .+ 0.1, array3[5:28, 5:28] / sum(array3[5:28, 5:28]) .+ 0.1, rtol=0.05)

        geometry = RadonParallelCircle(32, -15:0.1:15)
        sinogram2 = radon(array3, angles2; geometry)
        @test ≈(backproject_filtered(sinogram2, angles2; geometry, filter=ones((size(sinogram2, 1)))), backproject(sinogram2, angles2; geometry)) 
    end

    @testset "Test gradients" begin
        x1 = randn((10, 10, 1))
        x2 = randn((5,5,1)) 
        for x in [x1, x2]
            geometry = RadonParallelCircle(size(x, 1), -(size(x,1)-1)÷2:(size(x,1)-1)÷2) 
            geometry1 = RadonFlexibleCircle(size(x, 1), -(size(x,1)-1)÷2:(size(x,1)-1)÷2, 0 .* (-(size(x,1)-1)÷2:(size(x,1)-1)÷2), (-(size(x,1)-1)÷2:(size(x,1)-1)÷2)) 
            for geometry in [geometry, geometry1]
                test_rrule(RadonKA._radon, x, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), geometry ⊢ ChainRulesTestUtils.NoTangent(), nothing ⊢ ChainRulesTestUtils.NoTangent())
                test_rrule(RadonKA._radon, x, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), geometry ⊢ ChainRulesTestUtils.NoTangent(), randn(size(x)...) ⊢ ChainRulesTestUtils.NoTangent())

                y = radon(x, [0, π/4, π/2, 2π, 0.1, 0.00001])
                test_rrule(RadonKA._backproject, y, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), geometry ⊢ ChainRulesTestUtils.NoTangent(), nothing ⊢ ChainRulesTestUtils.NoTangent())
            
                test_rrule(RadonKA._radon, x, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), geometry ⊢ ChainRulesTestUtils.NoTangent(), 0.1⊢ ChainRulesTestUtils.NoTangent())
                y = radon(x, [0, π/4, π/2, 2π, 0.1, 0.00001])
                test_rrule(RadonKA._backproject, y, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), geometry ⊢ ChainRulesTestUtils.NoTangent(), randn(size(x)...) ⊢ ChainRulesTestUtils.NoTangent())
                test_rrule(RadonKA._backproject, y, [0, π/4, π/2, 2π, 0.1, 0.00001] ⊢ ChainRulesTestUtils.NoTangent(), geometry ⊢ ChainRulesTestUtils.NoTangent(), 0.1 ⊢ ChainRulesTestUtils.NoTangent())
            end
        end
    end

end
