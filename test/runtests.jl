using RadonKA
using Test

@testset "RadonKA.jl" begin


    @testset "radon without absorption" begin

        array = zeros((16, 16,1))
	    array[12,9] = 1
   
        angles = range(0, 2pi, 100)
        sg = radon(array, angles)
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

end
