module RadonKA

using KernelAbstractions
using IndexFunArrays
using FFTW
using Atomix
using PrecompileTools

include("utils.jl")
include("radon.jl")
include("iradon.jl")




@setup_workload begin
    img = randn((2,2))
    angles = range(0, π, 2)

    @compile_workload begin
        r = radon(Float32.(img), Float32.(angles)) 
        iradon(r, Float32.(angles)) 
        RadonKA.filtered_backprojection(r, angles)
        r = radon(img, angles) 
        iradon(r, angles) 
        RadonKA.filtered_backprojection(r, angles)
        
        r = radon(Float32.(img), Float32.(angles), 0.1f0) 
        iradon(r, Float32.(angles), 0.1f0) 
        r = radon(img, angles, 0.1) 
        iradon(r, angles, 0.1) 
    end
end

end
