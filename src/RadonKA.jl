module RadonKA

using KernelAbstractions
using IndexFunArrays
using FFTW
using Atomix
using ChainRulesCore
using PrecompileTools

include("geometry.jl")
include("utils.jl")
include("radon.jl")
include("backproject.jl")
include("backproject_filtered.jl")


# PrecompileTools
#@setup_workload begin
#    img = randn((2,2))
#    angles = range(0, π, 2)
#
#    @compile_workload begin
#        r = radon(Float32.(img), Float32.(angles)) 
#        backproject(r, Float32.(angles)) 
#        RadonKA.backproject_filtered(r, angles)
#        r = radon(img, angles) 
#        backproject(r, angles) 
#        RadonKA.backproject_filtered(r, angles)
#        
#        r = radon(Float32.(img), Float32.(angles), μ=0.1f0) 
#        backproject(r, Float32.(angles), μ=0.1f0) 
#        r = radon(img, angles, μ=0.1) 
#        backproject(r, angles, μ=0.1) 
#    end
#end

end
