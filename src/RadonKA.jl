module RadonKA

using KernelAbstractions
using IndexFunArrays
using FFTW
using Atomix
using ChainRulesCore
using PrecompileTools


include("utils.jl")
include("radon.jl")
include("iradon.jl")
include("radon_new.jl")
include("iradon_new.jl")


export RadonGeometry, RadonParallelCircle

abstract type RadonGeometry end

"""
    RadonParallelCircle(N, in_height)

`N` is the size of the first and second dimension of the array.
`in_height` is a vector or range indicating the positions of the detector
with respect to the midpoint which is located at `N ÷ 2 + 1`.

So an array of size `N=10` the default definition is: `RadonParallelCircle(10, -4:4)`
So the resulting sinogram has the shape: `(9, length(angles), size(array, 3))`
"""
struct RadonParallelCircle{T} <: RadonGeometry
    N::Int
    in_height::AbstractVector{T}
end

struct RadonFlexibleCircle{T} <: RadonGeometry
    N::Int
    in_height::AbstractVector{T}
    out_height::AbstractVector{T}
end


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
