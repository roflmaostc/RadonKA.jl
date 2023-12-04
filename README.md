# RadonKA.jl
A simple but still decently fast Radon and inverse Radon (iradon) transform based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

[![Build Status](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/roflmaostc/RadonKA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/roflmaostc/RadonKA.jl) [![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://roflmaostc.github.io/RadonKA.jl/stable) [![Documentation for development version](https://img.shields.io/badge/docs-main-blue.svg)](https://roflmaostc.github.io/RadonKA.jl/dev)


# Quick Overview
* [x] For 2D and 3D arrays 
* [x] parallel `radon` and `iradon`
* [x] parallel exponential `radon` and `iradon`
* [x] It is restricted to the incircle of radius `N ÷ 2 - 1` if the array has size `(N, N, N_z)`
* [x] based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
* [x] tested on `CPU()` and `CUDABackend`

# Installation
Requires Julia 1.9
```julia
julia> ]add https://github.com/roflmaostc/RadonKA.jl
```


# Simple use
```julia
julia> using RadonKA

julia> arr = zeros((4,4)); arr[3,3] = 1;

julia> radon(arr, [0, π/4, π/2])
3×3 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0      0.0
 1.0  1.41421  1.0
 0.0  0.0      0.0


julia> arr = zeros((5,2)); arr[2,:] .= 1; arr[4, :] .= 1
  2-element view(::Matrix{Float64}, 4, :) with eltype Float64:
   1.0
   1.0
  
julia> iradon(arr, [0, π/2])
  6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
   0.0  0.0  0.0        0.0  0.0        0.0
   0.0  0.0  0.1        0.0  0.1        0.0
   0.0  0.1  0.2        0.1  0.2        0.0232051
   0.0  0.0  0.1        0.0  0.1        0.0
   0.0  0.1  0.2        0.1  0.2        0.0232051
   0.0  0.0  0.0232051  0.0  0.0232051  0.0
  
julia> iradon(arr, [0, π/2], 1) # exponential
  6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
   0.0  0.0         0.0         0.0        0.0         0.0
   0.0  0.0         0.00145226  0.0        0.00145226  0.0
   0.0  0.00145226  0.00789529  0.0107308  0.033117    0.0183994
   0.0  0.0         0.0107308   0.0        0.0107308   0.0
   0.0  0.00145226  0.033117    0.0107308  0.0583388   0.0183994
   0.0  0.0         0.0183994   0.0        0.0183994   0.0
```

# Examples
See either the [documentation](https://roflmaostc.github.io/RadonKA.jl/dev/tutorial).
Otherwise, this [example](https://github.com/roflmaostc/RadonKA.jl/blob/main/examples/example_radon_iradon.jl) shows the main features, including CUDA support.
There is one tutorial about [Least Square Optimization](https://github.com/roflmaostc/RadonKA.jl/blob/main/examples/CT_reconstruction.jl).
Another one cover how the Radon transform is used in [Volumetric Additive Manufacturing](https://github.com/roflmaostc/RadonKA.jl/blob/main/examples/volumetric_printing.jl).

# Development
File an [issue](https://github.com/roflmaostc/RadonKA.jl/issues) on [GitHub](https://github.com/roflmaostc/RadonKA.jl) if you encounter any problems.
