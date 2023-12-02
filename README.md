# RadonKA
A simple but still decently fast Radon and IRadon transform based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

[![Build Status](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/roflmaostc/RadonKA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/roflmaostc/RadonKA.jl)


# Quick Overview
* For 2D and 3D arrays 
* parallel `radon` and `iradon`
* parallel exponential `radon` and `iradon`
* It is restricted to the incircle of radius `N ÷ 2 - 1` if the array has size `(N, N, N_z)`
* based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
* tested on `CPU()` and `CUDABackend`

# Installation
Requires Julia 1.9
```julia
julia> ]add https://github.com/roflmaostc/RadonKA.jl
```

# Development
File an [issue](https://github.com/roflmaostc/RadonKA.jl/issues) on [GitHub](https://github.com/roflmaostc/RadonKA.jl) if you encounter any problems.
