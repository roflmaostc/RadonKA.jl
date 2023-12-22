# RadonKA
A simple but still decently fast Radon and IRadon transform based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

[![Build Status](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/roflmaostc/RadonKA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/roflmaostc/RadonKA.jl) [![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://roflmaostc.github.io/RadonKA.jl/stable) [![Documentation for development version](https://img.shields.io/badge/docs-main-blue.svg)](https://roflmaostc.github.io/RadonKA.jl/dev)


# Quick Overview
* [x] For 2D and 3D arrays 
* [x] parallel `radon` and `iradon`
* [x] parallel exponential `radon` and `iradon`
* [x] It is restricted to the incircle of radius `N ÷ 2 - 1` if the array has size `(N, N, N_z)`
* [x] based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
* [x] tested on `CPU()` and `CUDABackend`
* [x] registered adjoint rules for both `radon` and `iradon`

# Installation
Requires Julia 1.9
```julia
julia> ]add RadonKA
```

# Development
File an [issue](https://github.com/roflmaostc/RadonKA.jl/issues) on [GitHub](https://github.com/roflmaostc/RadonKA.jl) if you encounter any problems.
