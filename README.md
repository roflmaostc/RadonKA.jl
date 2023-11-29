# RadonKA
A simple but still decently fast Radon and IRadon transform based on KernelAbstractions.jl

[![Build Status](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/roflmaostc/RadonKA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/roflmaostc/RadonKA.jl)


# Quick Overview
* For 3D arrays (2D arrays are special case of 3D).
* provides parallel `radon` and `iradon`
* provides parallel exponential `radon` and `iradon`
* It is restricted to the incircle of radius `N ÷ 2 - 1` if the array has size `(N, N, N_z)`
* based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
* tested on `CPU()` and `CUDABackend`
* Can be applied to Computed Tomography (CT) like problems


# Installation
Requires Julia 1.9
```julia
julia> ]add https://github.com/roflmaostc/RadonKA.jl
```
