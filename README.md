# RadonKA
A simple but still decently fast radon and iradon transform based on KernelAbstractions.jl

[![Build Status](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/roflmaostc/RadonKA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/roflmaostc/RadonKA.jl)



# Quick Overview
* provides `radon` and `iradon`
* based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
* tested on `CPU()` and `CUDABackend`


# Installation
Requires Julia 1.9
```julia
julia> ]add https://github.com/roflmaostc/RadonKA.jl
```
