# RadonKA.jl
A simple but still decently fast Radon and inverse Radon (iradon) transform based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

<a  href="docs/src/assets/RadonKA_logo.png"><img src="docs/src/assets/RadonKA_logo.png"  width="200"></a>

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
using RadonKA, ImageShow, ImageIO, TestImages

img = Float32.(testimage("resolution_test_512"))

angles = range(0f0, 2f0π, 500)[begin:end-1]

# 0.196049 seconds (145 allocations: 1009.938 KiB)
@time sinogram = radon(img, angles);

# 0.268649 seconds (147 allocations: 1.015 MiB)
@time backproject = RadonKA.iradon(sinogram, angles);

simshow(sinogram)
simshow(backproject)
```
<a  href="docs/src/assets/sinogram.png"><img src="docs/src/assets/sinogram.png"  width="300"></a>
<a  href="docs/src/assets/radonka_iradon.png"><img src="docs/src/assets/radonka_iradon.png"  width="308"></a>

# Examples
See either the [documentation](https://roflmaostc.github.io/RadonKA.jl/dev/tutorial).
Otherwise, this [example](https://github.com/roflmaostc/RadonKA.jl/blob/main/examples/example_radon_iradon.jl) shows the main features, including CUDA support.
There is one tutorial about [Least Square Optimization](https://github.com/roflmaostc/RadonKA.jl/blob/main/examples/CT_reconstruction.jl).
Another one cover how the Radon transform is used in [Volumetric Additive Manufacturing](https://github.com/roflmaostc/RadonKA.jl/blob/main/examples/volumetric_printing.jl).

# Development
File an [issue](https://github.com/roflmaostc/RadonKA.jl/issues) on [GitHub](https://github.com/roflmaostc/RadonKA.jl) if you encounter any problems.
