# RadonKA.jl
A simple yet sufficiently fast Radon and inverse Radon (iradon) transform implementation using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

#### ⚠️ This package is still very young. I would be happy to receive any feedback or if we can improve anything, just open an issue! ⚠️


<a  href="docs/src/assets/RadonKA_logo.png"><img src="docs/src/assets/RadonKA_logo.png"  width="200"></a>

[![Build Status](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/roflmaostc/RadonKA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/roflmaostc/RadonKA.jl) [![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://roflmaostc.github.io/RadonKA.jl/stable) [![Documentation for development version](https://img.shields.io/badge/docs-main-blue.svg)](https://roflmaostc.github.io/RadonKA.jl/dev)


# Quick Overview
* [x] For 2D and 3D arrays 
* [x] parallel `radon` and `iradon` (`?RadonParallelCircle`)
* [x] attenuated `radon` and `iradon` (see the parameter `μ`) and see this [paper](https://iopscience.iop.org/article/10.1088/0266-5611/17/1/309/meta) as reference)
* [x] arbitrary 2D geometries where starting and endpoint of each ray can be specified (fan beam could be a special case if this) (`?RadonFlexibleCircle`)
* [x] based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
* [x] tested on `CPU()` and `CUDABackend()`
* [x] registered adjoint rules for both `radon` and `iradon`
* [x] high performance however not ultra high performance. On par with ASTRA, on CUDA faster than Matlab.
* [x] simple and extensible API

# Installation
Requires Julia at least 1.9
```julia
julia> ]add RadonKA
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
There is one tutorial about [Gradient Descent optimization](https://github.com/roflmaostc/RadonKA.jl/blob/main/examples/CT_with_optimizer.jl).
Another one covers how the Radon transform is used in [Tomographic Volumetric Additive Manufacturing](https://github.com/roflmaostc/RadonKA.jl/blob/main/examples/Tomographic_Volumetric_Additive_Manufacturing_with_Refraction.jl).
One notebook explains how you can use the arbitrary geometries. See [here](examples/documentation_different_geometries.jl).

# Development
File an [issue](https://github.com/roflmaostc/RadonKA.jl/issues) on [GitHub](https://github.com/roflmaostc/RadonKA.jl) if you encounter any problems.


# Similar packages

## Python
There is [TIGRE](https://github.com/CERN/TIGRE) and [ASTRA](https://github.com/astra-toolbox/astra-toolbox) which both offer more functionality for classic CT problems.
They also feature GPU acceleration, however we did not observe that they outperform this package. Also, they don't allow to calculate the attenuated Radon transform
and don't allow for arbitrary ray geometries, as we do.
The fastest implementation we found, is the [unmaintained torch-radon](https://github.com/matteo-ronchetti/torch-radon). Its kernels are written in CUDA C code and offer a PyTorch interface.
There is a [torch-radon fork](https://github.com/carterbox/torch-radon) which allows to run it with newer versions. It offers no attenuated Radon transform.


## Julia
There exists [Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl) and in general the whole [JuliaImageRecon](https://github.com/JuliaImageRecon) organization.
Again, no arbitrary geometries can be specified. And also no attenuated Radon transform is possible.

## Matlab
Matlab has built-in a `radon` and `iradon` transform which is similar to our lightweight API. However, no CUDA acceleration, no 3D arrays and no attenuated Radon transform.
