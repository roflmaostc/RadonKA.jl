# RadonKA.jl
A simple yet sufficiently fast Radon and adjoint Radon (backproject) transform implementation using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
It offers multithreading and CUDA support and outperforms any existing Julia Radon transforms (at least the ones we are aware of). 
On CUDA it is faster much than Matlab and it offers the same or faster speed than ASTRA.

#### ⚠️ This package is still very young. I would be happy to receive any feedback or if we can improve anything, just open an issue! ⚠️

<a  href="docs/src/assets/logo.png"><img src="docs/src/assets/logo.png"  width="200"></a>

[![Build Status](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/roflmaostc/RadonKA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/roflmaostc/RadonKA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/roflmaostc/RadonKA.jl) [![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://roflmaostc.github.io/RadonKA.jl/stable) [![Documentation for development version](https://img.shields.io/badge/docs-main-blue.svg)](https://roflmaostc.github.io/RadonKA.jl/dev)


# Quick Overview
* [x] For 2D and 3D arrays 
* [x] parallel `radon` and `backproject` (`?RadonParallelCircle`)
* [x] attenuated `radon` and `backproject` (see the parameter `μ`) and see this [paper](https://iopscience.iop.org/article/10.1088/0266-5611/17/1/309/meta) as reference)
* [x] arbitrary 2D geometries where starting and endpoint of each ray can be specified (fan beam could be a special case of this) (`?RadonFlexibleCircle`)
* [x] different strength weighting of rays 
* [x] based on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) (tested on `CPU()` and `CUDABackend()`)
* [x] registered adjoint rules for both `radon` and `backproject` with ChainRulesCore.jl, hence automatic differentiation (AD) compatible.
* [x] high performance however not ultra high performance. On par with ASTRA, on CUDA faster than Matlab.
* [x] simple and extensible API

# Installation
This toolbox runs with CUDA support on Linux, Windows and MacOS!
Requires at least Julia 1.9
```julia
julia> ]add RadonKA
```

# Simple use
```julia
using RadonKA, ImageShow, ImageIO, TestImages

img = Float32.(testimage("resolution_test_512"))
angles = range(0f0, 2f0π, 500)[begin:end-1]

# 0.085398 seconds (260 allocations: 1.006 MiB)
@time sinogram = radon(img, angles);
# 0.127043 seconds (251 allocations: 1.036 MiB)
@time backproject = RadonKA.backproject(sinogram, angles);

simshow(sinogram)
simshow(backproject)

using CUDA
img_c = CuArray(img)
# 0.003363 seconds (244 CPU allocations: 18.047 KiB) (7 GPU allocations: 1007.934 KiB, 0.96% memmgmt time)
CUDA.@time sinogram = radon(img_c, angles);
# 0.005928 seconds (218 CPU allocations: 16.109 KiB) (7 GPU allocations: 1.012 MiB, 0.49% memmgmt time)
CUDA.@time backproject = RadonKA.backproject(sinogram, angles);
```
<a  href="docs/src/assets/sinogram.png"><img src="docs/src/assets/sinogram.png"  width="300"></a>
<a  href="docs/src/assets/radonka_backproject.png"><img src="docs/src/assets/radonka_backproject.png"  width="308"></a>

# Examples
See the [documentation](https://roflmaostc.github.io/RadonKA.jl/dev/tutorial).
You can also run the examples locally.
Download this repository and then do the following in your REPL:
```julia
julia> cd("examples/")

julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
  Activating project at `~/.julia/dev/RadonKA.jl/examples`

julia> using Pluto; Pluto.run()
```
A browser should open.
The following examples show case the ability of this package:
* Simple `radon` and `backproject`: [Pluto notebook](examples/0_example_radon_backproject.jl)
* Different geometries: [Pluto notebook](examples/0_example_radon_backproject.jl)
* Reconstruction of a CT dataset with an optimizer: [Pluto notebook](examples/2_CT_with_optimizer.jl)
* How this package is used in Tomographic Volumetric Additive Manufacturing (3D printing): [Pluto notebook](examples/4_Tomographic_Volumetric_Additive_Manufacturing_with_Refraction.jl)

# Citation
This package was created as part of scientific work. Please consider citing it :)
```bibtex
@misc{wechsler2024wave,
      title={Wave optical model for tomographic volumetric additive manufacturing}, 
      author={Felix Wechsler and Carlo Gigli and Jorge Madrid-Wolff and Christophe Moser},
      year={2024},
      eprint={2402.06283},
      archivePrefix={arXiv},
      primaryClass={physics.optics}
}
```

# Development
File an [issue](https://github.com/roflmaostc/RadonKA.jl/issues) on [GitHub](https://github.com/roflmaostc/RadonKA.jl) if you encounter any problems.
You can also join [my conference room](https://epfl.zoom.us/my/wechsler). Give me a minute to join!

# Similar packages

## Python
There is [TIGRE](https://github.com/CERN/TIGRE) and [ASTRA](https://github.com/astra-toolbox/astra-toolbox) which both offer more functionality for classic CT problems.
They also feature GPU acceleration, however we did not observe that they outperform this package. Also, they don't allow to calculate the attenuated Radon transform
and don't allow for arbitrary ray geometries, as we do.
The fastest implementation we found, is the [unmaintained torch-radon](https://github.com/matteo-ronchetti/torch-radon). Its kernels are written in CUDA C code and offer a PyTorch interface.
There is a [torch-radon fork](https://github.com/carterbox/torch-radon) which allows to run it with newer versions. It offers no attenuated Radon transform.


## Julia
There is [Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl) and the [JuliaImageRecon](https://github.com/JuliaImageRecon) organization.
No arbitrary geometries can be specified. And also no attenuated Radon transform is possible.

## Matlab
Matlab has built-in a `radon` and `iradon(...,'linear','none');` transform which is similar to our lightweight API. However, no CUDA acceleration, no 3D arrays and no attenuated Radon transform.
