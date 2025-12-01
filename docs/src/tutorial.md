# Tutorial

We offer some examples on [GitHub](https://github.com/roflmaostc/RadonKA.jl/tree/main/examples). 
To run them, git clone the whole repository.
Then do:
```julia
(@v1.9) pkg> activate examples/
  Activating project at `~/.julia/dev/RadonKA.jl/examples`

julia> using Pluto; Pluto.run()
```

## Radon transform
The radon transform only requires the image (or volume) and some angles:
```julia
using RadonKA, ImageShow, ImageIO, TestImages

img = Float32.(testimage("resolution_test_512"))

angles = range(0f0, 2f0π, 500)[begin:end-1]

# 0.196049 seconds (145 allocations: 1009.938 KiB)
@time sinogram = radon(img, angles);
```
![](../assets/sinogram.png)

# adjoint Radon (backproject) transform
```julia
# 0.268649 seconds (147 allocations: 1.015 MiB)
@time backproject = RadonKA.backproject(sinogram, angles);

# in Pluto or Jupyter
simshow(sinogram)

[simshow(img) simshow(backproject)]
```
Left is the original sample and right the simple backprojection.
![](../assets/sample_backproject.png)

## Odd sized arrays
For the `backproject` of an odd-sized array is an ambiguity since we don't know 
whether the original array was even-sized or odd.
As default, we return an even-sized array.
But this can be changed with the following:
```
julia> x = randn((5,5));

julia> y = radon(x, range(0,2π, 5))
5×5 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 -0.428792   -2.37766   -0.0232206  -1.71798   -0.428792
  0.887089    2.63651   -0.342234    0.831      0.887089
 -1.62588    -0.519691   0.751784   -0.496471   0.0921012
 -0.40562    -0.508413   0.301532    3.64494   -0.40562
 -0.0232206  -1.71798   -0.428792   -2.37766   -0.0232206

julia> backproject(y, range(0, 2π, 5))
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0   0.0       0.0        0.0        0.0        0.0
 0.0   0.0       0.220737  -4.31006   -0.250533   0.0
 0.0   0.608334  2.3983    -0.459407  -0.830887  -0.372184
 0.0  -2.30254   1.05955   -1.79816   -2.16964   -0.589353
 0.0   2.66828   8.35716    5.49946    5.12798    1.93006
 0.0   0.0       1.29879   -5.53732   -0.593869   0.0

julia> backproject(y, range(0, 2π, 5), geometry=RadonParallelCircle(5, -size(y,1)÷2:size(y,1)÷2))
5×5 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
  0.0       0.220737  -4.31006   -0.250533   0.0
  0.608334  2.3983    -0.459407  -0.830887  -0.372184
 -2.30254   1.05955   -1.79816   -2.16964   -0.589353
  2.66828   8.35716    5.49946    5.12798    1.93006
  0.0       1.29879   -6.28911   -0.593869   0.0
```


# Filtered Backprojection
In the absence of noise, the filtered backprojection works well:
```julia
#   0.252915 seconds (171 allocations: 13.664 MiB)
@time filtered_backproject = RadonKA.backproject_filtered(sinogram, angles);
```
![](../assets/filtered.png)


# CUDA Support
RadonKA.jl supports CUDA and usually provides a 10-20x speedup on typical RTX 3xxx or 4xxx GPUs.
Pay attention that the element type of the array/img should be `Float32` for good speedup.
In my case we used a AMD Ryzen 5 5600X 6-Core Processor and a CUDA RTX 3060 Super.
```julia
using CUDA

img_c = CuArray(img);
angles_c = CuArray(angles);

# 0.005611 seconds (8.95 k CPU allocations: 363.828 KiB) (2 GPU allocations: 998.047 KiB, 0.26% memmgmt time)
CUDA.@time CUDA.@sync sinogram_c = radon(img_c, angles_c);
```


# 3D example
Simply attach a third trailing dimension to the array. The radon transform is computed along the first and second dimension.
The third dimension is just a *batch* dimension.
```julia
volume = randn(Float32,(512, 512, 512));
volume_c = CuArray(randn(Float32,(512, 512, 512)));

# 86.795338 seconds (153 allocations: 498.039 MiB, 0.02% gc time)
@time radon(volume, angles);

# 2.527153 seconds (8.95 k CPU allocations: 363.703 KiB) (2 GPU allocations: 498.027 MiB, 0.06% memmgmt time)
CUDA.@time CUDA.@sync radon(volume_c, angles_c);
```
