var documenterSearchIndex = {"docs":
[{"location":"functions/#Radon","page":"Function Docstrings","title":"Radon","text":"","category":"section"},{"location":"functions/","page":"Function Docstrings","title":"Function Docstrings","text":"radon","category":"page"},{"location":"functions/#RadonKA.radon","page":"Function Docstrings","title":"RadonKA.radon","text":"radon(I, θs; <kwargs>)\n\nCalculates the parallel Radon transform of the AbstractArray I. Intuitively, the radon sums array entries  of I along ray paths.\n\nThe first two dimensions are y and x. The third dimension is z, the rotational axis. size(I, 1) and size(I, 2) have to be equal. The Radon transform is rotated around the pixel size(I, 1) ÷ 2 + 1, so there is always an integer center pixel! Works either with a AbstractArray{T, 3} or AbstractArray{T, 2}.\n\nθs is a vector or range storing the angles in radians.\n\nIn principle, all backends of KernelAbstractions.jl should work but are not tested. CUDA and CPU arrays are actively tested. Both radon and backproject are differentiable with respect to I.\n\nKeywords\n\nμ=nothing - Attenuated Radon Transform\n\nIf μ != nothing, then the rays are attenuated with exp(-μ * dist) where dist  is the distance to the circular boundary of the field of view. μ is in units of pixel length. So μ=1 corresponds to an attenuation of exp(-1) if propagated through one pixel. If isnothing(μ), then the rays are not attenuated.\n\ngeometry = RadonParallelCircle(-(size(img,1)-1)÷2:(size(img,1)-1)÷2)\n\nThis corresponds to a parallel Radon transform.  See ?RadonGeometries for a full list of geometries. There is also the very flexible RadonFlexibleCircle.\n\nSee also backproject.\n\nExample\n\nThe reason the sinogram has the value 1.41421 for the diagonal ray π/4 is, that such a diagonal travels a longer distance through the pixel.\n\njulia> arr = zeros((4,4)); arr[3,3] = 1;\n\njulia> radon(arr, [0, π/4, π/2])\n3×3 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:\n 0.0  0.0      0.0\n 1.0  1.41421  1.0\n 0.0  0.0      0.0\n\nChoose different detector\n\njulia> array = ones((6,6))\n6×6 Matrix{Float64}:\n 1.0  1.0  1.0  1.0  1.0  1.0\n 1.0  1.0  1.0  1.0  1.0  1.0\n 1.0  1.0  1.0  1.0  1.0  1.0\n 1.0  1.0  1.0  1.0  1.0  1.0\n 1.0  1.0  1.0  1.0  1.0  1.0\n 1.0  1.0  1.0  1.0  1.0  1.0\n\njulia> radon(array, [0])\n5×1 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:\n 1.0\n 3.7320508075688767\n 5.0\n 3.7320508075688767\n 1.0\n\njulia> radon(array, [0], geometry=RadonParallelCircle(6, -2:2))\n5×1 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:\n 1.0\n 3.7320508075688767\n 5.0\n 3.7320508075688767\n 1.0\n\njulia> radon(array, [0], geometry=RadonParallelCircle(6, -2:2:2))\n3×1 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:\n 1.0\n 5.0\n 1.0\n\nApply some weights on the rays\n\njulia> array = ones((6,6));\n\njulia> radon(array, [0], geometry=RadonParallelCircle(6, -2:2, [2,1,0,1,2]))\n5×1 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:\n 2.0\n 3.7320508075688767\n 0.0\n 3.7320508075688767\n 2.0\n\n\n\n\n\n","category":"function"},{"location":"functions/#Backproject-(adjoint-Radon)","page":"Function Docstrings","title":"Backproject (adjoint Radon)","text":"","category":"section"},{"location":"functions/","page":"Function Docstrings","title":"Function Docstrings","text":"backproject\nbackproject_filtered","category":"page"},{"location":"functions/#RadonKA.backproject","page":"Function Docstrings","title":"RadonKA.backproject","text":"backproject(sinogram, θs; <kwargs>)\n\nConceptually the adjoint operation of radon. Intuitively, the backproject smears rays back into the space. See also radon.\n\nFor filtered backprojection see backproject_filtered.\n\nExample\n\njulia> arr = zeros((5,2)); arr[2,:] .= 1; arr[4, :] .= 1\n2-element view(::Matrix{Float64}, 4, :) with eltype Float64:\n 1.0\n 1.0\n\njulia> backproject(arr, [0, π/2])\n6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:\n 0.0  0.0  0.0       0.0  0.0       0.0\n 0.0  0.0  0.0       0.0  0.0       0.0\n 0.0  0.0  2.0       1.0  2.0       0.732051\n 0.0  0.0  1.0       0.0  1.0       0.0\n 0.0  0.0  2.0       1.0  2.0       0.732051\n 0.0  0.0  0.732051  0.0  0.732051  0.0\n\njulia> arr = ones((2,1)); \n\njulia> backproject(arr, [0], geometry=RadonFlexibleCircle(10, [-3, 3], [0,0]))\n10×10 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:\n 0.0  0.0  0.0       0.0       0.0        0.0      0.0        0.0       0.0       0.0\n 0.0  0.0  0.0       0.0       0.0        0.0      0.0        0.0       0.0       0.0\n 0.0  0.0  0.0       0.0       0.335172   1.49876  0.335172   0.0       0.0       0.0\n 0.0  0.0  0.0       0.0       1.08455    0.0      1.08455    0.0       0.0       0.0\n 0.0  0.0  0.0       0.0       1.08455    0.0      1.08455    0.0       0.0       0.0\n 0.0  0.0  0.0       1.00552   0.0790376  0.0      0.0790376  1.00552   0.0       0.0\n 0.0  0.0  0.0       1.08455   0.0        0.0      0.0        1.08455   0.0       0.0\n 0.0  0.0  0.591307  0.493247  0.0        0.0      0.0        0.493247  0.591307  0.0\n 0.0  0.0  0.700352  0.0       0.0        0.0      0.0        0.0       0.700352  0.0\n 0.0  0.0  0.0       0.0       0.0        0.0      0.0        0.0       0.0       0.0\n\n\n\n\n\n","category":"function"},{"location":"functions/#RadonKA.backproject_filtered","page":"Function Docstrings","title":"RadonKA.backproject_filtered","text":"backproject_filtered(sinogram, θs; \n                        geometry, μ, filter)\n\nReconstruct the image from the sinogram using the filtered backprojection algorithm.\n\nfilter=nothing: The filter to be applied in Fourier space. If nothing, a ramp filter is used. filter should be a 1D array of the same length as the sinogram.\n\nSee radon for the explanation of the keyword arguments\n\n\n\n\n\n","category":"function"},{"location":"functions/#Specifying-Geometries","page":"Function Docstrings","title":"Specifying Geometries","text":"","category":"section"},{"location":"functions/","page":"Function Docstrings","title":"Function Docstrings","text":"RadonGeometry\nRadonParallelCircle\nRadonFlexibleCircle","category":"page"},{"location":"functions/#RadonKA.RadonGeometry","page":"Function Docstrings","title":"RadonKA.RadonGeometry","text":"abstract type RadonGeometry end\n\nAbstract supertype for all geometries which are supported by radon and backproject.\n\nList of geometries:\n\nRadonParallelCircle\nRadonFlexibleCircle\n\nSee radon and backproject how to apply.\n\n\n\n\n\n","category":"type"},{"location":"functions/#RadonKA.RadonParallelCircle","page":"Function Docstrings","title":"RadonKA.RadonParallelCircle","text":"RadonParallelCircle(N, in_height, weights)\n\nN is the size of the first and second dimension of the input array for radon.\nin_height is a vector or a range indicating the positions of the detector with respect to the midpoint which is located at N ÷ 2 + 1. The rays travel along straight parallel paths through the array.\n\nFor example, for an array of size N=10 the default definition is: RadonParallelCircle(10, -4:4) So the resulting sinogram has the shape: (9, length(angles), size(array, 3)).\n\nweights can weight each of the rays with different strength. Per default weights = 0 .* in_height .+ 1\n\nSee radon and backproject how to apply.\n\n\n\n\n\n","category":"type"},{"location":"functions/#RadonKA.RadonFlexibleCircle","page":"Function Docstrings","title":"RadonKA.RadonFlexibleCircle","text":"RadonFlexibleCircle(N, in_height, out_height, weights)\n\nN is the size of the first and second dimension of the input for radon.\nin_height is a vector or range indicating the vertical positions of the rays entering the circle with respect to the midpoint which is located at N ÷ 2 + 1.\nout_height is a vector or range indicating the vertical positions of the rays exiting the circle with respect to the midpoint which is located at N ÷ 2 + 1.\n\nOne definition could be: RadonFlexibleCircle(10, -4:4, zeros((9,))) It would describe rays which enter the circle at positions -4:4 but all of them would focus at the position 0 when leaving the circle. This is an extreme form of fan beam tomography.\n\nweights can weight each of the rays with different strength. Per default weights = 0 .* in_height .+ 1\n\nSee radon and backproject how to apply.\n\n\n\n\n\n","category":"type"},{"location":"geometries/#Specify-Different-Geometries-and-Absorption","page":"Specifying different geometries and absorption","title":"Specify Different Geometries and Absorption","text":"","category":"section"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"RadonKA.jl offers two different geometries right now. See also this pluto notebook","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"The simple and default interface is RadonParallelCircle. This traces the rays parallel through the volume. However, only rays inside a circle of the image are considered.","category":"page"},{"location":"geometries/#RadonParallelCircle","page":"Specifying different geometries and absorption","title":"RadonParallelCircle","text":"","category":"section"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"See also the RadonParallelCircle docstring. The essence is the specification of a range or vector where the incoming position of a ray is. This is with respect to the center pixel at div(N, 2) +1.","category":"page"},{"location":"geometries/#Parallel","page":"Specifying different geometries and absorption","title":"Parallel","text":"","category":"section"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"The first example is the default. Just a parallel ray geometry.","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"angles = [0]\n\n# output image size\nN = 200\n\nsinogram = zeros((N - 1, length(angles)))\nsinogram[1:5:end] .= 1\n\ngeometry_parallel = RadonParallelCircle(N, -(N-1)÷2:(N-1)÷2)\n\nprojection_parallel = backproject(sinogram, angles; geometry=geometry_parallel);\n\nsimshow(projection_parallel)","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"(Image: )","category":"page"},{"location":"geometries/#Parallel-Small","page":"Specifying different geometries and absorption","title":"Parallel Small","text":"","category":"section"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"sinogram_small = zeros((99, length(angles)))\nsinogram_small[1:3:end] .= 1\n\ngeometry_small = RadonParallelCircle(200, -49:49)\n\nprojection_small = backproject(sinogram_small, angles; geometry=geometry_small);\n\nsimshow(projection_small)","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"(Image: )","category":"page"},{"location":"geometries/#RadonFlexibleCircle","page":"Specifying different geometries and absorption","title":"RadonFlexibleCircle","text":"","category":"section"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"See also the RadonFlexibleCircle docstring. This interface has a simple API but is quite powerful. The first range indicates the position upon entrance in the circle. The second range indicates the position upon exit of the circle.","category":"page"},{"location":"geometries/#fan-Beam","page":"Specifying different geometries and absorption","title":"fan Beam","text":"","category":"section"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"geometry_fan = RadonFlexibleCircle(N, -(N-1)÷2:(N-1)÷2, range(-(N-1)÷4, (N-1)÷4, N-1))\n\nprojected_fan = backproject(sinogram, angles; geometry=geometry_fan);\n\nsimshow(projected_fan, γ=0.01)","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"(Image: )","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"geometry_extreme = RadonFlexibleCircle(N, -(N-1)÷2:(N-1)÷2, zeros((199,)))\n\nprojected_extreme = backproject(sinogram, angles; geometry=geometry_extreme);\n\nsimshow(projected_extreme, γ=0.01)","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"(Image: )","category":"page"},{"location":"geometries/#Using-Different-weighting","page":"Specifying different geometries and absorption","title":"Using Different weighting","text":"","category":"section"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"For example, if in your application some rays are stronger than others you can include weight factor array into the API.","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"geometry_weight = RadonParallelCircle(N, -(N-1)÷2:(N-1)÷2, abs.(-(N-1)÷2:(N-1)÷2))\nprojection_weight = backproject(sinogram, angles; geometry=geometry_weight);\n\nsimshow(projection_weight)","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"(Image: )","category":"page"},{"location":"geometries/#Absorption","page":"Specifying different geometries and absorption","title":"Absorption","text":"","category":"section"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"The ray gets some attenuation with exp(-μ*x) where x is the distance traveled to the entry point of the circle. μ is in units of pixel.","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"projected_exp = backproject(sinogram, angles; geometry=geometry_extreme, μ=0.04);\n\nsimshow(projected_exp)","category":"page"},{"location":"geometries/","page":"Specifying different geometries and absorption","title":"Specifying different geometries and absorption","text":"(Image: )","category":"page"},{"location":"benchmark/#Benchmark-and-Comparison-with-Matlab-and-Astra","page":"Benchmark with Matlab and Astra","title":"Benchmark and Comparison with Matlab and Astra","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":"Tested on a AMD Ryzen 9 5900X 12-Core Processor with 24 Threads and a NVIDIA GeForce RTX 3060 with Julia 1.10.0 on Ubuntu 22.04.","category":"page"},{"location":"benchmark/#Results","page":"Benchmark with Matlab and Astra","title":"Results","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":" RadonKA.jl CPU RadonKA.jl GPU Matlab CPU Astra CPU Astra GPU\n2D sample - Radon 1.1s 0.07s 0.39s 7.0s 0.025s\n2D sample - Backprojection 1.4s 0.50s 0.37s 6.4s N/A\n3D sample - Radon 7.4s 0.28s 9.01s N/A 1.12s\n3D sample - Backprojection 7.9s 0.53s 3.24s N/A N/A","category":"page"},{"location":"benchmark/#RadonKA.jl","page":"Benchmark with Matlab and Astra","title":"RadonKA.jl","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":" using IndexFunArrays, ImageShow, Plots, ImageIO, PlutoUI, PlutoTest, TestImages\n using RadonKA\n using CUDA, CUDA.CUDAKernels\n using BenchmarkTools\n \n sample_2D = Float32.(testimage(\"resolution_test_1920\"));\n sample_2D_c = CuArray(sample_2D);\n simshow(sample_2D)\n \n angles = range(0, 2π, 500);\n angles_c = CuArray(angles);\n \n # run those cells multiple times\n sinogram = radon(sample_2D, angles);\n @btime sinogram = radon($sample_2D, $angles);\n simshow(sinogram)\n @btime sample_backproject = backproject($sinogram, $angles);\n \n @btime CUDA.@sync sinogram_c = radon($sample_2D_c, $angles_c);\n sinogram_c = radon(sample_2D_c, angles_c);\n @btime CUDA.@sync sample_backproject_c = backproject($sinogram_c, $angles_c);\n \n \n sample_3D = randn(Float32, (512, 512, 100));\n sample_3D_c = CuArray(sample_3D)\n \n sinogram_3D = radon(sample_3D, angles);\n @btime radon($sample_3D, $angles)\n @btime backproject($sinogram_3D, $angles)\n \n sinogram_3D_c = radon(sample_3D_c, angles_c)\n @btime CUDA.@sync radon($sample_3D_c, $angles_c)\n @btime CUDA.@sync backproject($sinogram_3D_c, $angles_c)","category":"page"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":"(Image: ) (Image: )","category":"page"},{"location":"benchmark/#Matlab-(R2023a)","page":"Benchmark with Matlab and Astra","title":"Matlab (R2023a)","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":"arr = single(imread(\"/tmp/sample.jpg\")); \n%arr = rand(1920, 1920, \"single\");\ntheta = linspace(0, 360, 500);\n\ntic;\nR = radon(arr, theta);\ntoc;\nR = R / max(R(:));\nimwrite(R, \"/tmp/matlab_sinogram.png\");\n\ntic; \niR = backproject(R, theta, \"linear\", \"none\");\ntoc;\niR = iR / max(iR(:));\nimwrite(iR, \"/tmp/matlab_backproject.png\");\n\n\n\nx = (rand(100, 512, 512, 'single'));\n\ntheta = linspace(0, 360, 500);\ny = (zeros(size(x, 1), size(radon(squeeze(x(1, :, :)), theta), 1), size(radon(squeeze(x(1, :, :)), theta), 2), 'single'));\nix = zeros(100, 514, 514, \"single\");\n\ntic;\nfor i = 1:size(x, 1)\n    y(i, :, :) = radon(squeeze(x(i, :, :)), theta);\nend\ntoc;\n\n\ntic;\nfor i = 1:size(y, 1)\n    ix(i, :, :) = backproject(squeeze(y(i, :, :)), theta);\nend\ntoc;","category":"page"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":"(Image: ) (Image: )","category":"page"},{"location":"benchmark/#Astra","page":"Benchmark with Matlab and Astra","title":"Astra","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":"Some of the benchmarks did not run properly or were providing non-sense. Astra's docs are unfortunately slightly confusing...","category":"page"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":"import numpy as np\nimport matplotlib.pyplot as plt\nimport imageio.v3 as iio\nimport astra\n\nim = np.array(iio.imread('/tmp/sample.png'), dtype=np.float32)\n\nplt.imshow(im)\n\n\nvol_geom = astra.create_vol_geom(1920, 1920)\nangles =  np.linspace(0,2 * np.pi,500)\n\nproj_geom = astra.create_proj_geom('parallel', 1.0, 1920, angles)\nproj_id = astra.create_projector('line', proj_geom,vol_geom)\n\n\n%%time\nsinogram_id, sinogram = astra.create_sino(im, proj_id)\nnp.copy(sinogram);\n\nplt.imshow(sinogram)\n\nproj_geom = astra.create_proj_geom('parallel', 1.0, 1920, angles)\nproj_id = astra.create_projector('cuda', proj_geom,vol_geom)\n\n%%time\nsinogram_id, sinogram = astra.create_sino(im, proj_id)\nnp.copy(sinogram);\n\nrec_id = astra.data2d.create(\"-vol\", vol_geom)\ncfg = astra.astra_dict('BP')\ncfg[\"ReconstructionDataId\"] = rec_id\ncfg[\"ProjectionDataId\"] = sinogram_id\ncfg[\"ProjectorId\"] = proj_id\n# Create the algorithm object from the configuration structure\nalg_id = astra.algorithm.create(cfg)\n# Run back-projection and get the reconstruction\n\n%%time\nastra.algorithm.run(alg_id)\nf_rec = astra.data2d.get(rec_id)\n#np.copy(f_rec)\n\n\nim_3D = np.random.rand(100, 512, 512)\n\nvol_geom = astra.create_vol_geom(512, 512, 100)\n\nproj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 512, 512, angles)\n#proj_id = astra.create_projector('line', proj_geom,vol_geom)\n\n%%time\nproj_id, proj_data = astra.create_sino3d_gpu(im_3D, proj_geom, vol_geom)\nnp.copy(proj_data)\n\n\nrec_id = astra.data3d.create('-vol', vol_geom)\n\n# Set up the parameters for a reconstruction algorithm using the GPU\ncfg = astra.astra_dict('BP3D_CUDA')\ncfg['ReconstructionDataId'] = rec_id\ncfg['ProjectionDataId'] = proj_id\n\n\n# Create the algorithm object from the configuration structure\nalg_id = astra.algorithm.create(cfg)\n\n%time\nastra.algorithm.run(alg_id, 1)\nrec = astra.data3d.get(rec_id)\nnp.copy(rec)","category":"page"},{"location":"benchmark/","page":"Benchmark with Matlab and Astra","title":"Benchmark with Matlab and Astra","text":"(Image: ) (Image: )","category":"page"},{"location":"#RadonKA.jl","page":"RadonKA.jl","title":"RadonKA.jl","text":"","category":"section"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"A simple yet sufficiently fast Radon and adjoint Radon (backproject) transform implementation using KernelAbstractions.jl.","category":"page"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"<a  href=\"assets/RadonKA_logo.png\"><img src=\"assets/RadonKA_logo.png\"  width=\"200\"></a>","category":"page"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"(Image: Build Status) (Image: Coverage) (Image: Documentation for stable version) (Image: Documentation for development version)","category":"page"},{"location":"#Quick-Overview","page":"RadonKA.jl","title":"Quick Overview","text":"","category":"section"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"[x] For 2D and 3D arrays \n[x] parallel radon and backproject (?RadonParallelCircle)\n[x] attenuated radon and backproject (see the parameter μ)\n[x] arbitrary 2D geometries where starting and endpoint of each ray can be specified (fan beam could be a special case if this) (?RadonFlexibleCircle)\n[x] It is restricted to the incircle of radius N ÷ 2 - 1 if the array has size (N, N, N_z)\n[x] based on KernelAbstractions.jl\n[x] tested on CPU() and CUDABackend\n[x] registered adjoint rules for both radon and backproject\n[x] high performance however not ultra high performance\n[x] simple API","category":"page"},{"location":"#Installation","page":"RadonKA.jl","title":"Installation","text":"","category":"section"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"Requires Julia 1.9","category":"page"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"julia> ]add RadonKA","category":"page"},{"location":"#Simple-use","page":"RadonKA.jl","title":"Simple use","text":"","category":"section"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"using RadonKA, ImageShow, ImageIO, TestImages\n\nimg = Float32.(testimage(\"resolution_test_512\"))\n\nangles = range(0f0, 2f0π, 500)[begin:end-1]\n\n# 0.196049 seconds (145 allocations: 1009.938 KiB)\n@time sinogram = radon(img, angles);\n\n# 0.268649 seconds (147 allocations: 1.015 MiB)\n@time backproject = RadonKA.backproject(sinogram, angles);\n\nsimshow(sinogram)\nsimshow(backproject)","category":"page"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"<a  href=\"assets/sinogram.png\"><img src=\"assets/sinogram.png\"  width=\"300\"></a>\n<a  href=\"assets/radonka_backproject.png\"><img src=\"assets/radonka_backproject.png\"  width=\"308\"></a>","category":"page"},{"location":"#Examples","page":"RadonKA.jl","title":"Examples","text":"","category":"section"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"See either the documentation. Otherwise, this example shows the main features, including CUDA support. There is one tutorial about Gradient Descent optimization. Another one covers how the Radon transform is used in Volumetric Additive Manufacturing.","category":"page"},{"location":"#Development","page":"RadonKA.jl","title":"Development","text":"","category":"section"},{"location":"","page":"RadonKA.jl","title":"RadonKA.jl","text":"File an issue on GitHub if you encounter any problems.","category":"page"},{"location":"tutorial/#Tutorial","page":"Simple Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"We offer some examples on GitHub.  To run them, git clone the whole repository. Then do:","category":"page"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"(@v1.9) pkg> activate examples/\n  Activating project at `~/.julia/dev/RadonKA.jl/examples`\n\njulia> using Pluto; Pluto.run()","category":"page"},{"location":"tutorial/#Radon-transform","page":"Simple Tutorial","title":"Radon transform","text":"","category":"section"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"The radon transform only requires the image (or volume) and some angles:","category":"page"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"using RadonKA, ImageShow, ImageIO, TestImages\n\nimg = Float32.(testimage(\"resolution_test_512\"))\n\nangles = range(0f0, 2f0π, 500)[begin:end-1]\n\n# 0.196049 seconds (145 allocations: 1009.938 KiB)\n@time sinogram = radon(img, angles);","category":"page"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"(Image: )","category":"page"},{"location":"tutorial/#adjoint-Radon-(backproject)-transform","page":"Simple Tutorial","title":"adjoint Radon (backproject) transform","text":"","category":"section"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"# 0.268649 seconds (147 allocations: 1.015 MiB)\n@time backproject = RadonKA.backproject(sinogram, angles);\n\n# in Pluto or Jupyter\nsimshow(sinogram)\n\n[simshow(img) simshow(backproject)]","category":"page"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"Left is the original sample and right the simple backprojection. (Image: )","category":"page"},{"location":"tutorial/#Filtered-Backprojection","page":"Simple Tutorial","title":"Filtered Backprojection","text":"","category":"section"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"In the absence of noise, the filtered backprojection works well:","category":"page"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"#   0.252915 seconds (171 allocations: 13.664 MiB)\n@time filtered_backproject = RadonKA.backproject_filtered(sinogram, angles);","category":"page"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"(Image: )","category":"page"},{"location":"tutorial/#CUDA-Support","page":"Simple Tutorial","title":"CUDA Support","text":"","category":"section"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"RadonKA.jl supports CUDA and usually provides a 10-20x speedup on typical RTX 3xxx or 4xxx GPUs. Pay attention that the element type of the array/img should be Float32 for good speedup. In my case we used a AMD Ryzen 5 5600X 6-Core Processor and a CUDA RTX 3060 Super.","category":"page"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"using CUDA\n\nimg_c = CuArray(img);\nangles_c = CuArray(angles);\n\n# 0.005611 seconds (8.95 k CPU allocations: 363.828 KiB) (2 GPU allocations: 998.047 KiB, 0.26% memmgmt time)\nCUDA.@time CUDA.@sync sinogram_c = radon(img_c, angles_c);","category":"page"},{"location":"tutorial/#3D-example","page":"Simple Tutorial","title":"3D example","text":"","category":"section"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"Simply attach a third trailing dimension to the array. The radon transform is computed along the first and second dimension. The third dimension is just a batch dimension.","category":"page"},{"location":"tutorial/","page":"Simple Tutorial","title":"Simple Tutorial","text":"volume = randn(Float32,(512, 512, 512));\nvolume_c = CuArray(randn(Float32,(512, 512, 512)));\n\n# 86.795338 seconds (153 allocations: 498.039 MiB, 0.02% gc time)\n@time radon(volume, angles);\n\n# 2.527153 seconds (8.95 k CPU allocations: 363.703 KiB) (2 GPU allocations: 498.027 MiB, 0.06% memmgmt time)\nCUDA.@time CUDA.@sync radon(volume_c, angles_c);","category":"page"}]
}