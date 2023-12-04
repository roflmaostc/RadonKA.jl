# Benchmark and Comparison with Matlab and Astra
Tested on a AMD Ryzen 9 5900X 12-Core Processor with 24 Threads and a NVIDIA GeForce RTX 3060 with Julia 1.9.4 on Ubuntu 22.04.

# Results

|                   |RadonKA.jl CPU | RadonKA.jl GPU    | Matlab CPU | Astra CPU | Astra GPU |
|-------------------|---------------|-------------------|------------|-----------|-----------|
|2D sample - Radon  |1.2s           |0.091s             |0.39s       |7.0s       |0.025s     |
|2D sample - IRadon |1.8s           |0.52s              |0.37s       |6.4s       |N/A        |
|3D sample - Radon  |8.4s           |0.47s              |9.01s       |N/A        |1.12s      |
|3D sample - IRadon |10.5s          |0.59s              |3.24s       |N/A        |N/A        |



## RadonKA.jl
See this [benchmark example](examples/benchmark_radon_iradon.jl)
```julia
using IndexFunArrays, ImageShow, Plots, ImageIO, PlutoUI, PlutoTest, TestImages
using RadonKA
using CUDA, CUDA.CUDAKernels

sample_2D = Float32.(testimage("resolution_test_1920"));
sample_2D_c = CuArray(sample_2D);
simshow(sample_2D)

angles = range(0, 2π, 500);
angles_c = CuArray(angles);

# run those cells multiple times
@time sinogram = radon(sample_2D, angles);
simshow(sinogram)
@time sample_iradon = iradon(sinogram, angles);

CUDA.@time CUDA.@sync sinogram_c = radon(sample_2D_c, angles_c, backend=CUDABackend());
CUDA.@time CUDA.@sync sample_iradon_c = iradon(sinogram_c, angles_c, backend=CUDABackend());
```
![](../assets/radonka_sinogram.png)
![](../assets/radonka_iradon.png)


## Matlab (R2023a)
```matlab
arr = single(imread("/tmp/sample.jpg")); 
%arr = rand(1920, 1920, "single");
theta = linspace(0, 360, 500);

tic;
R = radon(arr, theta);
toc;
R = R / max(R(:));
imwrite(R, "/tmp/matlab_sinogram.png");

tic; 
iR = iradon(R, theta, "linear", "none");
toc;
iR = iR / max(iR(:));
imwrite(iR, "/tmp/matlab_iradon.png");



x = (rand(100, 512, 512, 'single'));

theta = linspace(0, 360, 500);
y = (zeros(size(x, 1), size(radon(squeeze(x(1, :, :)), theta), 1), size(radon(squeeze(x(1, :, :)), theta), 2), 'single'));
ix = zeros(100, 514, 514, "single");

tic;
for i = 1:size(x, 1)
    y(i, :, :) = radon(squeeze(x(i, :, :)), theta);
end
toc;


tic;
for i = 1:size(y, 1)
    ix(i, :, :) = iradon(squeeze(y(i, :, :)), theta);
end
toc;
```

![](../assets/matlab_sinogram.png)
![](../assets/matlab_iradon.png)

## Astra
Some of the benchmarks did not run properly or were providing non-sense.
Astra's docs are unfortunately slightly confusing...
```python
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import astra

im = np.array(iio.imread('/tmp/sample.png'), dtype=np.float32)

plt.imshow(im)


vol_geom = astra.create_vol_geom(1920, 1920)
angles =  np.linspace(0,2 * np.pi,500)

proj_geom = astra.create_proj_geom('parallel', 1.0, 1920, angles)
proj_id = astra.create_projector('line', proj_geom,vol_geom)


%%time
sinogram_id, sinogram = astra.create_sino(im, proj_id)
np.copy(sinogram);

plt.imshow(sinogram)

proj_geom = astra.create_proj_geom('parallel', 1.0, 1920, angles)
proj_id = astra.create_projector('cuda', proj_geom,vol_geom)

%%time
sinogram_id, sinogram = astra.create_sino(im, proj_id)
np.copy(sinogram);

rec_id = astra.data2d.create("-vol", vol_geom)
cfg = astra.astra_dict('BP')
cfg["ReconstructionDataId"] = rec_id
cfg["ProjectionDataId"] = sinogram_id
cfg["ProjectorId"] = proj_id
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
# Run back-projection and get the reconstruction

%%time
astra.algorithm.run(alg_id)
f_rec = astra.data2d.get(rec_id)
#np.copy(f_rec)


im_3D = np.random.rand(100, 512, 512)

vol_geom = astra.create_vol_geom(512, 512, 100)

proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 512, 512, angles)
#proj_id = astra.create_projector('line', proj_geom,vol_geom)

%%time
proj_id, proj_data = astra.create_sino3d_gpu(im_3D, proj_geom, vol_geom)
np.copy(proj_data)


rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('BP3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id


# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

%time
astra.algorithm.run(alg_id, 1)
rec = astra.data3d.get(rec_id)
np.copy(rec)
```

![](../assets/astra_sinogram.png)
![](../assets/astra_iradon.png)

