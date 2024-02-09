### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 985b045c-91f0-11ee-1053-79ebc2227cf8
begin
	begin
		using Pkg
		Pkg.activate(".")
		Pkg.develop(path="../.")
		using Revise
	end
end

# ╔═╡ 9054a2a8-0998-4e3d-a796-aeac419d072e
using RadonKA, ImageShow, ImageIO, TestImages

# ╔═╡ 25a6820f-9eb0-42b7-97e8-9561c0465cd5
using CUDA, CUDA.CUDAKernels

# ╔═╡ dcb889cb-72ae-473c-a7f5-9d803b3a4811
img = Float32.(testimage("resolution_test_512"));

# ╔═╡ fab34ee8-45c4-4895-a36f-786416ff3e11
simshow(img)

# ╔═╡ 4177b5ed-020d-4d86-abb8-e8b1f66763fd
angles = range(0f0, 2f0π, 500)[begin:end-1]

# ╔═╡ 304dc63a-1aa0-43d2-b600-c994a272e349
@time sinogram = radon(img, angles);

# ╔═╡ 08e2c129-c7cd-4ea7-b183-a58b3383b991
simshow(sinogram)

# ╔═╡ 451af1b2-c840-4ae8-935f-eb7965e7104b
@time backproject = RadonKA.backproject(sinogram, angles);

# ╔═╡ 42c42599-96b4-4a69-bbbd-5602cca700f4
[simshow(img) simshow(backproject)]

# ╔═╡ ebd1170c-84ea-420d-a2e1-9afe0acb637b
@time filtered_backproject = RadonKA.filtered_backprojection(sinogram, angles);

# ╔═╡ 00756bf0-e309-4991-84a8-4afed5cdfa93
simshow(filtered_backproject)

# ╔═╡ a439bca4-a807-4fde-a5fd-86e6ef4fb6ea
img_c = CuArray(img);

# ╔═╡ 32d23869-db0a-494e-978d-7d781f57c6e8
angles_c = CuArray(angles);

# ╔═╡ da0213e7-0df2-4d07-8eaa-e08001d47f22
CUDA.@time CUDA.@sync sinogram_c = radon(img_c, angles_c, backend=CUDABackend());

# ╔═╡ fddabb94-810e-4540-ace5-8a1da4937d5f
volume = randn(Float32,(512, 512, 512));

# ╔═╡ 92d1c2c5-d6a6-4430-a8f8-6a3bd4369f77
volume_c = CuArray(randn(Float32,(512, 512, 512)));

# ╔═╡ 0f6068c5-8e28-4a40-99e2-0899ae42d333
@time radon(volume, angles);

# ╔═╡ e4a33a2e-cfce-493a-a74a-61a1105369c4
CUDA.@time CUDA.@sync radon(volume_c, angles_c, backend=CUDABackend());

# ╔═╡ efd70d17-42b1-4368-a889-f633047c5756


# ╔═╡ Cell order:
# ╠═985b045c-91f0-11ee-1053-79ebc2227cf8
# ╠═9054a2a8-0998-4e3d-a796-aeac419d072e
# ╠═dcb889cb-72ae-473c-a7f5-9d803b3a4811
# ╠═fab34ee8-45c4-4895-a36f-786416ff3e11
# ╠═4177b5ed-020d-4d86-abb8-e8b1f66763fd
# ╠═304dc63a-1aa0-43d2-b600-c994a272e349
# ╠═08e2c129-c7cd-4ea7-b183-a58b3383b991
# ╠═451af1b2-c840-4ae8-935f-eb7965e7104b
# ╠═42c42599-96b4-4a69-bbbd-5602cca700f4
# ╠═ebd1170c-84ea-420d-a2e1-9afe0acb637b
# ╠═00756bf0-e309-4991-84a8-4afed5cdfa93
# ╠═25a6820f-9eb0-42b7-97e8-9561c0465cd5
# ╠═a439bca4-a807-4fde-a5fd-86e6ef4fb6ea
# ╠═32d23869-db0a-494e-978d-7d781f57c6e8
# ╠═da0213e7-0df2-4d07-8eaa-e08001d47f22
# ╠═fddabb94-810e-4540-ace5-8a1da4937d5f
# ╠═92d1c2c5-d6a6-4430-a8f8-6a3bd4369f77
# ╠═0f6068c5-8e28-4a40-99e2-0899ae42d333
# ╠═e4a33a2e-cfce-493a-a74a-61a1105369c4
# ╠═efd70d17-42b1-4368-a889-f633047c5756
