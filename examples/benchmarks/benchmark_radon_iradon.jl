### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ e5a21d4e-928b-11ee-3909-53f97530eefa
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ 3757ba3d-8ff2-4bc8-93bc-850592ac0ec9
Pkg.add("FileIO")

# ╔═╡ 114e830d-9bd0-4546-ad6d-2e62b4a1194b
using IndexFunArrays, ImageShow, Plots, ImageIO, PlutoUI, PlutoTest, TestImages

# ╔═╡ dc48ab00-c044-4292-a908-a155af022fde
using FileIO

# ╔═╡ a4dc63b8-4ba3-42fe-9743-3d7169604415
using RadonKA

# ╔═╡ c9b6dc38-dcfd-4b1f-91a8-2d105cda9c40
using BenchmarkTools

# ╔═╡ 1fddfd52-e964-4e36-a2f6-53302aa7e217
using CUDA, CUDA.CUDAKernels

# ╔═╡ a96e7332-dab7-41f0-b979-9dcf39c9e0bf
sample_2D = Float32.(testimage("resolution_test_1920"));

# ╔═╡ 50565c52-fa1a-4ed2-96d8-15ca67b44081
save("/tmp/sample.png", sample_2D)

# ╔═╡ 8a7a0f9d-fa4f-4a7f-b385-a02d441aa974
sample_2D_c = CuArray(sample_2D);

# ╔═╡ 3c11a680-a6f2-4f81-9ae0-091ae95002e5
simshow(sample_2D)

# ╔═╡ df8f25ee-dc19-442d-97fe-3100fb31c5d0
angles = range(0, 2π, 500);

# ╔═╡ 4adb8e5d-e6ca-4978-bb3f-98788b38defe
angles_c = CuArray(angles);

# ╔═╡ 6672cff8-e1af-4425-8c19-fbd2448f40ee
@benchmark sinogram = radon($sample_2D, $angles)

# ╔═╡ 3aaf53a3-1087-4d8f-9533-1963319ece01
sinogram = radon(sample_2D, angles)

# ╔═╡ 1fb7d9bd-daa8-4898-80e2-282d03bb53a8
simshow(sinogram)

# ╔═╡ 76df1a12-2e52-4440-8c39-9f126715e77c
save("/tmp/radonka_sinogram.png", sinogram ./ maximum(sinogram));

# ╔═╡ a11fe70a-3425-4335-ba09-00efbae6c9af
 sample_backproject = backproject(sinogram,angles);

# ╔═╡ 29f7d88c-3bab-43a9-a253-d3bcfc98b706
save("/tmp/radonka_backproject.png", sample_backproject ./ maximum(sample_backproject));

# ╔═╡ da297929-f4f9-4fa2-b860-cfc159ac39f1
@benchmark sample_backproject = backproject($sinogram, $angles)

# ╔═╡ a61ea8f2-6329-4b6b-b4a2-1f5ac17bc618
sinogram_c = radon(sample_2D_c, angles_c, backend=CUDABackend())

# ╔═╡ c7bb742c-5f2e-44e1-9304-b14db275043d
@benchmark CUDA.@sync sinogram_c = radon($sample_2D_c, $angles_c, backend=CUDABackend())

# ╔═╡ 1b8e4e3f-d9db-4646-8ccf-840c21ecc328
@benchmark CUDA.@sync sample_backproject_c = backproject($sinogram_c, $angles_c, backend=CUDABackend())

# ╔═╡ c602a8df-d646-4f31-9d6f-902456f07ae5
md"# 3D"

# ╔═╡ 3570e170-77e5-449d-8e97-5ed6090ebc55
sample_3D = randn(Float32, (512, 512, 100));

# ╔═╡ d5e9a965-1e93-4998-b953-57ca0fdd1a50
sample_3D_c = CuArray(sample_3D)

# ╔═╡ 137f6c8b-c774-49e4-b163-eede8b60abfd
@time sinogram_3D = radon(sample_3D, angles);

# ╔═╡ a5bf8b01-aefb-42e6-8ec5-588190bb3fa2
@benchmark radon($sample_3D, $angles)

# ╔═╡ b5825676-6600-4cbe-865b-788ef1567b85
@benchmark backproject($sinogram_3D, $angles)

# ╔═╡ 494190e4-e28a-42ad-b1c1-926a740ea2cb
sinogram_3D_c = radon(sample_3D_c, angles_c, backend=CUDABackend())

# ╔═╡ 52c781ab-c08a-4b10-9568-011abec63bda
@benchmark CUDA.@sync radon($sample_3D_c, $angles_c, backend=CUDABackend())

# ╔═╡ cd166ef5-863f-48b9-bb89-a2a32daa166b
@benchmark CUDA.@sync backproject($sinogram_3D_c, $angles_c, backend=CUDABackend())

# ╔═╡ Cell order:
# ╠═e5a21d4e-928b-11ee-3909-53f97530eefa
# ╠═114e830d-9bd0-4546-ad6d-2e62b4a1194b
# ╠═dc48ab00-c044-4292-a908-a155af022fde
# ╠═3757ba3d-8ff2-4bc8-93bc-850592ac0ec9
# ╠═a4dc63b8-4ba3-42fe-9743-3d7169604415
# ╠═c9b6dc38-dcfd-4b1f-91a8-2d105cda9c40
# ╠═1fddfd52-e964-4e36-a2f6-53302aa7e217
# ╠═a96e7332-dab7-41f0-b979-9dcf39c9e0bf
# ╠═50565c52-fa1a-4ed2-96d8-15ca67b44081
# ╠═8a7a0f9d-fa4f-4a7f-b385-a02d441aa974
# ╠═3c11a680-a6f2-4f81-9ae0-091ae95002e5
# ╠═df8f25ee-dc19-442d-97fe-3100fb31c5d0
# ╠═4adb8e5d-e6ca-4978-bb3f-98788b38defe
# ╠═6672cff8-e1af-4425-8c19-fbd2448f40ee
# ╠═3aaf53a3-1087-4d8f-9533-1963319ece01
# ╠═1fb7d9bd-daa8-4898-80e2-282d03bb53a8
# ╠═76df1a12-2e52-4440-8c39-9f126715e77c
# ╠═29f7d88c-3bab-43a9-a253-d3bcfc98b706
# ╠═a11fe70a-3425-4335-ba09-00efbae6c9af
# ╠═da297929-f4f9-4fa2-b860-cfc159ac39f1
# ╠═a61ea8f2-6329-4b6b-b4a2-1f5ac17bc618
# ╠═c7bb742c-5f2e-44e1-9304-b14db275043d
# ╠═1b8e4e3f-d9db-4646-8ccf-840c21ecc328
# ╠═c602a8df-d646-4f31-9d6f-902456f07ae5
# ╠═3570e170-77e5-449d-8e97-5ed6090ebc55
# ╠═d5e9a965-1e93-4998-b953-57ca0fdd1a50
# ╠═137f6c8b-c774-49e4-b163-eede8b60abfd
# ╠═a5bf8b01-aefb-42e6-8ec5-588190bb3fa2
# ╠═b5825676-6600-4cbe-865b-788ef1567b85
# ╠═494190e4-e28a-42ad-b1c1-926a740ea2cb
# ╠═52c781ab-c08a-4b10-9568-011abec63bda
# ╠═cd166ef5-863f-48b9-bb89-a2a32daa166b
