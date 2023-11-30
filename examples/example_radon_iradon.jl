### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 4eb3148e-8f8b-11ee-3cfe-854d3bd5cc80
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ 4ee97893-5e8f-42cf-9ecd-c63c9db76869
using RadonKA

# ╔═╡ 1311e853-c4cd-42bb-8bf3-5e0d564bf9c5
using IndexFunArrays, ImageShow, Plots, ImageIO, PlutoUI, PlutoTest

# ╔═╡ 03bccb92-b47f-477a-9bdb-74cc404da690
using KernelAbstractions, CUDA, CUDA.CUDAKernels

# ╔═╡ d25c1381-baf1-429b-8150-622b8f731d83
md"# Example Image"

# ╔═╡ 54208d78-cf55-41d7-b4bf-6d1ab4927bbb
begin
	N = 320
	N_z = 100
	img = box(Float32, (N, N, N_z), (N ÷4, N ÷ 4, 20), offset=(N ÷ 2 + 50, N ÷ 2 -100, N_z ÷ 2)) |> collect
	
	img .+= 0.0f0 .+ (rr2(Float32, (N, N, N_z)) .< 100 .^2)

	#img = box(Float32, (100, 100), (3, 3), offset=(51, 51)) |> collect
end;

# ╔═╡ 1393d029-66be-40aa-a2f9-f31317222575
img_c = CuArray(img);

# ╔═╡ 8be220a4-293d-411d-bbce-e39b64780814
md"# Radon Transform"

# ╔═╡ b8618268-0892-4abc-ae26-e25e41d07968
angles = range(0f0, 2f0π, 360)

# ╔═╡ d2cc6fc6-135b-4c4a-8453-9c5bf9e4a24f
@time sinogram = radon(img, angles);

# ╔═╡ dc14103d-993c-402f-a8b5-a35843f3f4ac
CUDA.@time CUDA.@sync sinogram_c = radon(img_c, CuArray(angles),
										 backend=CUDABackend());

# ╔═╡ 783f05e0-2640-4ecd-8c19-1c15a99ee294
@bind i_z Slider(1:size(sinogram, 3), show_value=true)

# ╔═╡ 01b4b8f8-37d5-425f-975e-ebb3890d8624
simshow(img[:, :, i_z])

# ╔═╡ db2676fd-3305-408f-93b4-08a3d04fdd02
@test sinogram ≈ Array(sinogram_c)

# ╔═╡ 1a931e03-6a29-4c3e-b66f-bc1b5936a6f4
simshow(sinogram[:, :, i_z])

# ╔═╡ 3d584d94-b88f-4738-a470-7db1fb3fb996
simshow(Array(sinogram_c[:, :, i_z]))

# ╔═╡ edbf1577-0fd4-4261-bd04-499bc1a0debd
md"# IRadon Transform"

# ╔═╡ ed54c930-4f34-4f3d-9180-514dc59fde15


# ╔═╡ 365ee0e7-3545-4345-8b0c-8338a59c53b3


# ╔═╡ Cell order:
# ╠═4eb3148e-8f8b-11ee-3cfe-854d3bd5cc80
# ╠═4ee97893-5e8f-42cf-9ecd-c63c9db76869
# ╠═1311e853-c4cd-42bb-8bf3-5e0d564bf9c5
# ╠═03bccb92-b47f-477a-9bdb-74cc404da690
# ╟─d25c1381-baf1-429b-8150-622b8f731d83
# ╠═54208d78-cf55-41d7-b4bf-6d1ab4927bbb
# ╠═1393d029-66be-40aa-a2f9-f31317222575
# ╠═01b4b8f8-37d5-425f-975e-ebb3890d8624
# ╟─8be220a4-293d-411d-bbce-e39b64780814
# ╠═b8618268-0892-4abc-ae26-e25e41d07968
# ╠═d2cc6fc6-135b-4c4a-8453-9c5bf9e4a24f
# ╠═dc14103d-993c-402f-a8b5-a35843f3f4ac
# ╟─783f05e0-2640-4ecd-8c19-1c15a99ee294
# ╠═db2676fd-3305-408f-93b4-08a3d04fdd02
# ╠═1a931e03-6a29-4c3e-b66f-bc1b5936a6f4
# ╠═3d584d94-b88f-4738-a470-7db1fb3fb996
# ╠═edbf1577-0fd4-4261-bd04-499bc1a0debd
# ╠═ed54c930-4f34-4f3d-9180-514dc59fde15
# ╠═365ee0e7-3545-4345-8b0c-8338a59c53b3
