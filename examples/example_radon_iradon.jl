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

# ╔═╡ b336e55e-0be4-422f-b48a-0a2242cb0915
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
	#img = box(Float32, (N, N, N_z), (N ÷4, N ÷ 4, 20), offset=(N ÷ 2 + 60, N ÷ 2 -50, N_z ÷ 2)) |> collect

	#img = box(Float32, (N, N, N_z), (1, 1, 1)) |> collect
	img = box(Float32, (N, N, N_z), (N ÷2, N ÷ 2, 50), offset=(N ÷ 2 - 50, N ÷ 2 + 50, N_z ÷ 2)) |> collect
	
	img .+= 0.0f0 .+ (rr2(Float32, (N, N, N_z)) .< 100 .^2)

	#img = box(Float32, (100, 100), (3, 3), offset=(51, 51)) |> collect
end;

# ╔═╡ 1393d029-66be-40aa-a2f9-f31317222575
img_c = CuArray(img);

# ╔═╡ 8be220a4-293d-411d-bbce-e39b64780814
md"# Radon Transform"

# ╔═╡ b8618268-0892-4abc-ae26-e25e41d07968
angles = range(0f0, 2f0π, 361)

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

# ╔═╡ 375a0179-8592-4d02-9686-d6a85a3eb048
size(sinogram)

# ╔═╡ edbf1577-0fd4-4261-bd04-499bc1a0debd
md"# IRadon Transform"

# ╔═╡ 61f17d9e-ed0a-4176-9466-464527c1b10e
@bind angle_cut1 Slider(1:361, show_value=true)

# ╔═╡ 7bbc33af-7082-42e4-ad5f-1d4273e87fbf
rad2deg.(Float32.(angles))

# ╔═╡ 7d08ba55-4490-400f-8497-5cbfb3f257c7
@bind angle_cut2 Slider(angle_cut1:361, show_value=true)

# ╔═╡ ed54c930-4f34-4f3d-9180-514dc59fde15
@time backproject = iradon(sinogram[:, angle_cut1:angle_cut2, :], angles[angle_cut1:angle_cut2]);

# ╔═╡ 037e9d64-505e-40f9-b710-20f57d29bd17
# ╠═╡ disabled = true
#=╠═╡
#CUDA.@time CUDA.@sync backproject_c = iradon(sinogram_c, CuArray(angles),
											 backend=CUDABackend());
  ╠═╡ =#

# ╔═╡ 72d63cbe-67d6-4a9c-80fa-d22743709105
@bind i_z2 Slider(1:size(sinogram, 3), show_value=true)

# ╔═╡ 365ee0e7-3545-4345-8b0c-8338a59c53b3
simshow(backproject[:, :, i_z2] .+ 0 .* img[:, :, i_z2])

# ╔═╡ 4c5038bd-d346-4070-822b-bc3e5331d99c
begin
	plot(backproject[end, :, i_z2])
	#plot!(vcat([0.0], sinogram[:, 91, i_z2]))
	#plot!(vcat([0.0], sinogram[:, 1, i_z2]))

end

# ╔═╡ 3c85e03a-a126-40f7-bcc2-0e928190f757
vcat([0.0], sinogram[:, 1, i_z2])

# ╔═╡ b06bb885-175c-4cac-8346-f69f7172a9aa
simshow(Array(backproject_c[:, :, i_z2]))

# ╔═╡ 30f1016a-5b66-48ec-93f5-8c1f1129207e
Revise.errors()

# ╔═╡ 447b163b-cc06-4427-b734-e0498df35260
sum(backproject)

# ╔═╡ 267d7684-4a1b-402d-96ae-c3e26b957e0b
sum(backproject_c)

# ╔═╡ ec342e4e-4ce2-4d26-91f8-4d06a4fad46e
begin
	arr = zeros(2,2)
	arr[1,1] = 1
	simshow(arr)
end

# ╔═╡ e24bb409-bd5d-4eca-8cdc-884daece26fa


# ╔═╡ 7e27da5a-1b04-4d4c-8c62-eaffa7f4f9ce
@time backproject2 = RadonKA.iradon2(sinogram, angles);

# ╔═╡ 93a7ab4a-b2dc-4fc2-bf69-66e6d615103f
CUDA.@time CUDA.@sync backproject2_cu = RadonKA.iradon2(sinogram_c[:, :, :], CuArray(angles[:]), backend=CUDABackend());

# ╔═╡ a29be556-174a-4ec5-962d-9fdf203d51aa


# ╔═╡ 52a86ed8-4504-4d9e-9ea6-6aeaf8540406
@bind i_z3 Slider(1:size(sinogram, 3), show_value=true)

# ╔═╡ 9d7c41db-adb5-4da2-98ae-96e967c1056e
simshow(Array(backproject2_cu[:, :, i_z3]))

# ╔═╡ c9b84c39-7a74-4893-bbb6-5241a121df04
Revise.errors()

# ╔═╡ Cell order:
# ╠═4eb3148e-8f8b-11ee-3cfe-854d3bd5cc80
# ╠═b336e55e-0be4-422f-b48a-0a2242cb0915
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
# ╠═783f05e0-2640-4ecd-8c19-1c15a99ee294
# ╠═db2676fd-3305-408f-93b4-08a3d04fdd02
# ╠═1a931e03-6a29-4c3e-b66f-bc1b5936a6f4
# ╠═3d584d94-b88f-4738-a470-7db1fb3fb996
# ╠═375a0179-8592-4d02-9686-d6a85a3eb048
# ╟─edbf1577-0fd4-4261-bd04-499bc1a0debd
# ╠═61f17d9e-ed0a-4176-9466-464527c1b10e
# ╠═7bbc33af-7082-42e4-ad5f-1d4273e87fbf
# ╠═7d08ba55-4490-400f-8497-5cbfb3f257c7
# ╠═ed54c930-4f34-4f3d-9180-514dc59fde15
# ╠═037e9d64-505e-40f9-b710-20f57d29bd17
# ╠═72d63cbe-67d6-4a9c-80fa-d22743709105
# ╠═365ee0e7-3545-4345-8b0c-8338a59c53b3
# ╠═4c5038bd-d346-4070-822b-bc3e5331d99c
# ╠═3c85e03a-a126-40f7-bcc2-0e928190f757
# ╠═b06bb885-175c-4cac-8346-f69f7172a9aa
# ╠═30f1016a-5b66-48ec-93f5-8c1f1129207e
# ╠═447b163b-cc06-4427-b734-e0498df35260
# ╠═267d7684-4a1b-402d-96ae-c3e26b957e0b
# ╠═ec342e4e-4ce2-4d26-91f8-4d06a4fad46e
# ╠═e24bb409-bd5d-4eca-8cdc-884daece26fa
# ╠═7e27da5a-1b04-4d4c-8c62-eaffa7f4f9ce
# ╠═93a7ab4a-b2dc-4fc2-bf69-66e6d615103f
# ╠═a29be556-174a-4ec5-962d-9fdf203d51aa
# ╟─52a86ed8-4504-4d9e-9ea6-6aeaf8540406
# ╠═9d7c41db-adb5-4da2-98ae-96e967c1056e
# ╠═c9b84c39-7a74-4893-bbb6-5241a121df04
