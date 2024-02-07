### A Pluto.jl notebook ###
# v0.19.38

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
	Pkg.instantiate()
	using Revise
end

# ╔═╡ b336e55e-0be4-422f-b48a-0a2242cb0915
using RadonKA

# ╔═╡ 1311e853-c4cd-42bb-8bf3-5e0d564bf9c5
using IndexFunArrays, ImageShow, Plots, ImageIO, PlutoUI, PlutoTest, TestImages

# ╔═╡ 03bccb92-b47f-477a-9bdb-74cc404da690
using KernelAbstractions, CUDA, CUDA.CUDAKernels

# ╔═╡ 6f6c8d28-5a54-440e-9b7a-52e1fce959ab
md"# Activate example environment"

# ╔═╡ f5e2024b-deaf-4344-b610-a4b956abbfaa
md"# Load CUDA
In case you have no CUDA card available, it will run on CPU :)
"

# ╔═╡ ef92457a-87c0-43bf-a046-9fe82afbbe13
begin
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ 810aebe4-2c6e-4ba6-916b-9e4306df33c9
TableOfContents()

# ╔═╡ d25c1381-baf1-429b-8150-622b8f731d83
md"# Example Image"

# ╔═╡ 54208d78-cf55-41d7-b4bf-6d1ab4927bbb
begin
	N = 512
	N_z = 20
	img = box(Float32, (N, N, N_z), (N ÷4, N ÷ 4, 20), offset=(N ÷ 2 + 60, N ÷ 2 -50, N_z ÷ 2)) |> collect

	#img = box(Float32, (N, N, N_z), (1, 1, 1)) |> collect
	#img = box(Float32, (N, N, N_z), (N ÷2, N ÷ 2, 50), offset=(N ÷ 2 - 50, N ÷ 2 + 50, N_z ÷ 2)) |> collect
	
	img .+= 0.0f0 .+ (rr2(Float32, (N, N, N_z), offset=(180, 210, N_z÷2)) .< 30 .^2)
	img[:, :, 1] .= testimage("resolution_test_512")
	#img = box(Float32, (100, 100), (3, 3), offset=(51, 51)) |> collect
end;

# ╔═╡ 1393d029-66be-40aa-a2f9-f31317222575
img_c = togoc(img);

# ╔═╡ 8be220a4-293d-411d-bbce-e39b64780814
md"# Radon Transform"

# ╔═╡ b8618268-0892-4abc-ae26-e25e41d07968
angles = range(0f0, 2π, 1000)[begin:end-1]

# ╔═╡ 135e728b-efd8-44bc-81d9-6a2244ce4c31
angles_c = togoc(angles);

# ╔═╡ d2cc6fc6-135b-4c4a-8453-9c5bf9e4a24f
@mytime sinogram = radon(img, angles);

# ╔═╡ dc14103d-993c-402f-a8b5-a35843f3f4ac
@mytime sinogram_c = radon(img_c, angles_c);

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

# ╔═╡ 7e27da5a-1b04-4d4c-8c62-eaffa7f4f9ce
@time backproject = RadonKA.iradon(sinogram, angles);

# ╔═╡ 4e367035-eb2f-4dfa-9646-7c182a111c49
md"Use this slider to add more and more angles to the iradon transform"

# ╔═╡ b9bf49a0-7320-4269-9a6a-ac2533ab5fde
@bind angle_limit Slider(1:size(sinogram, 2), default=size(sinogram, 2), show_value=true)

# ╔═╡ 93a7ab4a-b2dc-4fc2-bf69-66e6d615103f
CUDA.@time CUDA.@sync backproject_cu = RadonKA.iradon(sinogram_c[:, begin:angle_limit, :], togoc(angles[begin:angle_limit]));

# ╔═╡ 52a86ed8-4504-4d9e-9ea6-6aeaf8540406
@bind i_z3 Slider(1:size(sinogram, 3), show_value=true)

# ╔═╡ 9d7c41db-adb5-4da2-98ae-96e967c1056e
simshow(Array(backproject_cu[:, :, i_z3]))

# ╔═╡ 81b07387-ede6-4a66-8260-8605cd978ede
md"# Filtered Backprojection"

# ╔═╡ 32b15077-5e09-4693-8f12-3b2029fe63cc
@mytime filtered_bproj = RadonKA.filtered_backprojection(sinogram_c, togoc(angles));

# ╔═╡ bc6e2d40-fcd1-4d7b-8f96-3d4d9e4336de
@bind i_z4 Slider(1:size(sinogram, 3), show_value=true)

# ╔═╡ eee184e3-8d5d-42fb-81fb-a5d7e59a083a
simshow(Array(filtered_bproj[:, :, i_z4]))

# ╔═╡ a29be556-174a-4ec5-962d-9fdf203d51aa
backproject ≈ Array(backproject_cu)

# ╔═╡ Cell order:
# ╠═6f6c8d28-5a54-440e-9b7a-52e1fce959ab
# ╠═4eb3148e-8f8b-11ee-3cfe-854d3bd5cc80
# ╠═b336e55e-0be4-422f-b48a-0a2242cb0915
# ╠═1311e853-c4cd-42bb-8bf3-5e0d564bf9c5
# ╟─f5e2024b-deaf-4344-b610-a4b956abbfaa
# ╠═03bccb92-b47f-477a-9bdb-74cc404da690
# ╠═ef92457a-87c0-43bf-a046-9fe82afbbe13
# ╟─810aebe4-2c6e-4ba6-916b-9e4306df33c9
# ╟─d25c1381-baf1-429b-8150-622b8f731d83
# ╠═54208d78-cf55-41d7-b4bf-6d1ab4927bbb
# ╠═1393d029-66be-40aa-a2f9-f31317222575
# ╠═01b4b8f8-37d5-425f-975e-ebb3890d8624
# ╟─8be220a4-293d-411d-bbce-e39b64780814
# ╠═b8618268-0892-4abc-ae26-e25e41d07968
# ╠═135e728b-efd8-44bc-81d9-6a2244ce4c31
# ╠═d2cc6fc6-135b-4c4a-8453-9c5bf9e4a24f
# ╠═dc14103d-993c-402f-a8b5-a35843f3f4ac
# ╠═783f05e0-2640-4ecd-8c19-1c15a99ee294
# ╠═db2676fd-3305-408f-93b4-08a3d04fdd02
# ╠═1a931e03-6a29-4c3e-b66f-bc1b5936a6f4
# ╠═3d584d94-b88f-4738-a470-7db1fb3fb996
# ╟─edbf1577-0fd4-4261-bd04-499bc1a0debd
# ╠═7e27da5a-1b04-4d4c-8c62-eaffa7f4f9ce
# ╟─4e367035-eb2f-4dfa-9646-7c182a111c49
# ╟─b9bf49a0-7320-4269-9a6a-ac2533ab5fde
# ╠═93a7ab4a-b2dc-4fc2-bf69-66e6d615103f
# ╠═52a86ed8-4504-4d9e-9ea6-6aeaf8540406
# ╠═9d7c41db-adb5-4da2-98ae-96e967c1056e
# ╟─81b07387-ede6-4a66-8260-8605cd978ede
# ╠═32b15077-5e09-4693-8f12-3b2029fe63cc
# ╟─bc6e2d40-fcd1-4d7b-8f96-3d4d9e4336de
# ╠═eee184e3-8d5d-42fb-81fb-a5d7e59a083a
# ╠═a29be556-174a-4ec5-962d-9fdf203d51aa
