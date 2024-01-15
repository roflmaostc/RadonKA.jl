### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 179d107e-b3be-11ee-0a6c-49f1bf2a10fd
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ e3a3dc83-6536-445f-b56b-d8ad62cd9a85
using TestImages, RadonKA, ImageShow, ImageIO, Noise, PlutoUI, BenchmarkTools

# ╔═╡ e442d738-7a82-4f51-bb5b-17490401c19a
# ╠═╡ disabled = true
#=╠═╡
angles = [0.0]
  ╠═╡ =#

# ╔═╡ c1d555be-8380-4df5-8574-69fe19f56f5b
#smeared = iradon(sinogram, angles, ray_endpoints=range(0, 0, 49))

# ╔═╡ 48be3824-1a8e-495b-8746-a0b09b93f094
# ╠═╡ disabled = true
#=╠═╡
@btime smeared = iradon($sinogram, $angles)
  ╠═╡ =#

# ╔═╡ 17171a53-b47d-4546-bba9-16a3e75ff9f8
# ╠═╡ disabled = true
#=╠═╡
@btime smeared2 = iradon($sinogram, $angles, ray_endpoints=range(-20, 20, 199))
  ╠═╡ =#

# ╔═╡ 8b50b28c-7641-4338-bddc-ea52a7a3a8a2
size(-10:9)

# ╔═╡ 88d2d11a-4e36-4bf7-be8b-246e79f4f165
Revise.errors()

# ╔═╡ 332045ee-b897-42c1-aee2-775a5b338d94


# ╔═╡ 23ee4b67-adef-4f19-a8d6-e49387c43480
x = randn(Float32, (512, 512))

# ╔═╡ e92cab58-a783-4d12-ae38-f1fd60ce4b90
angles2 = deg2rad.(range(0, 360, 360))

# ╔═╡ 3ef03b3a-b412-4b6b-893a-f2f7772d319c
@time smeared = iradon(sinogram, angles)

# ╔═╡ 7e8ea923-3536-4977-ad8d-ff02c1213943
simshow(smeared, γ=0.1)

# ╔═╡ d627d41d-d363-4c3a-a0d8-e79c6e271b12
@time smeared2 = iradon(sinogram, angles, ray_endpoints=range(-30, 30, 199))

# ╔═╡ 5688c466-5cb6-4b57-8e84-c39a2a08e5b8
simshow(smeared2, γ=0.2)

# ╔═╡ f0abe747-5539-4274-b639-f184183eb099


# ╔═╡ 5dc20628-490c-4edf-9735-6eae988546e6
md"# img"

# ╔═╡ 5987412b-b4c9-4da1-b11b-f61ea63494b4
img = Float32.(testimage("resolution_test_512"));

# ╔═╡ 82cddfda-a315-41aa-8653-d6ea1104f73f
simshow(img)

# ╔═╡ 16bb225c-1fe6-4e4f-b532-49e73e82f26f
y_dist_end = range(-10, 10, size(img, 1)-1)

# ╔═╡ 40992985-b12f-47e2-aee6-28730d3e808e
sinogram2 = radon(img, angles2, ray_endpoints=y_dist_end);

# ╔═╡ a1546c7f-56aa-402b-93db-4474698404c3
simshow(sinogram2)

# ╔═╡ e29c286d-63ad-4825-a745-0a702e3bf9ac
img2 = iradon(sinogram2, angles2, ray_endpoints=y_dist_end)

# ╔═╡ 62550c6c-0047-436a-90a1-3b9c2f17a66f
simshow(img2)

# ╔═╡ 37272216-d78e-42b0-a7f4-1bc8164ae2f2
simshow(img2 ./ maximum(img2) .- img / maximum(img))

# ╔═╡ 9829fe1e-6b42-4dba-9813-7ec04304e902
# ╠═╡ disabled = true
#=╠═╡
begin
	sinogram = zeros(Float32, (199, 1))
	sinogram[1:3:end, 1] .= 1
end
  ╠═╡ =#

# ╔═╡ 6add1095-8a4a-4a11-b368-b50ef87b557c
sinogram = radon(x, angles2);

# ╔═╡ Cell order:
# ╠═179d107e-b3be-11ee-0a6c-49f1bf2a10fd
# ╠═e3a3dc83-6536-445f-b56b-d8ad62cd9a85
# ╠═e442d738-7a82-4f51-bb5b-17490401c19a
# ╠═9829fe1e-6b42-4dba-9813-7ec04304e902
# ╠═c1d555be-8380-4df5-8574-69fe19f56f5b
# ╠═3ef03b3a-b412-4b6b-893a-f2f7772d319c
# ╠═48be3824-1a8e-495b-8746-a0b09b93f094
# ╠═17171a53-b47d-4546-bba9-16a3e75ff9f8
# ╠═d627d41d-d363-4c3a-a0d8-e79c6e271b12
# ╠═8b50b28c-7641-4338-bddc-ea52a7a3a8a2
# ╠═88d2d11a-4e36-4bf7-be8b-246e79f4f165
# ╠═7e8ea923-3536-4977-ad8d-ff02c1213943
# ╠═5688c466-5cb6-4b57-8e84-c39a2a08e5b8
# ╟─332045ee-b897-42c1-aee2-775a5b338d94
# ╠═23ee4b67-adef-4f19-a8d6-e49387c43480
# ╠═e92cab58-a783-4d12-ae38-f1fd60ce4b90
# ╠═6add1095-8a4a-4a11-b368-b50ef87b557c
# ╠═f0abe747-5539-4274-b639-f184183eb099
# ╠═5dc20628-490c-4edf-9735-6eae988546e6
# ╠═5987412b-b4c9-4da1-b11b-f61ea63494b4
# ╠═82cddfda-a315-41aa-8653-d6ea1104f73f
# ╠═16bb225c-1fe6-4e4f-b532-49e73e82f26f
# ╠═40992985-b12f-47e2-aee6-28730d3e808e
# ╠═a1546c7f-56aa-402b-93db-4474698404c3
# ╠═e29c286d-63ad-4825-a745-0a702e3bf9ac
# ╠═62550c6c-0047-436a-90a1-3b9c2f17a66f
# ╠═37272216-d78e-42b0-a7f4-1bc8164ae2f2
