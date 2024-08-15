### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ cd52f230-c539-11ee-0c61-0d1afcad5372
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 90b4b415-3ce9-4990-81f1-346c5015de21
using RadonKA, ImageShow, PlutoUI, TestImages

# ╔═╡ fab527ae-223c-47d0-93a5-ade5e5e74f4b
TableOfContents()

# ╔═╡ 3ebc1e08-39da-412e-8902-308a395b7c59
md"# Parallel Geometry"

# ╔═╡ 20e238e8-357e-4896-98b0-3c6d0e34a48c
angles = [0]

# ╔═╡ 9ed9ed78-11b3-491d-97d5-ce656d0bc270
# output image size
N = 200

# ╔═╡ 174067b0-c786-4392-8d06-a167f255e4bf
begin
	sinogram = zeros((N - 1, length(angles)))
	sinogram[1:6:end] .= 1
end

# ╔═╡ 22fe45cd-2f73-4a94-821e-57ee57d0116f
geometry_parallel = RadonParallelCircle(N, -(N-1)÷2:(N-1)÷2)

# ╔═╡ 07bec511-7802-45e4-b331-0d35e8f850c1
projection_parallel = backproject(sinogram, angles; geometry=geometry_parallel);

# ╔═╡ 9e5a58be-0de5-4820-bc3b-6dff931271a9
simshow(projection_parallel)

# ╔═╡ 511f6c64-6da5-4c8d-8a65-93d648b8887e
md"# Make Parallel geometry smaller"

# ╔═╡ 01bcdaab-bc35-458e-8932-7554c83759f2
begin
	sinogram_small = zeros((97, length(angles)))
	sinogram_small[1:3:end] .= 1
end

# ╔═╡ dadbe1af-a570-4c66-bf8e-0fb2263f18ea
 collect(1:3:97)

# ╔═╡ 97b8932e-7a5b-47fc-bf52-36cb9e1e999b
geometry_small = RadonParallelCircle(198, -48:48)

# ╔═╡ 27d5aac8-2528-46f2-a0dd-91c3fe5d275c
projection_small = backproject(sinogram_small, angles; geometry=geometry_small);

# ╔═╡ 29c9d1ff-694c-4fff-b551-836cc9eaf347
simshow(projection_small)

# ╔═╡ da663a25-6dcd-4664-9bd1-c84970e58346
md"# Similar to fan Beam Tomography"

# ╔═╡ 65d32c65-6e1f-417b-aba3-3c34dac35e05
geometry_fan = RadonFlexibleCircle(N, -(N-1)÷2:(N-1)÷2, range(-(N-1)÷4, (N-1)÷4, N-1))

# ╔═╡ 37d760fa-74e6-47d1-b8e6-3315c1747b4c
projected_fan = backproject(sinogram, angles; geometry=geometry_fan);

# ╔═╡ 878121c5-4ad5-477b-88c7-f53df7510052
simshow(projected_fan, γ=0.01)

# ╔═╡ afe43c8c-cb98-411b-af8f-1228983ee2e0
md"# Extreme fan Beam Tomography"

# ╔═╡ 7b0c8263-50d0-4569-96cb-297b4746ece3
geometry_extreme = RadonFlexibleCircle(N, -(N-1)÷2:(N-1)÷2, zeros((199,)))

# ╔═╡ 0756e11b-35ca-4eef-be48-52b8b5c098cd
projected_extreme = backproject(sinogram, angles; geometry=geometry_extreme);

# ╔═╡ 1e22032d-d57d-42d3-a9c5-9f2e94531346
simshow(projected_extreme, γ=0.01)

# ╔═╡ 17d64cfb-2aca-4998-8028-cbc8509f5459
md"# Using Different weighting
For example, if in your application some rays are stronger than others you can include weight factor array into the API.
"

# ╔═╡ 72209fb0-e671-402a-adc4-bef553d81721
geometry_weight = RadonParallelCircle(N, -(N-1)÷2:(N-1)÷2, abs.(-(N-1)÷2:(N-1)÷2))

# ╔═╡ 45b82171-c581-4fcf-9695-bc51464a2172
projection_weight = backproject(sinogram, angles; geometry=geometry_weight);

# ╔═╡ bbb51355-fa72-4064-b10e-46e29f4b2809
simshow(projection_weight)

# ╔═╡ 53603134-07a7-477b-a74b-7c5a5ba1f84b
md"# Attenuated Radon Transform
The ray gets some attenuation with `exp(-μ*x)` where `x` is the distance traveled to the entry point of the circle. `μ` is in units of pixel.
"

# ╔═╡ b11cb860-3fab-4fc9-8921-bef32a8b7c12
projected_exp = backproject(sinogram, angles; geometry=geometry_extreme, μ=0.04);

# ╔═╡ 51561641-67a3-46d4-ae87-b896dc351a60
simshow(projected_exp)

# ╔═╡ 2289263b-f7a6-4347-8585-81993a273af3
md"# Testimage"

# ╔═╡ 3cc8ad0a-cb16-4b74-9c97-ae854931742b
begin
	img = Float32.(testimage("resolution_test_512"))
	simshow(img)
end

# ╔═╡ 5fc97390-6a83-49ab-9881-f1421476f80c
N2 = size(img, 1)

# ╔═╡ ddf85391-7a51-4f2e-8aa5-559b6b36938e
angles2 = range(0, 2π, 300)

# ╔═╡ 3a63c390-a02f-41d9-a123-edbb249522bb
geometry_extreme2 = RadonFlexibleCircle(N2, -(N2-1)÷2:(N2-1)÷2, zeros((N2-1,)))

# ╔═╡ 267098dc-790a-4995-b462-a69868193916
@time sg_img = radon(img, angles2, geometry=geometry_extreme2, μ=0.008);

# ╔═╡ ded517b8-77a8-4c91-af16-78f3934f8634
simshow(sg_img)

# ╔═╡ 29c25bd9-6359-489d-b0a3-ea767f832a02
img_backproject = backproject(sg_img, angles2, geometry=geometry_extreme2, μ=0.008);

# ╔═╡ 940d3b8b-1797-45f1-b493-02592a39eb19
simshow(img_backproject)

# ╔═╡ c7dcc341-0afe-4913-b0a8-1869d8f819b4
md"# How to reconstruct?
For proper reconstruction a optimization would be required in this case.
See the other notebooks how to set it up."

# ╔═╡ Cell order:
# ╠═cd52f230-c539-11ee-0c61-0d1afcad5372
# ╠═90b4b415-3ce9-4990-81f1-346c5015de21
# ╠═fab527ae-223c-47d0-93a5-ade5e5e74f4b
# ╟─3ebc1e08-39da-412e-8902-308a395b7c59
# ╠═20e238e8-357e-4896-98b0-3c6d0e34a48c
# ╠═9ed9ed78-11b3-491d-97d5-ce656d0bc270
# ╟─174067b0-c786-4392-8d06-a167f255e4bf
# ╠═22fe45cd-2f73-4a94-821e-57ee57d0116f
# ╠═07bec511-7802-45e4-b331-0d35e8f850c1
# ╠═9e5a58be-0de5-4820-bc3b-6dff931271a9
# ╟─511f6c64-6da5-4c8d-8a65-93d648b8887e
# ╠═01bcdaab-bc35-458e-8932-7554c83759f2
# ╠═dadbe1af-a570-4c66-bf8e-0fb2263f18ea
# ╠═97b8932e-7a5b-47fc-bf52-36cb9e1e999b
# ╠═27d5aac8-2528-46f2-a0dd-91c3fe5d275c
# ╠═29c9d1ff-694c-4fff-b551-836cc9eaf347
# ╟─da663a25-6dcd-4664-9bd1-c84970e58346
# ╠═65d32c65-6e1f-417b-aba3-3c34dac35e05
# ╠═37d760fa-74e6-47d1-b8e6-3315c1747b4c
# ╠═878121c5-4ad5-477b-88c7-f53df7510052
# ╟─afe43c8c-cb98-411b-af8f-1228983ee2e0
# ╠═7b0c8263-50d0-4569-96cb-297b4746ece3
# ╠═0756e11b-35ca-4eef-be48-52b8b5c098cd
# ╠═1e22032d-d57d-42d3-a9c5-9f2e94531346
# ╟─17d64cfb-2aca-4998-8028-cbc8509f5459
# ╠═72209fb0-e671-402a-adc4-bef553d81721
# ╠═45b82171-c581-4fcf-9695-bc51464a2172
# ╠═bbb51355-fa72-4064-b10e-46e29f4b2809
# ╟─53603134-07a7-477b-a74b-7c5a5ba1f84b
# ╠═b11cb860-3fab-4fc9-8921-bef32a8b7c12
# ╠═51561641-67a3-46d4-ae87-b896dc351a60
# ╟─2289263b-f7a6-4347-8585-81993a273af3
# ╠═3cc8ad0a-cb16-4b74-9c97-ae854931742b
# ╠═5fc97390-6a83-49ab-9881-f1421476f80c
# ╠═ddf85391-7a51-4f2e-8aa5-559b6b36938e
# ╠═3a63c390-a02f-41d9-a123-edbb249522bb
# ╠═267098dc-790a-4995-b462-a69868193916
# ╠═ded517b8-77a8-4c91-af16-78f3934f8634
# ╠═29c25bd9-6359-489d-b0a3-ea767f832a02
# ╠═940d3b8b-1797-45f1-b493-02592a39eb19
# ╟─c7dcc341-0afe-4913-b0a8-1869d8f819b4
