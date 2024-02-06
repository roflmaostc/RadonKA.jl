### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

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

# ╔═╡ e90a05dd-5781-44ef-9b39-5b1ee85ca477
using BenchmarkTools

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

# ╔═╡ 23288c14-cd27-4bca-b28a-8fb02bd58bd8


# ╔═╡ f80cbda7-1efd-4323-913b-fb7cdcf7fcff
# ╠═╡ disabled = true
#=╠═╡
begin
	img2 = zeros((10,10,1))
	img2[6,5,1] = 3
	img2[6,6,1] = 1
end
  ╠═╡ =#

# ╔═╡ 59c69bee-e15b-4e43-b824-7c588c3aabf7
begin
	img2 = zeros((8,8,1))
	img2[5,4,1] = 3
	img2[5,5,1] = 1
end

# ╔═╡ b8c43ce0-9a08-489c-a8bc-0f9d18473369
img2

# ╔═╡ 51d38dd3-fc3b-4c14-9b80-e6b39051ac51
simshow(img2[:, :, 1])

# ╔═╡ c508389f-a605-4457-9360-410a8021b46c
Revise.errors()

# ╔═╡ c4a249b8-ac6d-4389-91c0-7df12954871f
@time RadonKA.radon2(randn(Float32, (100, 100)), range(0, 2*π, 100))

# ╔═╡ 29a1afa0-7225-4851-98b1-17b06606ba3a
Revise.errors()

# ╔═╡ aab7ddad-8c54-4c6c-9096-0f043dc2716a
x = abs.(ones((100,100)))

# ╔═╡ e575b9c3-5410-4e9f-9ccf-7809edae4721
a = [0]

# ╔═╡ 80532842-b487-4cc1-9798-f0aa3762034a
simshow(x)

# ╔═╡ bbbc01c6-c70d-42bf-969b-7b4fc4bbdfd6
g = RadonParallelCircle(size(x,1), -(size(x,1)-1)÷2:(size(x,1)-1)÷2)

# ╔═╡ 46956618-d12f-4096-a91d-e4fdfa32a7c7
simshow(iradon2(radon2(x, a; geometry=g), a; geometry=g, μ=1/30))

# ╔═╡ 57fca1d7-cf47-4a59-afc8-bf27497d27e4
Revise.errors()

# ╔═╡ 29ea08cf-f148-4ad6-aad1-f11d51cdfe82


# ╔═╡ ae0912af-af3a-4049-ba7e-6a4d6d61af35
Revise.errors()

# ╔═╡ c5a80f13-736e-4650-8d70-0c0d28914a37


# ╔═╡ badba484-fedc-4251-8095-01860c826ee2
Revise.errors()

# ╔═╡ ea813d04-865e-4f3e-af1c-b50c3f40da54
Revise.errors()

# ╔═╡ 28ee6c01-a5dd-47f0-a7b0-177523183c3a
RadonKA.next_cell_intersection

# ╔═╡ 6ca44a82-2104-4959-b2be-d1d222685353
# ╠═╡ disabled = true
#=╠═╡
simshow(radon(img,range(0f0, 2f0*π, 100))[:, :, 1])
  ╠═╡ =#

# ╔═╡ 6f639b7e-4d50-4696-b566-37b28072a168
# ╠═╡ disabled = true
#=╠═╡
@btime sinogram2 = RadonKA.radon2($img2, range(0, 2π, 20))
  ╠═╡ =#

# ╔═╡ 71789bcf-9c42-4b78-897b-fa254cdc98e4
# ╠═╡ disabled = true
#=╠═╡
@btime sinogram2 = RadonKA.radon($img2, range(0, 2π, 20))
  ╠═╡ =#

# ╔═╡ df04736c-17fe-4fd4-b53e-46ac5afbc6b4
# ╠═╡ disabled = true
#=╠═╡
@time radon(img2, range(0, 2π, 20));
  ╠═╡ =#

# ╔═╡ ceda298a-43b3-4012-a9cc-0d5276678532
# ╠═╡ disabled = true
#=╠═╡
simshow(radon(img2, range(0, 2π, 50))[:, :, 1])
  ╠═╡ =#

# ╔═╡ 190129d1-984c-4582-9d44-880f40b53fcf
Revise.errors()

# ╔═╡ d0e59091-7a0e-4828-8c17-7c2e643ef4d5
1 ≤ 3 ≤ 5

# ╔═╡ 7d45d80d-d7f4-4db8-9326-1d036ee04881
RadonKA.RadonCircle(nothing,[1,2])

# ╔═╡ d25c1381-baf1-429b-8150-622b8f731d83
md"# Example Image"

# ╔═╡ ecd4662c-639a-49fb-b947-b759a733ba22


# ╔═╡ d8d230e5-f870-46cf-a254-47248e3953cd


# ╔═╡ bd921606-320b-43d8-8355-e7014037d085


# ╔═╡ 6932f9ea-65fc-408a-b604-65213a17a776


# ╔═╡ 80535445-0bc8-4afe-a497-c9f764428749


# ╔═╡ 04ccf689-b0f6-4fc3-bcb0-9dfc9697d0ec


# ╔═╡ 2ccd4d32-269a-43e3-9967-5bf173b18e45


# ╔═╡ 7e40f887-ae05-40f1-b417-5536c8f97b09


# ╔═╡ bfbba881-ee79-4ba1-8aa8-9c94955e7d87


# ╔═╡ c0b3a91c-10f0-45cc-a227-107a7b5d05f2


# ╔═╡ ed576ff5-1b51-4ff2-b3bf-16cb62553f86


# ╔═╡ 54208d78-cf55-41d7-b4bf-6d1ab4927bbb
begin
	N = 512
	N_z = 5
	img = box(Float32, (N, N, N_z), (N ÷4, N ÷ 4, 20), offset=(N ÷ 2 + 60, N ÷ 2 -50, N_z ÷ 2)) |> collect

	#img = box(Float32, (N, N, N_z), (1, 1, 1)) |> collect
	#img = box(Float32, (N, N, N_z), (N ÷2, N ÷ 2, 50), offset=(N ÷ 2 - 50, N ÷ 2 + 50, N_z ÷ 2)) |> collect
	
	img .+= 0.0f0 .+ (rr2(Float32, (N, N, N_z), offset=(180, 210, N_z÷2)) .< 30 .^2)
	img[:, :, 1] .= testimage("resolution_test_512")
	#img = box(Float32, (100, 100), (3, 3), offset=(51, 51)) |> collect
end;

# ╔═╡ dccc67e0-411f-40d8-b28b-33a1701230bc
@benchmark RadonKA.radon($img, Float32.(range(0, 2*π, 100)))

# ╔═╡ 72074d52-9ce0-4a1c-9b02-cbc593fadf66
sinogram = @time RadonKA.radon2(img, Float32.(range(0, 2*π, 100)));

# ╔═╡ b5f14855-5f26-490e-92cc-e5345d024a70
simshow(iradon(sinogram, range(0, 2*π, 100)))[:, :, 1]

# ╔═╡ 7185a544-8a84-426b-bb3f-7b227ca865ff
simshow(iradon2(sinogram, range(0, 2*π, 100)))[:, :, 1]

# ╔═╡ d12b4d0f-dea0-448e-9982-6d1d8c410bf8
@benchmark RadonKA.radon2($img, Float32.(range(0, 2*π, 100)))

# ╔═╡ ce5aa941-b7bb-41f9-96b4-b6e1fe052bc2
@benchmark RadonKA.radon($img, Float32.(range(0, 2*π, 100)))

# ╔═╡ 7ad93d26-6041-46ed-9fb9-d116ea2e5608
@benchmark RadonKA.radon2($img, Float32.(range(0, 2*π, 100)))

# ╔═╡ 14074b7b-c937-4dfc-896c-fbe9d4f226de
@benchmark RadonKA.radon2($img, Float32.(range(0, 2*π, 100)), μ=0.1f0)

# ╔═╡ b2c6a7fb-0d8c-45c3-8de4-0b6ae09450cb
@benchmark RadonKA.radon2($img, Float32.(range(0, 2*π, 100)))

# ╔═╡ 117f1b48-c96b-48d3-8940-d6639e29ef76
@time sinogram3 = RadonKA.radon(img, range(0, 2*π, 1000))

# ╔═╡ f7dbad0d-ba37-473c-82e5-0fa441fc66a6
simshow(sinogram3[:, :, 1])

# ╔═╡ 9b3ac362-3194-4b56-ad41-8848eb17b2e2
@time sinogram2 = RadonKA.radon2(img, Float32.(range(0, 2*π, 1000)),
								geometry=RadonParallelCircle(-250:1:250));

# ╔═╡ 9315be14-1822-4091-a1ed-f570bc379414
simshow(sinogram2[:, :, 1])

# ╔═╡ 969fa003-3050-46d3-92b6-6df44e7f903f
simshow(sinogram2[:, :, 1])

# ╔═╡ c946a431-a3dc-4dd5-92e1-a5d24bd0c55e
sinogram2

# ╔═╡ 1393d029-66be-40aa-a2f9-f31317222575
img_c = togoc(img);

# ╔═╡ aecb76ad-d6a2-4d73-b8d5-8327e2448d5b
img_c

# ╔═╡ b87ee00b-d071-41be-a2c1-f7c32b6105e6
@benchmark CUDA.@sync RadonKA.radon($img_c, CuArray(range(0, 2*π, 100)))

# ╔═╡ 4e6924ba-c877-4e31-9e92-425dfbd0c3c6
sinogramc = RadonKA.radon2(img_c, CuArray(range(0f0, 2f0*π, 100)))

# ╔═╡ c0cd7d00-ae6e-4ba8-aad6-5b1c9c63736e
@benchmark CUDA.@sync RadonKA.iradon2($sinogramc, CuArray(range(0f0, 2f0*π, 100)))

# ╔═╡ 0cd70b79-b526-46c5-9f34-8056d1a3478a
@benchmark CUDA.@sync RadonKA.radon2($img_c, CuArray(range(0f0, 2f0*π, 100)))

# ╔═╡ 08e670a7-4d64-4a01-acc8-35a2f4fc7401
@benchmark CUDA.@sync RadonKA.radon2($img_c, range(0f0, 2f0*π, 100))

# ╔═╡ 1414d566-24d5-4d79-8806-9f3d42f9bbff
CUDA.@time RadonKA.radon2(img_c, collect(range(0f0, 2f0*π, 100)));

# ╔═╡ f9ad870a-204f-4e0d-8a60-2ddfa42d1506
CUDA.@time RadonKA.radon2(img_c, CuArray(range(0f0, 2f0*π, 100)));

# ╔═╡ 43da57cb-f777-4d0b-9754-1e863743f72f
CUDA.@time RadonKA.radon2(img_c, CuArray(range(0f0, 2f0*π, 100)), μ=0.10f0);

# ╔═╡ f994ed14-67a7-4ee1-878f-61927882f136
CUDA.@time RadonKA.radon2(img_c, CuArray(range(0f0, 2f0*π, 100)));

# ╔═╡ 1611bd14-6b59-4e37-8bfb-09c737cd6035
CUDA.@time RadonKA.radon(img_c, CuArray(range(0f0, 2f0*π, 100)));

# ╔═╡ 1e027ca9-5050-4b74-af5a-b92f28ee7205
@benchmark RadonKA.radon2($img_c, CuArray(range(0f0, 2f0*π, 100)),
geometry=RadonParallelCircle(-250.0:1.0:250.0))

# ╔═╡ 18f1fd94-bc04-4602-a7f4-b88e0dab2904
@benchmark RadonKA.radon2($img_c, CuArray(range(0f0, 2f0*π, 100)),
geometry=RadonParallelCircle(-250.0f0:1.0f0:250f0))

# ╔═╡ 419f16ba-f4fe-4421-91fe-412d92d51df6
@code_warntype RadonKA.radon2(img_c, CuArray(range(0f0, 2f0*π, 100)));

# ╔═╡ 3ce983fc-fd0a-48bf-b2ed-a39457f877d0
size(img_c)

# ╔═╡ 01b4b8f8-37d5-425f-975e-ebb3890d8624
simshow(img[:, :, 1])

# ╔═╡ Cell order:
# ╠═6f6c8d28-5a54-440e-9b7a-52e1fce959ab
# ╠═4eb3148e-8f8b-11ee-3cfe-854d3bd5cc80
# ╠═b336e55e-0be4-422f-b48a-0a2242cb0915
# ╠═1311e853-c4cd-42bb-8bf3-5e0d564bf9c5
# ╠═e90a05dd-5781-44ef-9b39-5b1ee85ca477
# ╟─f5e2024b-deaf-4344-b610-a4b956abbfaa
# ╠═03bccb92-b47f-477a-9bdb-74cc404da690
# ╠═ef92457a-87c0-43bf-a046-9fe82afbbe13
# ╟─810aebe4-2c6e-4ba6-916b-9e4306df33c9
# ╠═23288c14-cd27-4bca-b28a-8fb02bd58bd8
# ╠═f80cbda7-1efd-4323-913b-fb7cdcf7fcff
# ╠═59c69bee-e15b-4e43-b824-7c588c3aabf7
# ╠═b8c43ce0-9a08-489c-a8bc-0f9d18473369
# ╠═51d38dd3-fc3b-4c14-9b80-e6b39051ac51
# ╠═c508389f-a605-4457-9360-410a8021b46c
# ╠═aecb76ad-d6a2-4d73-b8d5-8327e2448d5b
# ╠═dccc67e0-411f-40d8-b28b-33a1701230bc
# ╠═c4a249b8-ac6d-4389-91c0-7df12954871f
# ╠═29a1afa0-7225-4851-98b1-17b06606ba3a
# ╠═aab7ddad-8c54-4c6c-9096-0f043dc2716a
# ╠═e575b9c3-5410-4e9f-9ccf-7809edae4721
# ╠═80532842-b487-4cc1-9798-f0aa3762034a
# ╠═bbbc01c6-c70d-42bf-969b-7b4fc4bbdfd6
# ╠═46956618-d12f-4096-a91d-e4fdfa32a7c7
# ╠═b5f14855-5f26-490e-92cc-e5345d024a70
# ╠═7185a544-8a84-426b-bb3f-7b227ca865ff
# ╠═72074d52-9ce0-4a1c-9b02-cbc593fadf66
# ╠═57fca1d7-cf47-4a59-afc8-bf27497d27e4
# ╠═d12b4d0f-dea0-448e-9982-6d1d8c410bf8
# ╠═ce5aa941-b7bb-41f9-96b4-b6e1fe052bc2
# ╠═b87ee00b-d071-41be-a2c1-f7c32b6105e6
# ╠═29ea08cf-f148-4ad6-aad1-f11d51cdfe82
# ╠═4e6924ba-c877-4e31-9e92-425dfbd0c3c6
# ╠═c0cd7d00-ae6e-4ba8-aad6-5b1c9c63736e
# ╠═0cd70b79-b526-46c5-9f34-8056d1a3478a
# ╠═08e670a7-4d64-4a01-acc8-35a2f4fc7401
# ╠═1414d566-24d5-4d79-8806-9f3d42f9bbff
# ╠═f9ad870a-204f-4e0d-8a60-2ddfa42d1506
# ╠═43da57cb-f777-4d0b-9754-1e863743f72f
# ╠═f994ed14-67a7-4ee1-878f-61927882f136
# ╠═1611bd14-6b59-4e37-8bfb-09c737cd6035
# ╠═1e027ca9-5050-4b74-af5a-b92f28ee7205
# ╠═18f1fd94-bc04-4602-a7f4-b88e0dab2904
# ╠═419f16ba-f4fe-4421-91fe-412d92d51df6
# ╠═ae0912af-af3a-4049-ba7e-6a4d6d61af35
# ╠═3ce983fc-fd0a-48bf-b2ed-a39457f877d0
# ╠═7ad93d26-6041-46ed-9fb9-d116ea2e5608
# ╠═14074b7b-c937-4dfc-896c-fbe9d4f226de
# ╠═c5a80f13-736e-4650-8d70-0c0d28914a37
# ╠═badba484-fedc-4251-8095-01860c826ee2
# ╠═ea813d04-865e-4f3e-af1c-b50c3f40da54
# ╠═b2c6a7fb-0d8c-45c3-8de4-0b6ae09450cb
# ╠═9315be14-1822-4091-a1ed-f570bc379414
# ╠═117f1b48-c96b-48d3-8940-d6639e29ef76
# ╠═9b3ac362-3194-4b56-ad41-8848eb17b2e2
# ╠═28ee6c01-a5dd-47f0-a7b0-177523183c3a
# ╠═f7dbad0d-ba37-473c-82e5-0fa441fc66a6
# ╠═969fa003-3050-46d3-92b6-6df44e7f903f
# ╠═6ca44a82-2104-4959-b2be-d1d222685353
# ╠═6f639b7e-4d50-4696-b566-37b28072a168
# ╠═71789bcf-9c42-4b78-897b-fa254cdc98e4
# ╠═df04736c-17fe-4fd4-b53e-46ac5afbc6b4
# ╠═ceda298a-43b3-4012-a9cc-0d5276678532
# ╠═c946a431-a3dc-4dd5-92e1-a5d24bd0c55e
# ╠═190129d1-984c-4582-9d44-880f40b53fcf
# ╠═d0e59091-7a0e-4828-8c17-7c2e643ef4d5
# ╠═7d45d80d-d7f4-4db8-9326-1d036ee04881
# ╠═d25c1381-baf1-429b-8150-622b8f731d83
# ╠═ecd4662c-639a-49fb-b947-b759a733ba22
# ╠═d8d230e5-f870-46cf-a254-47248e3953cd
# ╠═bd921606-320b-43d8-8355-e7014037d085
# ╠═6932f9ea-65fc-408a-b604-65213a17a776
# ╠═80535445-0bc8-4afe-a497-c9f764428749
# ╠═04ccf689-b0f6-4fc3-bcb0-9dfc9697d0ec
# ╠═2ccd4d32-269a-43e3-9967-5bf173b18e45
# ╠═7e40f887-ae05-40f1-b417-5536c8f97b09
# ╠═bfbba881-ee79-4ba1-8aa8-9c94955e7d87
# ╠═c0b3a91c-10f0-45cc-a227-107a7b5d05f2
# ╠═ed576ff5-1b51-4ff2-b3bf-16cb62553f86
# ╠═54208d78-cf55-41d7-b4bf-6d1ab4927bbb
# ╠═1393d029-66be-40aa-a2f9-f31317222575
# ╠═01b4b8f8-37d5-425f-975e-ebb3890d8624
