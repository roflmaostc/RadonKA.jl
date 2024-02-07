### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 1f8b489e-1028-11ee-3580-150afa948930
begin
	begin
		using Pkg
		Pkg.activate("../examples/.")
		Pkg.develop(path="../.")
		using Revise
	end
end

# ╔═╡ 2d1eb950-1c3b-4c3a-805b-1407d08813fb
using RadonKA, ImageShow, IndexFunArrays, PlutoUI, CUDA

# ╔═╡ 071b6c72-86ec-461a-ad7d-adf0fc696399
begin
	sinogram = zeros(Float32, (9, 2, 1))
	sinogram[5, :, 1] .= 1
end

# ╔═╡ 575487cf-faa2-41eb-9062-f030741be67b
simshow(sinogram[:, :, 1])

# ╔═╡ 7fcb40e4-4530-4ce1-a267-84b46309ab2a
@show iradon(sinogram, [0f0, π*0.5])

# ╔═╡ c4cd76e9-1685-40b4-bde6-2d1b6c03fa09
simshow(iradon(sinogram, [0f0, π*0.5]))[:, :, 1]

# ╔═╡ 49629e15-c5f4-42ff-b196-24e13124f8f3
@show I_r = iradon(sinogram, Float32[π/4 + π])

# ╔═╡ be118fec-3051-47a4-bf15-ac9d8d8c8e2e
sum(iradon(sinogram, Float32[2pi - pi/4]))

# ╔═╡ 80970ba6-a09f-46db-8b15-fa7226cdc400
sum(iradon(sinogram, Float32[0]))

# ╔═╡ c969f6d0-cfcc-4846-b970-9f6b734ac7cd
simshow(I_r[:, :, 1])

# ╔═╡ 0bbc2098-3bb8-44a7-8bca-361261267a28
exp(-3)

# ╔═╡ df8fdc51-f6e5-4757-9c77-51d1590dfdb1
begin
	img = zeros((10, 10, 1))
	
	img[8,5, 1] = 1
end

# ╔═╡ 1615a095-dbfb-4319-87e6-35a950ac4fe7
simshow(img[:, :, 1])

# ╔═╡ 2435a78b-5142-483e-afaf-6091b78724a5
begin
	sinogram_big = zeros(Float32, (29, 2, 1))
	sinogram_big[20, :, 1] .= 1
end

# ╔═╡ 8d9b6ea3-bbc0-48c1-a824-9749d6b8df0f
sum(iradon(sinogram_big, [0f0])[:, :, 1])

# ╔═╡ 73a756d9-9984-4452-b90a-9cced7959813
sum(iradon(sinogram_big, [pi/4f0])[:, :, 1])

# ╔═╡ b1607e05-d9ac-46cc-83b0-0f2b2c433059
simshow(iradon(sinogram_big, [3.4f0])[:, :, 1])

# ╔═╡ 773e3961-e398-4d4b-a5e6-76b457561976
simshow(iradon(sinogram_big, [0])[:, :, 1])

# ╔═╡ 5103d58e-34c8-4b9c-a0ab-c4f88f436186
md"# Radon"

# ╔═╡ 372f9ed3-a1d0-4368-bdba-800b4650af9e
begin
	array = zeros((16, 16,1))
	array[12,11] = 1
end

# ╔═╡ 2312af02-941f-4ab2-b8b8-02ba437bf5d0
simshow(array[:, :, 1] .+ 1im .* ((size(array, 1) / 2 - 1).>= rr(size(array)[1:2])))

# ╔═╡ efe3724b-0ec0-48db-98ae-8ae7d924419a
angles = range(0, 2pi, 100)

# ╔═╡ a1a808eb-316d-4e6b-b42a-11c45b55ae36
sg = radon(array, angles)

# ╔═╡ 13d70988-d480-4915-b9e6-67622d7dbca8
simshow(sg[:, :, 1])

# ╔═╡ fba7aedc-612c-40fd-b7b2-ebb1c364faa2
begin
	theory = zeros(size(sg))

	i = 1
	for θ in angles
		cc = size(sg, 1) ÷ 2 + 1
		x,y = (11, 10) .- (cc)
	
		y,x = [cos(θ) sin(θ); -sin(θ) cos(θ)] * [x, y]

		c =  cc + x + 1f-8

		c1 = floor(Int, c)
		c2 = ceil(Int, c)

		theory[c1, i] += (c2 - c)
		theory[c2, i] += (c - c1)
		i += 1
	end
end

# ╔═╡ e8f4af35-d908-4b0d-b5cd-6332a6a78964
theory

# ╔═╡ a9719576-ce8a-4adf-95e2-18e748bd5a02
15 ÷ 2

# ╔═╡ 5bceb117-c839-4bfb-84fe-2a92be024fa6
size(sg)

# ╔═╡ d35fcf11-fe18-494e-b5af-c5cba4b0a1a3
simshow(theory[:, :, 1])

# ╔═╡ e44d43b1-e54e-44f7-bd5d-44bd6a79b8e4
simshow(sg[:, :, 1])

# ╔═╡ a3c4fd5b-a297-4875-8389-b42d82fc0dfc
≈(sg, theory, rtol=0.4)

# ╔═╡ 937726a1-64af-4fac-8d86-d115533183a0
# ╠═╡ disabled = true
#=╠═╡
begin
	array2 = zeros((16, 16,1))
	array2[12,9] = 1
	array2[12, 12] = 1
end
  ╠═╡ =#

# ╔═╡ 8ce53564-3bdf-4ab7-9ec5-cd4560fdd4be
#=╠═╡
simshow(array2[:, :, 1])
  ╠═╡ =#

# ╔═╡ 53baff4f-25a4-4659-9b08-ee0d6abf3d05
#=╠═╡
simshow(radon(array2, [pi/2])[:, :, 1])
  ╠═╡ =#

# ╔═╡ cfc733ba-7551-4a56-b6e6-c45210439b86
#=╠═╡
@show radon(array2, [0, pi/4, pi/2, pi])[:, :, 1]
  ╠═╡ =#

# ╔═╡ 7584e03f-2469-4d03-815a-aed5c52d8295
simshow(array[:, :, 1])

# ╔═╡ 00b91330-5aca-4c3f-9d35-3164fabaea04
sg2 = radon(array, [pi/2, pi + pi/2, 2pi+ pi/2])

# ╔═╡ 7c75d09d-317c-4788-9df1-2644ef4d19fe
rad2deg(2.356194490192345)

# ╔═╡ ef2d6f02-4796-4597-adb8-1fe82056e179
exp(-3)

# ╔═╡ 4717a19c-d674-4ccb-9e36-dc2103e472c2
exp(-9)

# ╔═╡ c11adcd4-a478-4bd6-bb5e-fa746f1e3f69
simshow(sg2[:, :, 1])

# ╔═╡ c242d6c2-b19c-4cc1-b7dd-d54394eddfc6
@show radon(array, range(0, pi, 8));

# ╔═╡ 08f287ca-21e9-4ffe-b21d-ec036aca39f4
a = radon(array, [0, pi/2, pi, pi + pi/2,2pi]);

# ╔═╡ 4a580750-d9ed-46a6-92a7-6066f4aebbab


# ╔═╡ f07ccb05-518d-43f2-bbdf-c2851fa1b6aa
maximum(a, dims=1)

# ╔═╡ c04e6df6-6f48-45f6-b08e-b5648aa37e70
exp(-7)

# ╔═╡ ccfd6d3d-5d11-46d4-be4b-52bc7dbd7292
simshow(a[:, :, 1])

# ╔═╡ d4772dd9-16d2-462a-8efb-16954af831aa
simshow(array[:, :, 1])

# ╔═╡ 18c39411-899a-445b-8433-b524cf395ea0
radon(zeros((128, 128, 1)), range(0, 2pi, 300))

# ╔═╡ 58e706a6-14f1-404c-9cfa-2544dd6df098
md"# Test filtered Backprojection"

# ╔═╡ ca30b03e-815b-496a-b704-0c8496ac6a02
begin
	array3 = zeros((32, 32));
	
	array3[10:15, 10:11] .= 1
	
	array3[10:12, 20:26] .= 1
end

# ╔═╡ 0e7ffeb5-da26-4b99-b0c1-d9ac9af4e4d0


# ╔═╡ bbf9207c-c816-4fa7-9ddf-e9a37209f583


# ╔═╡ 12a1165d-2aa4-4066-87cb-1cec156511e2
simshow(array3)

# ╔═╡ 26ccc58c-00e8-4d57-bc74-de6a3a544ab8
simshow(sinogram2[:,:,1])

# ╔═╡ 2a3fe939-77b0-4388-9b9e-f9bee9e35e1b
I_2 = iradon(sinogram2, angles2);

# ╔═╡ a33c5c99-00c3-437d-bfa7-2c0d12fb2339
simshow(I_2[:,:,1])

# ╔═╡ 660a58b7-c626-465c-ab9d-86549d8f7668
simshow(sinogram2)

# ╔═╡ ef6de020-ba4a-4dce-8a77-f706c1910270
array_filtered = filtered_backprojection(sinogram2, angles2)

# ╔═╡ ecb578ff-4dca-4260-9124-0bc17499d071
array_iradon = iradon(sinogram2, angles2)

# ╔═╡ 2adbe3a7-c0bf-4e2d-bdec-4f62e29055af
sum(array_iradon)

# ╔═╡ 65781501-0237-4caf-aa4e-4a83338e4658
simshow(array_filtered / sum(array_filtered))

# ╔═╡ 9ad1a6ef-8fa0-4abb-9ed8-81f39c702229
sum(array_filtered)

# ╔═╡ 0a88ffea-434c-4e49-a319-95ff6a5f19e4
sum(sinogram2) / 32

# ╔═╡ b20e4dd4-75f2-402e-8521-32757f73040a
sum(array3)

# ╔═╡ c73a49c1-3249-4b8c-a7ba-75c9477ba999
≈(array_filtered / sum(array_filtered) .+ 0.1, array3 / sum(array3) .+ 0.1, rtol=0.05)

# ╔═╡ fb6a1ca0-639f-4f42-a252-f0fe3975107c
simshow(.≈(array_filtered / sum(array_filtered) .+ 0.1, array3 / sum(array3) .+ 0.1, rtol=0.1))

# ╔═╡ d661d92c-c11b-429f-8666-56a217da4776
# ╠═╡ disabled = true
#=╠═╡
sinogram2 = radon(img, angles2)
  ╠═╡ =#

# ╔═╡ d67b47ad-f259-4efd-9afa-ebdb43fba34a
sinogram2 = radon(array3, angles2)

# ╔═╡ 01eed37a-d5f0-4c19-bcc9-445b41f44fb0
angles2 = range(0, π, 200);

# ╔═╡ e272e8e5-a294-47b2-a012-192c31179a4a
# ╠═╡ disabled = true
#=╠═╡
angles2 = range(0, π, 100)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═1f8b489e-1028-11ee-3580-150afa948930
# ╠═2d1eb950-1c3b-4c3a-805b-1407d08813fb
# ╠═071b6c72-86ec-461a-ad7d-adf0fc696399
# ╠═575487cf-faa2-41eb-9062-f030741be67b
# ╠═7fcb40e4-4530-4ce1-a267-84b46309ab2a
# ╠═c4cd76e9-1685-40b4-bde6-2d1b6c03fa09
# ╠═49629e15-c5f4-42ff-b196-24e13124f8f3
# ╠═be118fec-3051-47a4-bf15-ac9d8d8c8e2e
# ╠═80970ba6-a09f-46db-8b15-fa7226cdc400
# ╠═c969f6d0-cfcc-4846-b970-9f6b734ac7cd
# ╠═0bbc2098-3bb8-44a7-8bca-361261267a28
# ╠═df8fdc51-f6e5-4757-9c77-51d1590dfdb1
# ╠═e272e8e5-a294-47b2-a012-192c31179a4a
# ╠═1615a095-dbfb-4319-87e6-35a950ac4fe7
# ╠═d661d92c-c11b-429f-8666-56a217da4776
# ╠═26ccc58c-00e8-4d57-bc74-de6a3a544ab8
# ╠═2a3fe939-77b0-4388-9b9e-f9bee9e35e1b
# ╠═a33c5c99-00c3-437d-bfa7-2c0d12fb2339
# ╠═2435a78b-5142-483e-afaf-6091b78724a5
# ╠═8d9b6ea3-bbc0-48c1-a824-9749d6b8df0f
# ╠═73a756d9-9984-4452-b90a-9cced7959813
# ╠═b1607e05-d9ac-46cc-83b0-0f2b2c433059
# ╠═773e3961-e398-4d4b-a5e6-76b457561976
# ╟─5103d58e-34c8-4b9c-a0ab-c4f88f436186
# ╠═372f9ed3-a1d0-4368-bdba-800b4650af9e
# ╠═2312af02-941f-4ab2-b8b8-02ba437bf5d0
# ╠═efe3724b-0ec0-48db-98ae-8ae7d924419a
# ╠═a1a808eb-316d-4e6b-b42a-11c45b55ae36
# ╠═e8f4af35-d908-4b0d-b5cd-6332a6a78964
# ╠═13d70988-d480-4915-b9e6-67622d7dbca8
# ╠═fba7aedc-612c-40fd-b7b2-ebb1c364faa2
# ╠═a9719576-ce8a-4adf-95e2-18e748bd5a02
# ╠═5bceb117-c839-4bfb-84fe-2a92be024fa6
# ╠═d35fcf11-fe18-494e-b5af-c5cba4b0a1a3
# ╠═e44d43b1-e54e-44f7-bd5d-44bd6a79b8e4
# ╠═a3c4fd5b-a297-4875-8389-b42d82fc0dfc
# ╠═937726a1-64af-4fac-8d86-d115533183a0
# ╠═8ce53564-3bdf-4ab7-9ec5-cd4560fdd4be
# ╠═53baff4f-25a4-4659-9b08-ee0d6abf3d05
# ╠═cfc733ba-7551-4a56-b6e6-c45210439b86
# ╠═7584e03f-2469-4d03-815a-aed5c52d8295
# ╠═00b91330-5aca-4c3f-9d35-3164fabaea04
# ╠═7c75d09d-317c-4788-9df1-2644ef4d19fe
# ╠═ef2d6f02-4796-4597-adb8-1fe82056e179
# ╠═4717a19c-d674-4ccb-9e36-dc2103e472c2
# ╠═c11adcd4-a478-4bd6-bb5e-fa746f1e3f69
# ╠═c242d6c2-b19c-4cc1-b7dd-d54394eddfc6
# ╠═08f287ca-21e9-4ffe-b21d-ec036aca39f4
# ╠═4a580750-d9ed-46a6-92a7-6066f4aebbab
# ╠═f07ccb05-518d-43f2-bbdf-c2851fa1b6aa
# ╠═c04e6df6-6f48-45f6-b08e-b5648aa37e70
# ╠═ccfd6d3d-5d11-46d4-be4b-52bc7dbd7292
# ╠═d4772dd9-16d2-462a-8efb-16954af831aa
# ╠═18c39411-899a-445b-8433-b524cf395ea0
# ╟─58e706a6-14f1-404c-9cfa-2544dd6df098
# ╠═ca30b03e-815b-496a-b704-0c8496ac6a02
# ╠═0e7ffeb5-da26-4b99-b0c1-d9ac9af4e4d0
# ╠═bbf9207c-c816-4fa7-9ddf-e9a37209f583
# ╠═01eed37a-d5f0-4c19-bcc9-445b41f44fb0
# ╠═12a1165d-2aa4-4066-87cb-1cec156511e2
# ╠═d67b47ad-f259-4efd-9afa-ebdb43fba34a
# ╠═660a58b7-c626-465c-ab9d-86549d8f7668
# ╠═ef6de020-ba4a-4dce-8a77-f706c1910270
# ╠═ecb578ff-4dca-4260-9124-0bc17499d071
# ╠═2adbe3a7-c0bf-4e2d-bdec-4f62e29055af
# ╠═65781501-0237-4caf-aa4e-4a83338e4658
# ╠═9ad1a6ef-8fa0-4abb-9ed8-81f39c702229
# ╠═0a88ffea-434c-4e49-a319-95ff6a5f19e4
# ╠═b20e4dd4-75f2-402e-8521-32757f73040a
# ╠═c73a49c1-3249-4b8c-a7ba-75c9477ba999
# ╠═fb6a1ca0-639f-4f42-a252-f0fe3975107c
