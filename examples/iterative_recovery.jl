### A Pluto.jl notebook ###
# v0.19.26

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

# ╔═╡ 33e4510a-1041-11ee-21af-0b8f937315e4
begin
	begin
		using Pkg
		Pkg.activate(".")
		Pkg.develop(path="../.")
		using Revise
	end
end

# ╔═╡ 9a52fab3-686b-489f-8f83-0de1b95ad04d
using RadonKA, ImageShow, IndexFunArrays, PlutoUI, CUDA, Plots

# ╔═╡ 06c3df4e-b1cd-4637-be86-ab5b018512a3
begin
	img = zeros(Float32, (512,512, 1))
	
	img .+= rr2(img, offset=(100, 100, 0) .+(70, 100, 1)) .< 30^2
	#img .-= rr2(img, offset=(70, 100)) .< 20^2
	img .+= box(img, (10, 70, 1), offset=(100, 100, 0) .+ (130, 80, 1))
	#img .+= box(img, (1, 1), offset=(130, 100))
	#img .+= box(img, (1, 1), offset=(110, 100))
	img_c = CuArray(img)
end;

# ╔═╡ 30f1b412-ebff-433b-bcbf-4bd0daca5c4c
simshow(img[:, :, 1])

# ╔═╡ 7485b19b-48c5-4f0f-bda2-f04b08b6756c


# ╔═╡ b765c575-85d6-4a3d-bf85-7cb9e9bb7d0d
θs = range(0f0 - pi/2, 1 * Float32(π) - pi/2, 352)

# ╔═╡ 612a9bd6-b5cd-48f5-9cce-37bb3ca72752
CUDA.@time CUDA.@sync sinogram_c = radon(img_c, θs, 0.003f0, backend=CUDABackend());

# ╔═╡ 305a6698-1cc8-4e90-bd83-4d8da64cc1fd
simshow(Array(sinogram_c[:, :, 1]))

# ╔═╡ e870e90d-d7dd-46f3-b6bd-36526fd54fbc
function iterative(sinogram, θs, N_iter=10)

	ir = let θs=θs, backend=CUDABackend()
		function ir(x)
			iradon(x, θs, backend=backend)
		end
	end

	r = let θs=θs, backend=CUDABackend()
		function r(x)
			radon(x, θs, backend=backend)
		end
	end


	rrr = CuArray(rr(Float32, (size(sinogram, 1), size(sinogram, 1))) .< size(sinogram, 1)÷2 - 3)
	
	obj = ir(sinogram)


	for i in 1:N_iter
		fwd = r(obj)
		diff = sinogram .- fwd
		
		bwd = ir(diff)
		obj .+= 2f-6 .* bwd
		obj .*= rrr
	end

	return obj
end

# ╔═╡ 9970d698-4b3f-4f8e-9724-d4ca90cfc995
rec = iterative(sinogram_c, θs, 50)

# ╔═╡ 0065c13a-5726-4442-9ac0-aa44b20c69fc
plot(Array(radon(rec, θs, backend=CUDABackend()))[:, 1, 1])

# ╔═╡ 76d6ea99-6b27-4a1e-babf-8301d6d3780d
simshow(Array(rec[:, :, 1]))

# ╔═╡ 91895c53-699b-4e4a-8100-2c5e203c8a90
simshow(Array(iradon(sinogram_c, θs, backend=CUDABackend()))[:, :,1])

# ╔═╡ 0d9bed96-a6f9-44a7-b8ed-77d612afe206
sum(rec)

# ╔═╡ c314ac85-51f6-4564-87a9-837dbb4d1343


# ╔═╡ 95fe923d-4004-4faa-9467-3585b663bad6


# ╔═╡ f5d5c560-fa40-4f22-b2a4-472996a558af
plot(sum(img[:, :, 1], dims=(2,)))

# ╔═╡ ba3bb7b6-366b-4c03-8192-4e8fcee2e1e4
begin
	plot(radon(img, [0])[:, 1, 1])
	plot!(sum(img[:, :, 1], dims=(2,)))
end

# ╔═╡ 6848f10c-fdbe-4d58-bfa6-f1053257319c
begin
	plot(radon(img, [π])[:, 1, 1])
	plot!(reverse(sum(img[:, :, 1], dims=(2,))))
end

# ╔═╡ c034b026-8941-4662-a5c9-4143e22e4201
begin
	plot(radon(img, [π + pi/2])[:, 1, 1])
	plot!(reverse(sum(img[:, :, 1], dims=(1,)))')
end

# ╔═╡ e16ad79e-4cb2-40df-8806-ef1df73e8ce2
sample = (rr(Float32, (256, 256,1), offset=(129, 100, 1)) .<= 5) * 1f0

# ╔═╡ ea67e3e4-fd00-4d41-ad49-ed05c6beb3b9
simshow(sample[:, :,1])

# ╔═╡ ee900962-48e6-43c4-846c-08b394c05728
@bind angle Slider(range(0, 2pi, 1000), show_value=true)

# ╔═╡ dcd872d5-79fd-4654-99dc-30cd4c586b33
begin
	plot(radon(sample, [0])[:])
	plot!(radon(sample, [angle])[:])
end

# ╔═╡ 73d64003-27bc-4990-89fd-646ba54cd41b
begin
	sinogram2 = zeros((500, 3,1))
	sinogram2[100:102, :,1] .= 1
end

# ╔═╡ be32bfb3-115e-44ed-a61f-369bb5aab1aa
simshow(iradon(sinogram2, [pi/4, 0])[:, :, 1])

# ╔═╡ d16465b7-05e6-44cf-8bb0-bf754eef5b2c
function iter!(buffer, img, θs; filtered=true, clip_sinogram=true)
	sinogram = radon(img, θs)
	sinogram ./= maximum(sinogram)

	if clip_sinogram
		sinogram .= max.(sinogram, 0)
	end
	
	img_recon = iradon(sinogram, θs)
	img_recon ./= maximum(img_recon)
	
	buffer .= max.(img_recon, 0)
	return buffer, sinogram
end

# ╔═╡ 145e0f23-2072-4473-b62f-00ae9f184963
function optimize2(img, θs, thresholds=(0.65, 0.75); N_iter = 2)
	N = size(img, 1)

	guess = copy(img)
	notobject = findall(iszero.(img))
	isobject = findall(isone.(img))
	
	buffer = copy(img)
	#s = radon(guess, θs, det)
	tmp, s = iter!(buffer, guess, θs, clip_sinogram=true)
	for i in 1:N_iter
		#tmp, s = iter!(buffer, guess, θs, det, rec_det,filtered=false, clip_sinogram=true)
		guess[notobject] .-= max.(0, tmp[notobject] .- thresholds[1])

		tmp, s = iter!(buffer, guess, θs, clip_sinogram=true)

		guess[isobject] .+= max.(0, thresholds[2] .- tmp[isobject])
	end

	return guess, s#radon(guess, θs, det)
end 

# ╔═╡ 6344079a-df81-4ba6-b5bb-4512186f4fb5
opt, s = optimize2(img, θs, N_iter=20)

# ╔═╡ bdf43e3c-fce4-4121-8603-d6132d74db40
simshow(opt[:, :, 1])

# ╔═╡ b4f5b56a-ed4d-4fc4-a0e9-260217118db2
begin
	printed = iradon(s, θs)[:, :, 1]
	printed ./= maximum(printed)
end

# ╔═╡ 65766d4a-f19c-43f7-9943-b8cc15e8a52f
simshow(printed .> 0.7)

# ╔═╡ 826547cb-5a2e-4392-bca8-51e476f87bce
simshow([printed .> 0.7 img[:, :, 1] (1 .- ((printed .> 0.7) .≈ img[:, :, 1]))])

# ╔═╡ Cell order:
# ╠═33e4510a-1041-11ee-21af-0b8f937315e4
# ╠═9a52fab3-686b-489f-8f83-0de1b95ad04d
# ╠═06c3df4e-b1cd-4637-be86-ab5b018512a3
# ╠═30f1b412-ebff-433b-bcbf-4bd0daca5c4c
# ╠═7485b19b-48c5-4f0f-bda2-f04b08b6756c
# ╠═b765c575-85d6-4a3d-bf85-7cb9e9bb7d0d
# ╠═612a9bd6-b5cd-48f5-9cce-37bb3ca72752
# ╠═305a6698-1cc8-4e90-bd83-4d8da64cc1fd
# ╠═e870e90d-d7dd-46f3-b6bd-36526fd54fbc
# ╠═9970d698-4b3f-4f8e-9724-d4ca90cfc995
# ╠═0065c13a-5726-4442-9ac0-aa44b20c69fc
# ╠═76d6ea99-6b27-4a1e-babf-8301d6d3780d
# ╠═91895c53-699b-4e4a-8100-2c5e203c8a90
# ╠═0d9bed96-a6f9-44a7-b8ed-77d612afe206
# ╠═c314ac85-51f6-4564-87a9-837dbb4d1343
# ╠═95fe923d-4004-4faa-9467-3585b663bad6
# ╠═f5d5c560-fa40-4f22-b2a4-472996a558af
# ╠═ba3bb7b6-366b-4c03-8192-4e8fcee2e1e4
# ╠═6848f10c-fdbe-4d58-bfa6-f1053257319c
# ╠═c034b026-8941-4662-a5c9-4143e22e4201
# ╠═e16ad79e-4cb2-40df-8806-ef1df73e8ce2
# ╠═ea67e3e4-fd00-4d41-ad49-ed05c6beb3b9
# ╠═dcd872d5-79fd-4654-99dc-30cd4c586b33
# ╠═ee900962-48e6-43c4-846c-08b394c05728
# ╠═73d64003-27bc-4990-89fd-646ba54cd41b
# ╠═be32bfb3-115e-44ed-a61f-369bb5aab1aa
# ╠═d16465b7-05e6-44cf-8bb0-bf754eef5b2c
# ╠═145e0f23-2072-4473-b62f-00ae9f184963
# ╠═6344079a-df81-4ba6-b5bb-4512186f4fb5
# ╠═bdf43e3c-fce4-4121-8603-d6132d74db40
# ╠═b4f5b56a-ed4d-4fc4-a0e9-260217118db2
# ╠═65766d4a-f19c-43f7-9943-b8cc15e8a52f
# ╠═826547cb-5a2e-4392-bca8-51e476f87bce
