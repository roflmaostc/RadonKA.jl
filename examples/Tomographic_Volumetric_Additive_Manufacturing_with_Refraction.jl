### A Pluto.jl notebook ###
# v0.19.36

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

# ╔═╡ 179d107e-b3be-11ee-0a6c-49f1bf2a10fd
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ e3a3dc83-6536-445f-b56b-d8ad62cd9a85
using TestImages, RadonKA, ImageShow, ImageIO, Noise, PlutoUI, BenchmarkTools, CUDA, Zygote, IndexFunArrays, FileIO, NDTools, Plots, ProgressMeter

# ╔═╡ eb4b7177-4f93-4728-8c91-4ff6a86f40cf
md"# Load packages and check CUDA"

# ╔═╡ be0df011-9ac4-46d7-8963-a8e339af6683


# ╔═╡ f3277f00-4162-43b2-98e0-53bcb508df1e
TableOfContents()

# ╔═╡ 42f31ceb-0f5d-46cc-b868-0314b41a5407
begin
	use_CUDA = Ref(CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ 78746412-00f2-40ad-9d24-e884d8dd7e77
md"""# CUDA is functional: $(CUDA.functional() ? "yes" : "no")"""

# ╔═╡ 980e9656-202d-4a47-837f-b2197fe5d3b4
md"# Load JOSS logo as target"

# ╔═╡ f0d86933-5eab-4d86-8909-c51ab9ca99d9
background = select_region(ImageShow.Gray.(load(download("https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/JOSS.png/459px-JOSS.png"))) .< 0.001, new_size=(700,700));

# ╔═╡ 36eab63f-5643-4b43-8e12-0f7d971c1040
img = select_region(.-ImageShow.Gray.(load(download("https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/JOSS.png/459px-JOSS.png"))), new_size=(700,700));

# ╔═╡ fc049fbc-4ae7-4256-b39f-27fdff98d45a
target = togoc(Float32.((img .- background) .> 0.1));

# ╔═╡ a6bdc90f-3c10-4524-9892-6e976d556a95
simshow(Array(target))

# ╔═╡ 066664f7-9ae5-4a4d-b541-663067cea361
md"# Specify Angles and ray endpoints"

# ╔═╡ 5ef75bb5-5ab0-4e98-9026-a8e2c5b7833d
angles = togoc(range(0, 2f0 * π, 1000));

# ╔═╡ 5aa03590-c552-410e-93aa-1e47dabd2f9e
N_s = size(target,1) - 1

# ╔═╡ 9d582f0d-7da4-41db-9369-fef7a6dbe590
range(-N_s/2 - 1, N_s / 2 - 1, N_s)

# ╔═╡ 6b91d2bd-1170-4ce5-bb20-10d69f8cd6d7
md"
The `ray_startpoints` and `ray_endpoints` indicate each the beginning and end position of the ray.

Each is calculated for the horizontal position on the left of the array and on the right.
"

# ╔═╡ af83cb81-49b8-41f4-a24e-c1e33e76d925
function distort_rays_vial(ys, R_outer, R_inner, n_vial, n_resin)
	#R₁ ist R_outer
	#R₂ is R_inner
	function quadratic_solve(a, b, c)
		(-b - √(b^2 - 4 * a * c)) / 2 / a
	end
	
	αs = asin.(ys ./ R_outer)
	βs = asin.(sin.(αs) ./ n_vial)
	xs = quadratic_solve.(1, -2 .* R_outer .* cos.(βs), R_outer^2 - R_inner^2)
	ϵs = acos.((-R_outer^2 .+ xs.^2 .+ R_inner^2) ./ 2 ./ R_inner ./ xs) .- π / 2

	β₂s = sign.(ys) .* (π / 2 .- ϵs)
	γs = asin.(n_vial .* sin.(β₂s) ./ n_resin)
	
	y₂s = sin.(γs) .* R_inner
	δ₁s = αs - βs
	δ₂s = β₂s - γs
	δ_ges = δ₁s + δ₂s

	Δh = sin.(δ₁s) .* xs
	Δxs = R_outer .- sqrt.(R_outer.^2 .- ys.^2)
	Δx2s = sqrt.(xs.^2 .- Δh.^2)
	y3s = tan.(δ₁s .+ δ₂s) .* (2*R_outer .- Δxs .- Δx2s)
	y4s = ys .- Δh .- y3s
	return ys .- Δh, y4s
end

# ╔═╡ 6721b65f-48ed-4dc7-99dd-c460bb124c1e
ray_startpoints, ray_endpoints = distort_rays_vial(range(-N_s/2 + 1, N_s / 2 - 1, N_s), 350, 340, 1.5, 1.4)

# ╔═╡ 4cdc5592-3b92-4e98-8718-d77dd939ebf1
md"## Ray distribution inside vial"

# ╔═╡ dc22dead-1b83-4787-a147-9811136af085
begin
	sinogram = zeros((size(target,1)-1, 1))
	sinogram[1:5:end] .= 1
end

# ╔═╡ 2f582074-44a6-405e-b572-4ed3819b3706
simshow(iradon(sinogram, [0.0]; ray_endpoints), γ=0.2)

# ╔═╡ 5d55464f-c7d9-4e45-a1e7-907c2137be95
md"# Define Optimization algorithm
See this paper: 
[*Charles M. Rackson, Kyle M. Champley, Joseph T. Toombs, Erika J. Fong, Vishal Bansal, Hayden K. Taylor, Maxim Shusteff, Robert R. McLeod, Object-space optimization of tomographic reconstructions for additive manufacturing*](https://www.sciencedirect.com/science/article/abs/pii/S2214860421005212)
"

# ╔═╡ 697d4c7d-697b-452c-8a6a-25f2a1089529
function iter!(buffer, img, θs, μ; clip_sinogram=true, ray_startpoints, ray_endpoints)
	sinogram = radon(img, θs, μ; ray_startpoints, ray_endpoints)

	if clip_sinogram
		sinogram .= max.(sinogram, 0)
	end
	
	img_recon = iradon(sinogram, θs, μ; ray_startpoints, ray_endpoints)
	img_recon ./= maximum(img_recon)
	buffer .= max.(img_recon, 0)
	return buffer, sinogram
end

# ╔═╡ 624858de-e7da-4e3c-91b5-6ab7b5e645d5
# see https://www.sciencedirect.com/science/article/abs/pii/S2214860421005212
function OSMO(img::AbstractArray{T}, θs, μ=nothing; 
			  thresholds=(0.65, 0.75), N_iter = 2,
				ray_startpoints, ray_endpoints) where T
	N = size(img, 1)
	guess = copy(img)

	notobject = togoc(iszero.(img))
	isobject = togoc(isone.(img))

	losses = T[]
	buffer = copy(img)
	tmp, s = iter!(buffer, guess, θs, μ; clip_sinogram=true, 						ray_startpoints, ray_endpoints)
	@showprogress for i in 1:N_iter
		guess[notobject] .-= max.(0, tmp[notobject] .- thresholds[1])
		tmp, s = iter!(buffer, guess, θs, μ; clip_sinogram=true,
						ray_startpoints, ray_endpoints)
		guess[isobject] .+= max.(0, thresholds[2] .- tmp[isobject])
	end
	
	printed = iradon(s, θs, μ; ray_startpoints, ray_endpoints)
	printed ./= maximum(printed)
	return guess, s, printed
end

# ╔═╡ d80f6c47-1c55-4a7e-9374-104806873a9a
md"# Run optimization
With a decent GPU (such as RTX 3060) it should take around ~20-30s.
With a 12 core multithreaded CPU around ~15min.
"

# ╔═╡ 42f6a082-66e0-4b0f-a962-c0e09c614e45
@mytime a_object, patterns, printed = OSMO(target, angles, 1/350f0; N_iter=300, ray_startpoints=togoc(ray_startpoints), ray_endpoints=togoc(ray_endpoints),
)

# ╔═╡ 05068957-20f4-4005-98e1-3f6eb19558a7


# ╔═╡ ec528b63-95eb-430b-b683-04e207dd61f0
simshow(Array(printed) .> 0.7, cmap=:turbo)

# ╔═╡ 43e4b941-b838-483d-9a8f-8897720f8b90
simshow(Array(printed), cmap=:turbo)

# ╔═╡ 7c998816-b4de-4dbf-977b-bdab9355be79
simshow(Array(iradon(patterns[:, 96:96], togoc(angles[96:96]), 1/350f0 ; ray_startpoints=togoc(ray_startpoints), ray_endpoints=togoc(ray_endpoints))), cmap=:turbo)

# ╔═╡ 28d8c378-b818-4d61-b9d4-a06ad7cc21e3
simshow(Array(iradon(patterns[:, 306:306], togoc(angles[306:306]), 1/350f0 ; ray_startpoints=togoc(ray_startpoints), ray_endpoints=togoc(ray_endpoints))), cmap=:turbo)

# ╔═╡ beb61bb1-3f7c-4301-92ca-5418bc0445cf
simshow(Array(iradon(patterns[:, 1:1], togoc(angles[1:1]), 1/350f0 ; ray_startpoints=togoc(ray_startpoints), ray_endpoints=togoc(ray_endpoints))), cmap=:turbo)

# ╔═╡ 88895d8d-432e-4337-a51c-12ac4e4a4325
@bind i Slider(1:800, show_value=true)

# ╔═╡ 61048e10-8e23-41a6-8c49-0b1a8c0adc24
simshow(Array(iradon(patterns[:, i:i], togoc(angles[i:i]), 1/350f0 ; ray_startpoints=togoc(ray_startpoints), ray_endpoints=togoc(ray_endpoints))), cmap=:turbo)

# ╔═╡ 21c0b369-a4b2-43ea-87b8-702474d88584
simshow(Array(patterns))

# ╔═╡ b53c8016-177f-4b1b-b07d-48cbd2b2fc65
function plot_histogram(target, object_printed, thresholds; yscale=:log10)
    # :stephist vs :barhist
    
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
	    linewidth=2, framestyle=:box, label=nothing, grid=false)
	plot(object_printed[target .== 0], seriestype=:stephist, bins=(0.0:0.01:1), xlim=(0.0, 1.0), label="dose distribution void", ylabel="voxel count", xlabel="normalized intensity",  ylim=(10, 500000),  linewidth=1, legend=:topleft, yscale=yscale, size=(500, 300))
	plot!(object_printed[target .== 1], seriestype=:stephist, bins=(0.0:0.01:1), xlim=(0.0, 1.0), label="dose distribution object", ylabel="voxel count", xlabel="normalized intensity",  ylim=(10, 500000),  linewidth=1, legend=:topleft, yscale=yscale, size=(500, 300))
	plot!([thresholds[1], thresholds[1]], [1, 10000_000], label="lower threshold", linewidth=3)
	plot!([thresholds[2], thresholds[2]], [1, 10000_000], label="upper threshold", linewidth=3)
	#plot!([chosen_threshold, chosen_threshold], [1, 30000000], label="chosen threshold", linewidth=3)
end

# ╔═╡ 03223eb4-3c15-4899-93bf-5440fb6b5c7a
plot_histogram(Array(target), Array(printed), (0.65, 0.75))

# ╔═╡ Cell order:
# ╠═179d107e-b3be-11ee-0a6c-49f1bf2a10fd
# ╟─eb4b7177-4f93-4728-8c91-4ff6a86f40cf
# ╠═e3a3dc83-6536-445f-b56b-d8ad62cd9a85
# ╠═be0df011-9ac4-46d7-8963-a8e339af6683
# ╠═f3277f00-4162-43b2-98e0-53bcb508df1e
# ╠═42f31ceb-0f5d-46cc-b868-0314b41a5407
# ╟─78746412-00f2-40ad-9d24-e884d8dd7e77
# ╟─980e9656-202d-4a47-837f-b2197fe5d3b4
# ╠═f0d86933-5eab-4d86-8909-c51ab9ca99d9
# ╠═36eab63f-5643-4b43-8e12-0f7d971c1040
# ╠═fc049fbc-4ae7-4256-b39f-27fdff98d45a
# ╠═a6bdc90f-3c10-4524-9892-6e976d556a95
# ╟─066664f7-9ae5-4a4d-b541-663067cea361
# ╠═5ef75bb5-5ab0-4e98-9026-a8e2c5b7833d
# ╠═5aa03590-c552-410e-93aa-1e47dabd2f9e
# ╠═9d582f0d-7da4-41db-9369-fef7a6dbe590
# ╟─6b91d2bd-1170-4ce5-bb20-10d69f8cd6d7
# ╠═6721b65f-48ed-4dc7-99dd-c460bb124c1e
# ╠═af83cb81-49b8-41f4-a24e-c1e33e76d925
# ╟─4cdc5592-3b92-4e98-8718-d77dd939ebf1
# ╠═dc22dead-1b83-4787-a147-9811136af085
# ╠═2f582074-44a6-405e-b572-4ed3819b3706
# ╟─5d55464f-c7d9-4e45-a1e7-907c2137be95
# ╠═697d4c7d-697b-452c-8a6a-25f2a1089529
# ╠═624858de-e7da-4e3c-91b5-6ab7b5e645d5
# ╠═d80f6c47-1c55-4a7e-9374-104806873a9a
# ╠═42f6a082-66e0-4b0f-a962-c0e09c614e45
# ╠═05068957-20f4-4005-98e1-3f6eb19558a7
# ╠═03223eb4-3c15-4899-93bf-5440fb6b5c7a
# ╠═ec528b63-95eb-430b-b683-04e207dd61f0
# ╠═43e4b941-b838-483d-9a8f-8897720f8b90
# ╠═7c998816-b4de-4dbf-977b-bdab9355be79
# ╠═28d8c378-b818-4d61-b9d4-a06ad7cc21e3
# ╠═beb61bb1-3f7c-4301-92ca-5418bc0445cf
# ╠═61048e10-8e23-41a6-8c49-0b1a8c0adc24
# ╟─88895d8d-432e-4337-a51c-12ac4e4a4325
# ╠═21c0b369-a4b2-43ea-87b8-702474d88584
# ╠═b53c8016-177f-4b1b-b07d-48cbd2b2fc65
