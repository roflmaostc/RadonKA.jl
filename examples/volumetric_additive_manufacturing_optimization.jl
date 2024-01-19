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

# ╔═╡ 453f6cca-91ea-11ee-0c48-b923a785e914
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ d956ede0-5a1f-4ed6-a819-1730f52a3536
using TestImages, RadonKA, ImageShow, ImageIO, Noise, PlutoUI, IndexFunArrays, FFTW

# ╔═╡ 9fe3b45a-3901-40e4-bad9-32ee3735d971
using Plots, FileIO

# ╔═╡ 57d629e6-4de8-4131-bd97-a505c31ab475
using NDTools, Statistics

# ╔═╡ 20f68941-4141-4d81-bae8-5ee9661dd80a
using CUDA, CUDA.CUDAKernels, KernelAbstractions

# ╔═╡ 310da375-4964-4165-aa6b-c50c151ea65e
begin
	plot_font = "Computer Modern"
	default(fontfamily=plot_font,
	        linewidth=2, framestyle=:box, label=nothing, grid=false)
	scalefontsizes(1.3)
end

# ╔═╡ a264aa76-2057-42d4-8dd6-254c3d0cdf44
# use CUDA if functional
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ 4678b348-f61e-4720-af21-7313c79365f7
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ fbd565e1-ea6c-441a-8ed4-39e65071b798
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 21f890e4-2bb7-4b06-bd79-5e8bf5cce6f5
md"# A sample we want to print"

# ╔═╡ 358dda4a-d589-4993-9d6f-d42bcddda1a3
# ╠═╡ disabled = true
#=╠═╡
begin
	img = zeros(Float32, (200,200))
	
	#img .+= rr2(img, offset=(70, 100)) .< 30^2
	#img .-= rr2(img, offset=(70, 100)) .< 20^2
	img .+= box(img, (10, 70), offset=(130, 80))
	#img .+= box(img, (1, 1), offset=(130, 100))
	#img .+= box(img, (1, 1), offset=(110, 100))
	img .+= rr2(img, offset=(70, 100)) .< 30^2
end;
  ╠═╡ =#

# ╔═╡ b6ed1974-d562-47d4-a342-66eb474bc556
# ╠═╡ disabled = true
#=╠═╡
function make_grid_hole()
	sz = (512, 512)

	img = zeros(Bool, sz)
	img = img .|| Bool.(box(Int, sz, (600, 70), offset=(251, 100)))
	img = img .|| Bool.(box(Int, sz, (70, 600), offset=(100, 251)))
	
	img = img .|| Bool.(box(Int, sz, (30, 600), offset=(170, 251)))
	img = img .|| Bool.(box(Int, sz, (600, 30), offset=(251, 170)))


	img = img .|| Bool.(box(Int, sz, (30, 600), offset=(420, 251)))
	img = img .|| Bool.(box(Int, sz, (600, 30), offset=(251, 420)))
		
	img = img .|| Bool.(box(Int, sz, (15, 600), offset=(200, 251)))
	img = img .|| Bool.(box(Int, sz, (600, 15), offset=(251, 200)))


		
	img = img .|| Bool.(box(Int, sz, (5, 600), offset=(240, 251)))
	img = img .|| Bool.(box(Int, sz, (600, 5), offset=(251, 240)))


	img = img .|| Bool.(rr2(Int, sz, offset=(300, 300)) .<= 80 .^2)
	img = img .- Bool.(rr2(Int, sz, offset=(300, 300)) .<= 30 .^2)

	# img = img .|| Bool.(box(Int, sz, (30, 50), offset=(50, 50)))

	return Float32.(img) .* (rr2(Int,sz) .<= 250 ^2)
end
  ╠═╡ =#

# ╔═╡ 79c1b042-2b9d-43db-9525-d66418db5da2
img = Float32.((Float32.(Gray.(select_region(load("/home/felix/Documents/code/Printing_Siemens_Star_Target/siemens_star_512px.png"), new_size=(512, 512)))) .* (rr2(Float32, (512, 512)) .<= 255 .^2)) .> 0.5);

# ╔═╡ aa1cccec-f846-4b1e-9aae-81e939cd3b8c
md"""# Algorithm

See this paper: Rackson, Charles M., et al. "Object-space optimization of tomographic reconstructions for additive manufacturing." Additive Manufacturing 48 (2021): 102367.
"""

# ╔═╡ 0c1abf70-8004-460a-b008-750050f6e718
function iter!(buffer, img, θs, μ; filtered=true, clip_sinogram=true)
	sinogram = radon(img, θs, μ)
	if filtered
		sinogram = ramp_filter(sinogram)
	end
	
	if clip_sinogram
		sinogram .= max.(sinogram, 0)
	end
	
	img_recon = iradon(sinogram, θs, μ)
	img_recon ./= maximum(img_recon)
	buffer .= max.(img_recon, 0)
	return buffer, sinogram
end

# ╔═╡ 5ac76842-f5a3-42c0-9764-9b20f98ab1d5
function optimize(img::AbstractArray{T}, θs, μ=nothing; thresholds=(0.65, 0.75), N_iter = 2) where T
	N = size(img, 1)
	fx = (-N / 2):1:(N /2 -1)
	R2D = similar(img)
	R2D .= sqrt.(fx'.^2 .+ fx.^2)

	p = plan_fft(img, (1,2))
	guess = max.(0, real.(inv(p) * ((p * img) .* ifftshift(R2D, (1,2)))))
	guess ./= maximum(guess)

	loss(x) = (sum(max.(0,thresholds[2] .- x[isobject])) + sum(max.(0, x[notobject] .- thresholds[1]))) / length(x)
	#guess = copy(img)
	notobject = iszero.(img)
	isobject = isone.(img)

	losses = T[]
	buffer = copy(img)
	tmp, s = iter!(buffer, guess, θs, μ; filtered=false, clip_sinogram=true)
	for i in 1:N_iter
		guess[notobject] .-= max.(0, tmp[notobject] .- thresholds[1])

		tmp, s = iter!(buffer, guess, θs, μ; filtered=false, clip_sinogram=true)

		guess[isobject] .+= max.(0, thresholds[2] .- tmp[isobject])

		push!(losses, loss(tmp))
	end
	
	printed = iradon(s, θs, μ)
	printed ./= maximum(printed)
	return guess, s, printed, losses
end 

# ╔═╡ e19a1fb6-d237-44c8-a1b1-45e38aebf948
function errors(printed, isobject, notobject, thresholds)
	mid_thresh = (thresholds[2] + thresholds[1]) / 2
	W_not = sum(printed[notobject] .> thresholds[1])
	W_not_is = sum(printed[notobject] .> mid_thresh)
	W_is = sum(printed[isobject] .< thresholds[2])
	
	N_not = sum(notobject)
	N_is = sum(isobject)
	
	voxels_object_wrong_printed = sum(abs.((printed .> mid_thresh)[isobject] .- img[isobject]))
	voxels_void_wrong_printed = sum(abs.((printed .> mid_thresh)[notobject] .- img[notobject]))

	#voxels_object_wrong_printed / N_is, W_not_is / N_not, W_not / N_not, W_is / N_is

	@info "Object pixels not printed $(round(voxels_object_wrong_printed / N_is * 100, digits=4))%"
	@info "Void pixels falsely printed $(round(voxels_void_wrong_printed / N_not * 100, digits=4))%"
end

# ╔═╡ dceec7d4-706c-4a3b-8cf4-fa73f9800e3d


# ╔═╡ 5290976e-f60e-4a02-b58d-f91a870a5f89
function plot_histogram(img, object_printed, thresholds, chosen_threshold)
	plot(object_printed[img .== 0], bins=(0.0:0.01:1),  seriestype=:barhist, xlim=(0.0, 1.0), label="dose distribution void", ylabel="voxel count", xlabel="normalized intensity",  ylim=(10, 1000000),  yscale=:log10, linewidth=1, legend=:topleft)
	plot!(object_printed[img .== 1], seriestype=:barhist, bins=(0.0:0.01:1), xlim=(0.0, 1.0), label="dose distribution object", ylabel="voxel count", xlabel="normalized intensity",  ylim=(10, 1000000), linewidth=1,  yscale=:log10,)
	plot!([thresholds[1], thresholds[1]], [1, 10000_000], label="lower threshold", linewidth=3)
	plot!([thresholds[2], thresholds[2]], [1, 10000_000], label="upper threshold", linewidth=3)
	#plot!([chosen_threshold, chosen_threshold], [1, 30000000], label="chosen threshold", linewidth=3)
end

# ╔═╡ 7d694ff4-d0f6-4f10-a26b-d3c43930dd4f
md"# Optimize"

# ╔═╡ 578f360d-66b2-434e-bdf2-469673116711
angles = range(0, 2π, 805)

# ╔═╡ 448f7462-6a87-4ca3-a2a3-0f76934e97a1
N_iter = 500

# ╔═╡ aff2fc67-5569-40e6-9ed3-936f16de1aba
thresholds = (0.65, 0.75)

# ╔═╡ 96f6eba3-1700-4c29-aff6-5f774d7d4b3b
@time _, patterns_065_075, printed_065_075, losses_065_075 = Array.(optimize(togoc(img), togoc(angles); N_iter=20, thresholds=(0.65, 0.75)))

# ╔═╡ 198375e7-6857-4971-92c8-6aec7cb25090
size(printed_065_075)

# ╔═╡ 4b04002a-449c-46a6-921e-b3fe72395018
size(patterns_065_075)

# ╔═╡ e55b5e16-1ea1-4cfa-a0df-c32e03848c9f
errors(printed_065_075, isone.(img), iszero.(img), thresholds)

# ╔═╡ 1aa687fd-9647-4bce-a800-5cfe6730b28b
@bind thresh1 Slider(0.0:0.005:1, show_value=true)

# ╔═╡ 0ac01881-af91-4067-a216-6cef66d43bc0
md"Threshold =$thresh1"

# ╔═╡ f6b67921-cf3f-4144-bbdb-3c423d6ee81d
sum(printed_065_075 .> 0.5)

# ╔═╡ 0c7375cb-2cd1-4ebd-9227-5b561783e37e
simshow([printed_065_075 printed_065_075 .> thresh1 abs.((printed_065_075 .> thresh1) .- img)])

# ╔═╡ c81c9248-29b5-494a-95e1-276a1f8ecfa2
begin
	save("/tmp/printed_065_075.png", simshow(printed_065_075))
	save("/tmp/printed_065_075_thresh.png", simshow(printed_065_075 .> 0.7))
	save("/tmp/printed_065_075_thresh_diff.png", simshow(abs.((printed_065_075 .> 0.7) .- img)))
end

# ╔═╡ bf5a5074-ad69-4d17-b8de-892688a76891
begin
	p = plot_histogram(img[1:1:end], printed_065_075[1:1:end], thresholds, 0.7)
	savefig(p, "/tmp/histogram_printed_065_075.pdf")
	p
end

# ╔═╡ ec51fa20-725c-46a4-b5ab-eaefdbdda501


# ╔═╡ ef196630-d7fd-4481-b4e9-ce410dec8e73
@time _, patterns_05_08, printed_05_08, losses_05_08 = Array.(optimize(togoc(img), togoc(angles); N_iter=N_iter, thresholds=(0.5, 0.8)));

# ╔═╡ 5e1f0420-45d6-48b5-a249-8e5b7048bd65
errors(printed_05_08, isone.(img), iszero.(img), thresholds)

# ╔═╡ 118cbae4-04af-461a-9301-dafc9b1efbf7
begin
	save("/tmp/printed_05_08.png", simshow(printed_05_08))
	save("/tmp/printed_05_08_thresh_2.png", simshow(printed_05_08 .> 0.65))
	save("/tmp/printed_05_08_thresh_diff_2.png", simshow(abs.((printed_05_08 .> 0.65) .- img)))
end

# ╔═╡ 6dd162d4-f099-49f7-860b-dc1e0088090c
begin 
 	p2 = plot_histogram(img, printed_05_08, (0.5, 0.8), 0.65)
 	savefig(p2, "/tmp/histogram_printed_05_08.pdf")
	p2
end

# ╔═╡ 0ca23230-da00-43fc-8ad7-c93a0f56e123


# ╔═╡ f2bfcd8f-2e65-4fd4-903b-667171d685ff
@time _, patterns_065_075_3mu, printed_065_075_3mu, losses_065_075_3mu = Array.(optimize(togoc(img), togoc(angles), 3/ size(img, 1); N_iter=N_iter, thresholds=(0.65, 0.75)))

# ╔═╡ d77c1ac9-9e08-47ca-89db-3680e550eb4e
simshow(printed_065_075_3mu)

# ╔═╡ d53fcc24-5fee-4e4f-a144-f9fa2b0ee5cb
errors(printed_065_075_3mu, isone.(img), iszero.(img), thresholds)

# ╔═╡ 9351709f-7831-433b-839f-84c5b8b315c1
begin
	save("/tmp/printed_065_075_3mu.png", simshow(printed_065_075_3mu))
	save("/tmp/printed_065_075_3mu_thresh.png", simshow(printed_065_075_3mu .> 0.7))
	save("/tmp/printed_065_075_3mu_thresh_diff.png", simshow(abs.((printed_065_075_3mu .> 0.7) .- img)))
end

# ╔═╡ 5670277d-65b8-40f1-95a2-78358f082bed
begin
	p3 = plot_histogram(img, printed_065_075_3mu, thresholds, 0.7)
	savefig(p3, "/tmp/histogram_printed_065_075_3mu.pdf")
	p3
end

# ╔═╡ ac932771-c8e1-4ecf-ab09-008350a39b28


# ╔═╡ 19def696-ef29-466b-9ace-a7ed8f9d8c28
@time _, patterns_065_075_1mu, printed_065_075_1mu, losses_065_075_1mu = Array.(optimize(togoc(img), togoc(angles), 1/ size(img, 1); N_iter=N_iter, thresholds=(0.65, 0.75)))

# ╔═╡ 51950c3e-bfb0-443d-89d3-fe52e0b720ac
errors(printed_065_075_1mu, isone.(img), iszero.(img), thresholds)

# ╔═╡ f28558d3-bab7-4c0d-9a48-e470df8d6b2a
begin
	save("/tmp/printed_065_075_1mu.png", simshow(printed_065_075_1mu))
	save("/tmp/printed_065_075_1mu_thresh.png", simshow(printed_065_075_1mu .> 0.7))
	save("/tmp/printed_065_075_1mu_thresh_diff.png", simshow(abs.((printed_065_075_1mu .> 0.7) .- img)))
end

# ╔═╡ fae06fab-a707-487c-aeea-6f0c8e5ca3f6
begin
	p4 = plot_histogram(img, printed_065_075_1mu, thresholds, 0.7)
	savefig(p4, "/tmp/histogram_printed_065_075_1mu.pdf")
	p4
end

# ╔═╡ 48aafe12-1e21-4960-b0af-37468ad613f8


# ╔═╡ b1309600-9d64-47cb-aaed-86a1fa3a591e
# ╠═╡ disabled = true
#=╠═╡
@time _, patterns_065_075_1mu_long, printed_065_075_1mu_long, losses_065_075_1mu_long = Array.(optimize(togoc(img), togoc(angles), 1/ size(img, 1); N_iter=10_000, thresholds=(0.65, 0.75)))
  ╠═╡ =#

# ╔═╡ e6329988-a919-4569-9288-b8c4620a2d67
#=╠═╡
errors(printed_065_075_1mu_long, isone.(img), iszero.(img), thresholds)
  ╠═╡ =#

# ╔═╡ a7cc2f68-5592-498c-8891-328f54e35019
#=╠═╡
begin
	p5 = plot_histogram(img, printed_065_075_1mu_long, thresholds, 0.7)
	savefig(p5, "/tmp/histogram_printed_065_075_1mu_long.pdf")
	p5
end
  ╠═╡ =#

# ╔═╡ 3ef9a22c-ce55-4a90-9a5d-d5a5b835253e
#=╠═╡
save("/tmp/patterns_065_075_1mu_long.png",simshow(patterns_065_075_1mu_long, cmap=:turbo))
  ╠═╡ =#

# ╔═╡ 2a9f6404-50ca-473a-89f7-a55f163e5f61
save("/tmp/patterns_065_075_1mu.png",simshow(patterns_065_075_1mu, cmap=:turbo))

# ╔═╡ ba576d5e-3f1f-4f1b-aa1c-ef0f425ca14a


# ╔═╡ 5fc73726-c51d-41ef-b258-511b435d7ee8


# ╔═╡ adb5d5bb-a925-433b-ba66-c5b7ec4f4d73
#=╠═╡
sum(patterns_065_075_1mu_long ./ maximum(patterns_065_075_1mu_long)/ length(patterns_065_075_1mu_long))
  ╠═╡ =#

# ╔═╡ ef4f6fa4-ee46-47d1-9587-65125609acb2
#=╠═╡
sum(patterns_065_075_1mu  ./ maximum(patterns_065_075_1mu) / length(patterns_065_075_1mu_long))
  ╠═╡ =#

# ╔═╡ 708e26fa-89d1-4821-a9b9-6efce3b27b2b
#=╠═╡
sum(patterns_065_075_1mu / length(patterns_065_075_1mu_long))
  ╠═╡ =#

# ╔═╡ fa62003a-2748-45fe-9b37-7c32d215c49b
#=╠═╡
extrema(patterns_065_075_1mu_long)
  ╠═╡ =#

# ╔═╡ 04bc9aed-dd0d-46bb-a1f8-87a385806127
maximum(patterns_05_08) / mean(patterns_05_08)

# ╔═╡ 8dbbfc81-7df2-481b-b758-64ed23134f2d
#=╠═╡
maximum(patterns_065_075_1mu_long) / mean(patterns_065_075_1mu_long)
  ╠═╡ =#

# ╔═╡ c41d4159-9c31-4ece-98d2-a38007f1049b
#=╠═╡
simshow(patterns_065_075_1mu_long)
  ╠═╡ =#

# ╔═╡ e99e8aaa-f538-4647-82af-8e85397d43c8


# ╔═╡ 16384727-05a8-48bb-8755-8c47bd28b97c
# ╠═╡ disabled = true
#=╠═╡
begin
	p4 = plot(losses, yscale=:log10, label="\$T_L =0.65\$, \$T_U=0.75\$", ylabel="normalized loss", xlabel="iterations", yticks=[10^2, 10^3, 10^4, 10^5], grid=true)
	plot!(losses_2, label="\$T_L =0.5\$, \$T_U=0.8\$")
	#plot!(losses_3, label="\$T_L =0.75\$, \$T_U=0.8\$")
	savefig(p4, "/tmp/loss.pdf")
	p4
end
  ╠═╡ =#

# ╔═╡ b974c854-fcf7-4cbc-b3c8-1a8ab34f9190


# ╔═╡ fb10aac8-2068-4452-83e1-1624a8375d2c


# ╔═╡ 40cd392b-8020-4cbb-ac7c-ea32cddd4410


# ╔═╡ 621f86b9-0856-4aae-8ad7-4f1d557ebf81


# ╔═╡ ec67bddb-5fd1-4d26-8e76-41930d22f09c


# ╔═╡ bcab5404-1816-4e5a-b0d2-f46735826a65


# ╔═╡ f5d516a4-dd22-4971-977b-1526db3571c8
md"# Large optimization"

# ╔═╡ 51dbb38f-ad62-4664-ba95-2f943a33ce60
begin
	volume = zeros(Float32, (512, 512, 100))
	volume .+= box(size(volume)[1:2], (30, 100), offset=(230, 380))
	volume .-= rr2(size(volume)[1:2], offset=(170, 200)) .< 30^2
	volume .+= rr2(size(volume)[1:2], offset=(170, 200)) .< 100^2
end;

# ╔═╡ bd6bca52-a797-4a12-a24d-7989e9d941c8
simshow(volume[:, :, 1])

# ╔═╡ 600dcf9d-bcf3-42ea-bc9b-bd399483f028
angles2 = range(0, π, 200)

# ╔═╡ 2fd28802-e7dd-4eec-903a-3b94b87cbc16
# ╠═╡ disabled = true
#=╠═╡
CUDA.@sync CUDA.@time virtual_object_c2, patterns_optimized_c2 = optimize(CuArray(volume), CuArray(angles2); N_iter=50, thresholds, backend=CUDABackend())
  ╠═╡ =#

# ╔═╡ da12770d-4e4e-457b-bbaa-affad72eac33
object_printed_c2 = Array(iradon(Array(patterns_optimized_c2), angles2));

# ╔═╡ 9bd41858-b191-4935-b9cc-24fdf6be8cbe
simshow(object_printed_c2 .> 0.75)[:, :, 1]

# ╔═╡ Cell order:
# ╠═453f6cca-91ea-11ee-0c48-b923a785e914
# ╠═d956ede0-5a1f-4ed6-a819-1730f52a3536
# ╠═9fe3b45a-3901-40e4-bad9-32ee3735d971
# ╠═57d629e6-4de8-4131-bd97-a505c31ab475
# ╠═310da375-4964-4165-aa6b-c50c151ea65e
# ╠═20f68941-4141-4d81-bae8-5ee9661dd80a
# ╠═a264aa76-2057-42d4-8dd6-254c3d0cdf44
# ╠═4678b348-f61e-4720-af21-7313c79365f7
# ╠═fbd565e1-ea6c-441a-8ed4-39e65071b798
# ╟─21f890e4-2bb7-4b06-bd79-5e8bf5cce6f5
# ╠═358dda4a-d589-4993-9d6f-d42bcddda1a3
# ╠═b6ed1974-d562-47d4-a342-66eb474bc556
# ╠═79c1b042-2b9d-43db-9525-d66418db5da2
# ╟─aa1cccec-f846-4b1e-9aae-81e939cd3b8c
# ╠═0c1abf70-8004-460a-b008-750050f6e718
# ╠═5ac76842-f5a3-42c0-9764-9b20f98ab1d5
# ╠═e19a1fb6-d237-44c8-a1b1-45e38aebf948
# ╠═dceec7d4-706c-4a3b-8cf4-fa73f9800e3d
# ╠═5290976e-f60e-4a02-b58d-f91a870a5f89
# ╟─7d694ff4-d0f6-4f10-a26b-d3c43930dd4f
# ╠═578f360d-66b2-434e-bdf2-469673116711
# ╠═448f7462-6a87-4ca3-a2a3-0f76934e97a1
# ╠═aff2fc67-5569-40e6-9ed3-936f16de1aba
# ╠═96f6eba3-1700-4c29-aff6-5f774d7d4b3b
# ╠═198375e7-6857-4971-92c8-6aec7cb25090
# ╠═4b04002a-449c-46a6-921e-b3fe72395018
# ╠═e55b5e16-1ea1-4cfa-a0df-c32e03848c9f
# ╟─0ac01881-af91-4067-a216-6cef66d43bc0
# ╠═1aa687fd-9647-4bce-a800-5cfe6730b28b
# ╠═f6b67921-cf3f-4144-bbdb-3c423d6ee81d
# ╠═0c7375cb-2cd1-4ebd-9227-5b561783e37e
# ╠═c81c9248-29b5-494a-95e1-276a1f8ecfa2
# ╠═bf5a5074-ad69-4d17-b8de-892688a76891
# ╠═ec51fa20-725c-46a4-b5ab-eaefdbdda501
# ╠═ef196630-d7fd-4481-b4e9-ce410dec8e73
# ╠═5e1f0420-45d6-48b5-a249-8e5b7048bd65
# ╠═118cbae4-04af-461a-9301-dafc9b1efbf7
# ╠═6dd162d4-f099-49f7-860b-dc1e0088090c
# ╠═0ca23230-da00-43fc-8ad7-c93a0f56e123
# ╠═f2bfcd8f-2e65-4fd4-903b-667171d685ff
# ╠═d77c1ac9-9e08-47ca-89db-3680e550eb4e
# ╠═d53fcc24-5fee-4e4f-a144-f9fa2b0ee5cb
# ╠═9351709f-7831-433b-839f-84c5b8b315c1
# ╠═5670277d-65b8-40f1-95a2-78358f082bed
# ╠═ac932771-c8e1-4ecf-ab09-008350a39b28
# ╠═19def696-ef29-466b-9ace-a7ed8f9d8c28
# ╠═51950c3e-bfb0-443d-89d3-fe52e0b720ac
# ╠═f28558d3-bab7-4c0d-9a48-e470df8d6b2a
# ╠═fae06fab-a707-487c-aeea-6f0c8e5ca3f6
# ╠═48aafe12-1e21-4960-b0af-37468ad613f8
# ╠═b1309600-9d64-47cb-aaed-86a1fa3a591e
# ╠═e6329988-a919-4569-9288-b8c4620a2d67
# ╠═a7cc2f68-5592-498c-8891-328f54e35019
# ╠═3ef9a22c-ce55-4a90-9a5d-d5a5b835253e
# ╠═2a9f6404-50ca-473a-89f7-a55f163e5f61
# ╠═ba576d5e-3f1f-4f1b-aa1c-ef0f425ca14a
# ╠═5fc73726-c51d-41ef-b258-511b435d7ee8
# ╠═adb5d5bb-a925-433b-ba66-c5b7ec4f4d73
# ╠═ef4f6fa4-ee46-47d1-9587-65125609acb2
# ╠═708e26fa-89d1-4821-a9b9-6efce3b27b2b
# ╠═fa62003a-2748-45fe-9b37-7c32d215c49b
# ╠═04bc9aed-dd0d-46bb-a1f8-87a385806127
# ╠═8dbbfc81-7df2-481b-b758-64ed23134f2d
# ╠═c41d4159-9c31-4ece-98d2-a38007f1049b
# ╠═e99e8aaa-f538-4647-82af-8e85397d43c8
# ╠═16384727-05a8-48bb-8755-8c47bd28b97c
# ╠═b974c854-fcf7-4cbc-b3c8-1a8ab34f9190
# ╠═fb10aac8-2068-4452-83e1-1624a8375d2c
# ╠═40cd392b-8020-4cbb-ac7c-ea32cddd4410
# ╠═621f86b9-0856-4aae-8ad7-4f1d557ebf81
# ╠═ec67bddb-5fd1-4d26-8e76-41930d22f09c
# ╠═bcab5404-1816-4e5a-b0d2-f46735826a65
# ╟─f5d516a4-dd22-4971-977b-1526db3571c8
# ╠═51dbb38f-ad62-4664-ba95-2f943a33ce60
# ╠═bd6bca52-a797-4a12-a24d-7989e9d941c8
# ╠═600dcf9d-bcf3-42ea-bc9b-bd399483f028
# ╠═2fd28802-e7dd-4eec-903a-3b94b87cbc16
# ╠═da12770d-4e4e-457b-bbaa-affad72eac33
# ╠═9bd41858-b191-4935-b9cc-24fdf6be8cbe
