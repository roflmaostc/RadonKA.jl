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

# ╔═╡ 453f6cca-91ea-11ee-0c48-b923a785e914
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ d956ede0-5a1f-4ed6-a819-1730f52a3536
using TestImages, RadonKA, ImageShow, ImageIO, Noise, PlutoUI, IndexFunArrays, FFTW

# ╔═╡ 9fe3b45a-3901-40e4-bad9-32ee3735d971
using Plots

# ╔═╡ 29703cc9-d5ba-4ff3-bc0c-2ad325c7aa7d
using CUDA, CUDA.CUDAKernels, KernelAbstractions

# ╔═╡ 21f890e4-2bb7-4b06-bd79-5e8bf5cce6f5
md"# A sample we want to print"

# ╔═╡ 358dda4a-d589-4993-9d6f-d42bcddda1a3
begin
	img = zeros(Float32, (200,200))
	
	#img .+= rr2(img, offset=(70, 100)) .< 30^2
	#img .-= rr2(img, offset=(70, 100)) .< 20^2
	img .+= box(img, (10, 70), offset=(130, 80))
	#img .+= box(img, (1, 1), offset=(130, 100))
	#img .+= box(img, (1, 1), offset=(110, 100))
	img .+= rr2(img, offset=(70, 100)) .< 30^2
end;

# ╔═╡ 57da750b-ec5c-4636-bac3-549e7fadc3c8
simshow(img)

# ╔═╡ aa1cccec-f846-4b1e-9aae-81e939cd3b8c
md"""# Algorithm

See this paper: Rackson, Charles M., et al. "Object-space optimization of tomographic reconstructions for additive manufacturing." Additive Manufacturing 48 (2021): 102367.
"""

# ╔═╡ 0c1abf70-8004-460a-b008-750050f6e718
function iter!(buffer, img, θs; filtered=true, clip_sinogram=true, backend=CPU())
	sinogram = radon(img, θs; backend)
	if filtered
		sinogram = ramp_filter(sinogram)
	end
	
	if clip_sinogram
		sinogram .= max.(sinogram, 0)
	end
	
	img_recon = iradon(sinogram, θs; backend)

	buffer .= max.(img_recon, 0)
	return buffer, sinogram
end

# ╔═╡ 5ac76842-f5a3-42c0-9764-9b20f98ab1d5
function optimize(img, θs; thresholds=(0.65, 0.75), N_iter = 2, backend=CPU())
	N = size(img, 1)
	fx = (-N / 2):1:(N /2 -1)
	R2D = similar(img)
	R2D .= sqrt.(fx'.^2 .+ fx.^2)

	p = plan_fft(img, (1,2))
	guess = max.(0, real.(inv(p) * ((p * img) .* ifftshift(R2D, (1,2)))))
	guess ./= maximum(guess)

	guess = copy(img)
	notobject = iszero.(img)
	isobject = isone.(img)
	
	buffer = copy(img)
	tmp, s = iter!(buffer, guess, θs; filtered=false, clip_sinogram=true, backend)
	for i in 1:N_iter
		guess[notobject] .-= max.(0, tmp[notobject] .- thresholds[1])

		tmp, s = iter!(buffer, guess, θs; backend, filtered=false, clip_sinogram=true)

		guess[isobject] .+= max.(0, thresholds[2] .- tmp[isobject])
	end

	return guess, s#radon(guess, θs, det)
end 

# ╔═╡ 7d694ff4-d0f6-4f10-a26b-d3c43930dd4f
md"# Optimize"

# ╔═╡ 578f360d-66b2-434e-bdf2-469673116711
angles = range(0, 2π, 350)

# ╔═╡ aff2fc67-5569-40e6-9ed3-936f16de1aba
thresholds = (0.7, 0.8)

# ╔═╡ 96f6eba3-1700-4c29-aff6-5f774d7d4b3b
@time virtual_object, patterns_optimized = optimize(img, angles; N_iter=50, thresholds)

# ╔═╡ 5568ca5b-2aad-4024-8cf0-d68260a4ac36
simshow(patterns_optimized)

# ╔═╡ 26613767-5e05-4349-8bc2-2cfe357749c8
object_printed = iradon(patterns_optimized, angles);

# ╔═╡ 2d602a28-c28b-4267-966b-81a7de2e2fbc
simshow(virtual_object)

# ╔═╡ 0ac01881-af91-4067-a216-6cef66d43bc0
md"Threshold =$(@bind thresh2 Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ 0c7375cb-2cd1-4ebd-9227-5b561783e37e
simshow([object_printed object_printed .> thresh2 abs.((object_printed .> thresh2) .- img)])

# ╔═╡ b5ae35db-bf7d-4c00-a069-8572af9bc1de
begin
		histogram(object_printed[:], bins=(0.0:0.01:1), xlim=(0.0, 1.0), label="dose distribution", ylabel="voxel count", xlabel="normalized intensity", ylim=(0, 1000))
		plot!([thresholds[1], thresholds[1]], [0, 100_000], label="Lower threshold")
		plot!([thresholds[2], thresholds[2]], [0, 100_000], label="Upper threshold")
		plot!([thresh2, thresh2], [0, 3000], label="Real threshold")
end

# ╔═╡ b277afc7-bd21-4031-866c-3fe0f0747adb
md"# Speed Up with CUDA"

# ╔═╡ 4dd89f19-68b2-4e01-9ad9-de8fcbcd00ba
radon(CuArray(img), CuArray(angles), backend=CUDABackend());

# ╔═╡ e08da75f-d3a9-4aab-acb0-e4a1d5371fae
CUDA.@sync CUDA.@time virtual_object_c, patterns_optimized_c = optimize(CuArray(img), CuArray(angles); N_iter=50, thresholds, backend=CUDABackend())

# ╔═╡ 2515881d-be85-4e5a-87fb-881aaa860464
object_printed_c = Array(iradon(Array(patterns_optimized_c), angles));

# ╔═╡ 178846ee-5678-4332-9cb9-f900090f06fa
Array(object_printed_c) ≈ object_printed

# ╔═╡ 8be0b6ee-5a54-4d7f-a7bf-e71b0a637439
simshow([object_printed object_printed_c .> thresh2 abs.((object_printed .> thresh2) .- img)])

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
CUDA.@sync CUDA.@time virtual_object_c2, patterns_optimized_c2 = optimize(CuArray(volume), CuArray(angles2); N_iter=50, thresholds, backend=CUDABackend())

# ╔═╡ da12770d-4e4e-457b-bbaa-affad72eac33
object_printed_c2 = Array(iradon(Array(patterns_optimized_c2), angles2));

# ╔═╡ 9bd41858-b191-4935-b9cc-24fdf6be8cbe
simshow(object_printed_c2 .> 0.74)[:, :, 1]

# ╔═╡ Cell order:
# ╠═453f6cca-91ea-11ee-0c48-b923a785e914
# ╠═d956ede0-5a1f-4ed6-a819-1730f52a3536
# ╠═9fe3b45a-3901-40e4-bad9-32ee3735d971
# ╟─21f890e4-2bb7-4b06-bd79-5e8bf5cce6f5
# ╠═358dda4a-d589-4993-9d6f-d42bcddda1a3
# ╠═57da750b-ec5c-4636-bac3-549e7fadc3c8
# ╟─aa1cccec-f846-4b1e-9aae-81e939cd3b8c
# ╠═0c1abf70-8004-460a-b008-750050f6e718
# ╠═5ac76842-f5a3-42c0-9764-9b20f98ab1d5
# ╟─7d694ff4-d0f6-4f10-a26b-d3c43930dd4f
# ╠═578f360d-66b2-434e-bdf2-469673116711
# ╠═aff2fc67-5569-40e6-9ed3-936f16de1aba
# ╠═96f6eba3-1700-4c29-aff6-5f774d7d4b3b
# ╠═5568ca5b-2aad-4024-8cf0-d68260a4ac36
# ╠═26613767-5e05-4349-8bc2-2cfe357749c8
# ╠═2d602a28-c28b-4267-966b-81a7de2e2fbc
# ╟─0ac01881-af91-4067-a216-6cef66d43bc0
# ╠═0c7375cb-2cd1-4ebd-9227-5b561783e37e
# ╠═b5ae35db-bf7d-4c00-a069-8572af9bc1de
# ╟─b277afc7-bd21-4031-866c-3fe0f0747adb
# ╠═29703cc9-d5ba-4ff3-bc0c-2ad325c7aa7d
# ╠═4dd89f19-68b2-4e01-9ad9-de8fcbcd00ba
# ╠═e08da75f-d3a9-4aab-acb0-e4a1d5371fae
# ╠═2515881d-be85-4e5a-87fb-881aaa860464
# ╠═178846ee-5678-4332-9cb9-f900090f06fa
# ╠═8be0b6ee-5a54-4d7f-a7bf-e71b0a637439
# ╟─f5d516a4-dd22-4971-977b-1526db3571c8
# ╠═51dbb38f-ad62-4664-ba95-2f943a33ce60
# ╠═bd6bca52-a797-4a12-a24d-7989e9d941c8
# ╠═600dcf9d-bcf3-42ea-bc9b-bd399483f028
# ╠═2fd28802-e7dd-4eec-903a-3b94b87cbc16
# ╠═da12770d-4e4e-457b-bbaa-affad72eac33
# ╠═9bd41858-b191-4935-b9cc-24fdf6be8cbe
