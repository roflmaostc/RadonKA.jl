### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 5cc65690-91d8-11ee-07f9-557c9a76e478
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ c4b5d8f3-7ccd-4888-8b83-4642c1e81916
using TestImages, RadonKA, ImageShow, ImageIO, Noise, PlutoUI

# ╔═╡ 4458d362-1aae-4a24-9bf7-2bed6286579a
using RegularizedLeastSquares, LinearMaps, IterativeSolvers, LinearOperators

# ╔═╡ 5b4e5e74-8dd1-45d4-90a7-9b1211c17696
using Random

# ╔═╡ b881ba80-7b02-4f2a-9fad-b61c4ee218c1
using CUDA, CUDA.CUDAKernels

# ╔═╡ f95dfc57-060f-4173-aedc-25e36b67fcb4
md"# Load Testimage"

# ╔═╡ 162a9e47-985c-406d-b8db-914a8ee17047
N = 256

# ╔═╡ 2e58525e-3c9b-4bd4-a903-149a27939af0
sample = Float32.(TestImages.shepp_logan(N));

# ╔═╡ 8503c029-c37f-44eb-8e66-3acc01dd678a
simshow(sample)

# ╔═╡ b31c0d18-4338-450f-bf58-c825ec813519
md"# Make Sinogram"

# ╔═╡ e90ac7f4-504b-47ea-b087-9a25afe8ec22
K = 300

# ╔═╡ 29ca822c-fe80-463c-8d6e-efcb178d8d67
angles = Float32.(range(0, π, K))

# ╔═╡ 0378cea6-8a61-4e16-a99a-e2160c83518b
@time sinogram = poisson(radon(sample, angles), 1000);

# ╔═╡ c93ecf01-1a06-4c3e-9c4d-ae1516f38749
size(sinogram)

# ╔═╡ f2573a91-c0a6-421d-8cb9-8745b5c5d276
simshow(sinogram)

# ╔═╡ c40638f2-9bcc-4954-b80a-126c6afbd344
md"# Backproject with iradon"

# ╔═╡ fa6f64e8-557f-4c4b-81ba-c1bae5b98f36
@time sample_iradon = iradon(sinogram, angles);

# ╔═╡ c9ffc233-0e0b-452d-8f53-09b8654c8bca
simshow(sample_iradon)

# ╔═╡ 7b7579a8-8799-4ba9-8931-ee4642a36124
md"# Filtered Backprojection"

# ╔═╡ ca4fdba3-2504-4f26-860e-e95c6aae8f12
sample_filtered = filtered_backprojection(sinogram, angles);

# ╔═╡ eff77ed3-a9ea-4459-b64e-c4459b31619b
simshow(sample_filtered)

# ╔═╡ a303b94f-8807-4fe5-974a-7914db10f8eb
md"# Iterative Reconstruction
For that we define a linear `radon_map` which defines the efficient forward operation and the adjoint. We use [LinearMaps](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) for that.

For the Regularized Least Squares we use [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl).

"

# ╔═╡ 90c2e9af-4528-4e30-afe5-bd570592bf27
Base.retry_load_extensions()

# ╔═╡ e49ade0d-0c14-49b4-b156-5a83ffa93fd3
struct RadonMap{F, A}
	forward::F
	adjoint::A
end

# ╔═╡ 90bc785e-9ae2-47f2-8d85-32c606a39954
struct RadonAdjoint{A}
	adjoint::A
end

# ╔═╡ 48ac5f37-918d-4aa4-a049-61e3779e8e13
Base.:*(rm::RadonMap, x) = rm.forward(x)

# ╔═╡ 53af9808-be5e-4386-9a82-460dfdf147b8
Base.:*(rm::RadonAdjoint, x) = rm.adjoint(x)

# ╔═╡ 843c8c0d-99fb-4198-a0cb-5f7ab829b792
Base.adjoint(rm::RadonMap) = RadonAdjoint(rm.adjoint)

# ╔═╡ e9e71b13-487c-4706-b8b8-831a60474ffc
rm = RadonMap(x -> vec(radon(reshape(x, (N, N)), angles)), x -> vec(iradon(reshape(x, (N-1, K)), angles)))

# ╔═╡ 4df0a17e-79fa-4fd5-8b8d-1ad0369b55eb
radon_map = LinearMap(
	x -> vec(radon(reshape(x, (N, N)), angles)), 
	x -> vec(iradon(reshape(x, (N-1, K)), angles)),
	(N - 1) * K, N*N)

# ╔═╡ 104b78de-022b-4419-867a-1d99a2663f8c
function radon_f!(res, v, α, β)
  if β == 0
    res .= α .* vec(radon(reshape(v, (N, N)), angles))
  else
    res .= α .*vec(radon(reshape(v, (N, N)), angles)) .+ β .* res
  end
end

# ╔═╡ 3873e3c9-b9cf-48d7-9ef5-b1372778c138
function iradon_f!(res, w, α, β)
  if β == 0
    res .= α .* vec(iradon(reshape(w, (N-1, K)), angles))
  else
    res .= α .* vec(iradon(reshape(w, (N-1, K)), angles)) .+ β .* res
  end
end

# ╔═╡ d3b27172-3373-4176-a8bd-475acd825e8d
dft = LinearOperator(Float32, (N - 1) * K, N*N, false, false,
                     radon_f!,
                     nothing,       # will be inferred
                     iradon_f!)

# ╔═╡ 90ef6756-60ac-4b5d-8620-73566a465d22
simshow(reshape(radon_map * vec(sample), (N - 1, K)))

# ╔═╡ e2207cd2-9471-460f-9eb4-fcfed59cbe7e
adjoint(radon_map) * vec(sinogram)

# ╔═╡ ad5b1b23-2187-411c-89c2-ea4a9c8bd2b0
guess = vec(copy(sample_iradon));

# ╔═╡ 1066c9df-026f-4877-8dbd-901dbb283ce3
dft * sample[:]

# ╔═╡ 10435932-ee34-42d1-88b1-562038c2aab4
dft * sample[:]

# ╔═╡ 79942dd9-17f6-4072-83ca-6cc2fa83c28f
size(sample[:])

# ╔═╡ 5ab727b8-8731-49d4-ab40-01d410e0b7ae
vec(sample) |> size

# ╔═╡ 48bbb97c-d12d-4282-92f9-02525d21f3ee
reg = TVRegularization(0.01; shape=(N,N))

# ╔═╡ ea6d5aa7-09de-4c1e-9a46-00e7d5267433
solver = createLinearSolver(ADMM, dft; reg=reg, ρ=0.1, iterations=20)

# ╔═╡ 8917be57-4a1b-4f1a-bcf2-c29b32e0c726
Ireco = solve(solver, vec(sinogram))

# ╔═╡ 924f005c-3b18-4ac7-9723-fda33068a90d


# ╔═╡ fb1fe392-c271-450f-bf35-7c7b6865e5f2
cg!(guess, radon_map, vec(sinogram))

# ╔═╡ 10e0d74e-9287-4997-85ae-bdbf99c83c6d
vec(guess) |> size

# ╔═╡ 01b6465b-eabf-48b2-95f9-a689784a2516
md"# Accelerate with CUDA"

# ╔═╡ 0166e1a8-03b5-480f-951f-4c1b73ebb954


# ╔═╡ 21532638-2948-4b80-add4-4f6489d19e5f


# ╔═╡ Cell order:
# ╠═5cc65690-91d8-11ee-07f9-557c9a76e478
# ╠═c4b5d8f3-7ccd-4888-8b83-4642c1e81916
# ╟─f95dfc57-060f-4173-aedc-25e36b67fcb4
# ╠═162a9e47-985c-406d-b8db-914a8ee17047
# ╠═2e58525e-3c9b-4bd4-a903-149a27939af0
# ╠═8503c029-c37f-44eb-8e66-3acc01dd678a
# ╟─b31c0d18-4338-450f-bf58-c825ec813519
# ╠═e90ac7f4-504b-47ea-b087-9a25afe8ec22
# ╠═29ca822c-fe80-463c-8d6e-efcb178d8d67
# ╠═0378cea6-8a61-4e16-a99a-e2160c83518b
# ╠═c93ecf01-1a06-4c3e-9c4d-ae1516f38749
# ╠═f2573a91-c0a6-421d-8cb9-8745b5c5d276
# ╟─c40638f2-9bcc-4954-b80a-126c6afbd344
# ╠═fa6f64e8-557f-4c4b-81ba-c1bae5b98f36
# ╠═c9ffc233-0e0b-452d-8f53-09b8654c8bca
# ╟─7b7579a8-8799-4ba9-8931-ee4642a36124
# ╠═ca4fdba3-2504-4f26-860e-e95c6aae8f12
# ╠═eff77ed3-a9ea-4459-b64e-c4459b31619b
# ╟─a303b94f-8807-4fe5-974a-7914db10f8eb
# ╠═4458d362-1aae-4a24-9bf7-2bed6286579a
# ╠═90c2e9af-4528-4e30-afe5-bd570592bf27
# ╠═e49ade0d-0c14-49b4-b156-5a83ffa93fd3
# ╠═90bc785e-9ae2-47f2-8d85-32c606a39954
# ╠═48ac5f37-918d-4aa4-a049-61e3779e8e13
# ╠═53af9808-be5e-4386-9a82-460dfdf147b8
# ╠═843c8c0d-99fb-4198-a0cb-5f7ab829b792
# ╠═e9e71b13-487c-4706-b8b8-831a60474ffc
# ╠═4df0a17e-79fa-4fd5-8b8d-1ad0369b55eb
# ╠═104b78de-022b-4419-867a-1d99a2663f8c
# ╠═3873e3c9-b9cf-48d7-9ef5-b1372778c138
# ╠═d3b27172-3373-4176-a8bd-475acd825e8d
# ╠═90ef6756-60ac-4b5d-8620-73566a465d22
# ╠═e2207cd2-9471-460f-9eb4-fcfed59cbe7e
# ╠═ad5b1b23-2187-411c-89c2-ea4a9c8bd2b0
# ╠═1066c9df-026f-4877-8dbd-901dbb283ce3
# ╠═10435932-ee34-42d1-88b1-562038c2aab4
# ╠═79942dd9-17f6-4072-83ca-6cc2fa83c28f
# ╠═5ab727b8-8731-49d4-ab40-01d410e0b7ae
# ╠═ea6d5aa7-09de-4c1e-9a46-00e7d5267433
# ╠═48bbb97c-d12d-4282-92f9-02525d21f3ee
# ╠═5b4e5e74-8dd1-45d4-90a7-9b1211c17696
# ╠═8917be57-4a1b-4f1a-bcf2-c29b32e0c726
# ╠═924f005c-3b18-4ac7-9723-fda33068a90d
# ╠═fb1fe392-c271-450f-bf35-7c7b6865e5f2
# ╠═10e0d74e-9287-4997-85ae-bdbf99c83c6d
# ╟─01b6465b-eabf-48b2-95f9-a689784a2516
# ╠═0166e1a8-03b5-480f-951f-4c1b73ebb954
# ╠═21532638-2948-4b80-add4-4f6489d19e5f
# ╠═b881ba80-7b02-4f2a-9fad-b61c4ee218c1
