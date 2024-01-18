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
using TestImages, RadonKA, ImageShow, ImageIO, Noise, PlutoUI, BenchmarkTools, CUDA, Zygote, IndexFunArrays

# ╔═╡ f5e6e461-dbd6-4898-a88d-4d3b354812ab
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# ╔═╡ 42f31ceb-0f5d-46cc-b868-0314b41a5407
begin
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ 035c5114-7081-490e-b8eb-8e7e1811f344
img = togoc(Float32.(testimage("resolution_test_512"))[200:421, 200:421]  .* (rr2(Float32, (222, 222)) .< 100^2));

# ╔═╡ a6bdc90f-3c10-4524-9892-6e976d556a95
simshow(Array(img))

# ╔═╡ 5ef75bb5-5ab0-4e98-9026-a8e2c5b7833d
angles = togoc(range(0, 2f0 * π, 700));

# ╔═╡ 80917ca2-1d18-405c-9206-5b76ecb71ddd
222 ÷ 2

# ╔═╡ b901a1c4-d346-4cfe-a6db-cae99ae792ca
ray_endpoints2 = togoc(range(-70, 70, size(img, 1)-1));

# ╔═╡ 372d0743-2825-4a67-b2ba-588bc6b7d0d5
measurement = togoc(poisson(Array(radon(img, angles; ray_endpoints=ray_endpoints2)), 2_000_000));

# ╔═╡ f05c24f6-9506-4403-a458-9e201e344757
simshow(Array(measurement))

# ╔═╡ e6294e5d-f3d5-4de6-b38c-6bc24efec840
simshow(Array(iradon(measurement, angles; ray_endpoints2)))

# ╔═╡ c9947a12-8b74-43c4-884e-ead8898d390a
simshow(Array(iradon(measurement[:, 1:1], angles[1:1]; ray_endpoints=ray_endpoints2)), γ=0.2)

# ╔═╡ 4eb50ae4-b5c2-4252-8d80-b758f838a5fb
opt_f(x, p) = sum(abs2, sqrt.(max.(0, radon(x, angles; ray_endpoints=ray_endpoints2)) .+ 3f0/8f0) .- sqrt.(3f0 / 8f0 .+ measurement))

# ╔═╡ 638561f3-6e6d-4faf-9daa-98694173e0a2
opt_fun = OptimizationFunction(opt_f, AutoZygote())

# ╔═╡ 4c8c7b5c-7112-4209-8b0b-a2463e2118a9
init0 = togoc(zeros(Float32, size(img)));

# ╔═╡ 1432bcd6-87b2-4881-867e-b190f8beb140
opt_f(init0, angles)

# ╔═╡ 1659b97f-d944-418a-adfc-0091a15efc84
problem = OptimizationProblem(opt_fun, init0, angles);

# ╔═╡ 90a8908a-5f81-40b1-acde-4fefd210709d
# ╠═╡ disabled = true
#=╠═╡
@mytime res = solve(problem, OptimizationOptimJL.LBFGS(), maxiters=20);
  ╠═╡ =#

# ╔═╡ 925688d4-78b4-404e-b4fd-0be2eaa31235
#=╠═╡
simshow(Array(res.u))
  ╠═╡ =#

# ╔═╡ c87031bb-e409-4283-a7e5-724935bf24c3
begin
	sinogram = zeros((49, 1))
	
	sinogram[2:2:end] .= 1
end

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

# ╔═╡ 1bc506b5-2ce4-4805-a159-30ba0f0c2236
6.358

# ╔═╡ 95af2e04-56f5-4df3-870b-630d22c130e6
 distort_rays_vial(7.5, 10, 8, 
	1.3, 1.3)

# ╔═╡ 6721b65f-48ed-4dc7-99dd-c460bb124c1e
ray_startpoints, ray_endpoints = distort_rays_vial(range(-8,8,49), 8, 7, 
	1.5, 1.5)

# ╔═╡ ca49418d-2f42-4943-b7e6-243947faf61d
ray_endpoints

# ╔═╡ 8e3f06a0-fd97-4474-a304-2ff046a9b76b
simshow(iradon(sinogram, [0.0]; 
		ray_startpoints,
		ray_endpoints), γ=0.1)[1:25, :]

# ╔═╡ cf7b835e-1d07-4efc-88b6-298d73ce0ffe
@time iradon(sinogram, [0.0]; 
		ray_startpoints=range(-12,12,49),
		ray_endpoints=range(-12, 12, 49))

# ╔═╡ d7afaf9d-146c-4129-92ba-f7c1f2c353e2
@time iradon(sinogram, [0.0]);

# ╔═╡ 807e4c4d-68ce-49e9-bbd2-93d7ee9fd170
Revise.errors()

# ╔═╡ efa0cf8a-e904-485b-a42c-1604dea66364
simshow(iradon(sinogram, [0.0]))

# ╔═╡ b53c8016-177f-4b1b-b07d-48cbd2b2fc65


# ╔═╡ Cell order:
# ╠═179d107e-b3be-11ee-0a6c-49f1bf2a10fd
# ╠═e3a3dc83-6536-445f-b56b-d8ad62cd9a85
# ╠═42f31ceb-0f5d-46cc-b868-0314b41a5407
# ╠═035c5114-7081-490e-b8eb-8e7e1811f344
# ╠═a6bdc90f-3c10-4524-9892-6e976d556a95
# ╠═5ef75bb5-5ab0-4e98-9026-a8e2c5b7833d
# ╠═80917ca2-1d18-405c-9206-5b76ecb71ddd
# ╠═b901a1c4-d346-4cfe-a6db-cae99ae792ca
# ╠═372d0743-2825-4a67-b2ba-588bc6b7d0d5
# ╠═f05c24f6-9506-4403-a458-9e201e344757
# ╠═f5e6e461-dbd6-4898-a88d-4d3b354812ab
# ╠═e6294e5d-f3d5-4de6-b38c-6bc24efec840
# ╠═c9947a12-8b74-43c4-884e-ead8898d390a
# ╠═4eb50ae4-b5c2-4252-8d80-b758f838a5fb
# ╠═638561f3-6e6d-4faf-9daa-98694173e0a2
# ╠═4c8c7b5c-7112-4209-8b0b-a2463e2118a9
# ╠═1432bcd6-87b2-4881-867e-b190f8beb140
# ╠═1659b97f-d944-418a-adfc-0091a15efc84
# ╠═90a8908a-5f81-40b1-acde-4fefd210709d
# ╠═925688d4-78b4-404e-b4fd-0be2eaa31235
# ╠═c87031bb-e409-4283-a7e5-724935bf24c3
# ╠═af83cb81-49b8-41f4-a24e-c1e33e76d925
# ╠═1bc506b5-2ce4-4805-a159-30ba0f0c2236
# ╠═95af2e04-56f5-4df3-870b-630d22c130e6
# ╠═6721b65f-48ed-4dc7-99dd-c460bb124c1e
# ╠═ca49418d-2f42-4943-b7e6-243947faf61d
# ╠═8e3f06a0-fd97-4474-a304-2ff046a9b76b
# ╠═cf7b835e-1d07-4efc-88b6-298d73ce0ffe
# ╠═d7afaf9d-146c-4129-92ba-f7c1f2c353e2
# ╠═807e4c4d-68ce-49e9-bbd2-93d7ee9fd170
# ╠═efa0cf8a-e904-485b-a42c-1604dea66364
# ╠═b53c8016-177f-4b1b-b07d-48cbd2b2fc65
