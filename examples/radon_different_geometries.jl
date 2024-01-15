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
angles = togoc(range(0, 2f0 * π, 500));

# ╔═╡ b901a1c4-d346-4cfe-a6db-cae99ae792ca
ray_endpoints = togoc(range(-30, 30, size(img, 1)-1));

# ╔═╡ 372d0743-2825-4a67-b2ba-588bc6b7d0d5
measurement = togoc(poisson(Array(radon(img, angles; ray_endpoints)), 2_000_000));

# ╔═╡ f05c24f6-9506-4403-a458-9e201e344757
simshow(Array(measurement))

# ╔═╡ 4eb50ae4-b5c2-4252-8d80-b758f838a5fb
opt_f(x, p) = sum(abs2, sqrt.(max.(0, radon(x, angles; ray_endpoints)) .+ 3f0/8f0) .- sqrt.(3f0 / 8f0 .+ measurement))

# ╔═╡ 638561f3-6e6d-4faf-9daa-98694173e0a2
opt_fun = OptimizationFunction(opt_f, AutoZygote())

# ╔═╡ 4c8c7b5c-7112-4209-8b0b-a2463e2118a9
init0 = togoc(zeros(Float32, size(img)));

# ╔═╡ 1432bcd6-87b2-4881-867e-b190f8beb140
opt_f(init0, angles)

# ╔═╡ 1659b97f-d944-418a-adfc-0091a15efc84
problem = OptimizationProblem(opt_fun, init0, angles);

# ╔═╡ 90a8908a-5f81-40b1-acde-4fefd210709d
@mytime res = solve(problem, OptimizationOptimJL.LBFGS(), maxiters=20);

# ╔═╡ 925688d4-78b4-404e-b4fd-0be2eaa31235
simshow(Array(res.u))

# ╔═╡ Cell order:
# ╠═179d107e-b3be-11ee-0a6c-49f1bf2a10fd
# ╠═e3a3dc83-6536-445f-b56b-d8ad62cd9a85
# ╠═42f31ceb-0f5d-46cc-b868-0314b41a5407
# ╠═035c5114-7081-490e-b8eb-8e7e1811f344
# ╠═a6bdc90f-3c10-4524-9892-6e976d556a95
# ╠═5ef75bb5-5ab0-4e98-9026-a8e2c5b7833d
# ╠═b901a1c4-d346-4cfe-a6db-cae99ae792ca
# ╠═372d0743-2825-4a67-b2ba-588bc6b7d0d5
# ╠═f05c24f6-9506-4403-a458-9e201e344757
# ╠═f5e6e461-dbd6-4898-a88d-4d3b354812ab
# ╠═4eb50ae4-b5c2-4252-8d80-b758f838a5fb
# ╠═638561f3-6e6d-4faf-9daa-98694173e0a2
# ╠═4c8c7b5c-7112-4209-8b0b-a2463e2118a9
# ╠═1432bcd6-87b2-4881-867e-b190f8beb140
# ╠═1659b97f-d944-418a-adfc-0091a15efc84
# ╠═90a8908a-5f81-40b1-acde-4fefd210709d
# ╠═925688d4-78b4-404e-b4fd-0be2eaa31235
