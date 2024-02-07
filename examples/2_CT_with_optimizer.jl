### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 72f79d9e-a0dc-11ee-3eee-979a9f94c302
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 1a255b6f-e83a-481b-8017-ecc996bc4929
using Zygote, Optim, RadonKA, TestImages, ImageShow, Noise, Plots

# ╔═╡ 8d73ca31-2b63-4543-a5e5-82fd08917140
using PlutoUI

# ╔═╡ a0057ab8-5c84-4f63-882f-1ac6f84fbc5e
using CUDA

# ╔═╡ 9310a824-ee3b-4e51-b870-57401c411809
using Statistics

# ╔═╡ 8a66c5e3-fa8c-42ec-82c3-7342dc9ca330
using Tullio

# ╔═╡ 2f0ca285-fd1f-42bf-bb07-7fafb7ddd7f6
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# ╔═╡ f4ca400d-2405-40bf-95d8-35580b9871c3
begin
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ 063d95c7-d150-48f6-b04a-b2026ac4ba69
md"# CUDA is enabled: $(use_CUDA[])"

# ╔═╡ a351afa1-33e6-4f03-ae06-2215646b70db
TableOfContents()

# ╔═╡ 0b49cba6-d506-422e-8bba-35a6e3476f37
md"# Set up testimage"

# ╔═╡ 5fc5b324-90ca-4e4a-8b5b-35b9f276c47d
img = Float32.(testimage("resolution_test_512"))[200:421, 200:421];

# ╔═╡ 274e71b3-d4b5-406b-848b-6c8847179125
simshow(img)

# ╔═╡ b36b1679-d4bd-4a7d-b55e-c0ad930be9d5
md"# Make a measurement
We also add some Poisson noise to the measurement.
"

# ╔═╡ e0043bf0-c55e-4160-a97b-41e2acceb19f
angles = range(0, 1f0 * π, 360)

# ╔═╡ 8b8a4aa2-e583-4f50-9082-06b3511b853e
measurement = poisson(radon(img, angles), 2000);

# ╔═╡ d2f035c7-8bc4-4c2b-aeb5-56ce955b8f4c
simshow(measurement)

# ╔═╡ 144d3c73-8b7e-4049-924e-a2e20423f3f7
md"""# Simple Backprojection"""

# ╔═╡ 464f022d-0773-43dd-81a7-ca2f6fc91634
img_bp = filtered_backprojection(measurement, angles);

# ╔═╡ 49e59001-0e8d-4872-ae50-47c38486b3fd
img_iradon = iradon(measurement, angles);

# ╔═╡ 6fac5606-2350-4f22-9f35-120936114d5a
[simshow(img_bp) simshow(img_iradon)]

# ╔═╡ 1feccdec-cc35-4cc8-9a76-0e0e99bc7be3
md"# Optimization with gradient descent

We construct the loss function `f` and its gradient `g!`. 
This format is typically used with [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/).
The gradient is calculated by automatic differentiation and the reverse rules of RadonKA.jl
"

# ╔═╡ e501ede0-62d7-42a1-b258-518be737bd9c
function make_fg!(fwd_operator, measurement; λ=0.01f0, regularizer=x -> zero(eltype(x)))

	f = let measurement=measurement
		f(x) = mean(abs2, fwd_operator(x) .- measurement) + λ * regularizer(x)
	end

	g! = let f=f
		g!(G, x) = begin
			if !isnothing(G)
				G .= Zygote.gradient(f, x)[1]
			end
		end
	end

	return f, g!
end

# ╔═╡ 6a6b53c8-b5f2-46e6-ada1-ee53d99db4d2
f, g! = make_fg!(x -> radon(x, angles), measurement)

# ╔═╡ 8f7688e7-208e-466b-b58d-b3e92aae87b3
init0 = zeros(Float32, size(img));

# ╔═╡ ae7a0720-78b4-472a-97d7-301c27c2b877
@time f(init0);

# ╔═╡ 482ae232-b188-4f34-90ea-797406b518ed
@time g!(copy(init0), init0);

# ╔═╡ fc5aae11-793a-47f0-9759-2cc5d856d6a4
res = Optim.optimize(f, g!, init0, LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ 3149e668-7479-42e4-969a-f2901a2be9d7
a = res.trace[1]

# ╔═╡ 27488594-0dfd-419d-be5c-0047b2ebdb59
plot([a.value for a in res.trace], xlabel="Iterations", ylabel="loss value", yscale=:log10)

# ╔═╡ 9f3d751e-6576-420d-8e5f-a4245be1bb0f
[simshow(res.minimizer) simshow(img_bp) simshow(img)]

# ╔═╡ 8b9c5804-877c-4679-a732-8042031892de
f(img_bp ./ maximum(img_bp) .* maximum(img))

# ╔═╡ 13695e7c-b91e-4443-a811-d705de1478e2
f(res.minimizer)

# ╔═╡ 538fec79-94c8-4374-9176-c4dc2c149a74
md"# Add a TV regularizer
[Tullio.jl](https://github.com/mcabbott/Tullio.jl) is a very elegant and performant way to add a regularizer to the reconstruction!
In principle all code except the regularizer runs fast with CUDA!
"

# ╔═╡ 7224f60a-c29a-48e3-8c99-689aa9587506
reg(x) = @tullio r := sqrt(1f-8 + (x[i,j] - x[i-1, j])^2 + (x[i,j] - x[i, j-1])^2)

# ╔═╡ 70aa0bff-41f7-409b-a191-10f196dd9233
reg(init0)

# ╔═╡ a69281cf-f024-4cd6-9ae8-a6f52644956a
f2, g2! = make_fg!(x -> radon(x, angles), measurement, regularizer=reg, λ=0.002f0)

# ╔═╡ a9d74730-2096-4e79-85b6-323ef8a2f54c
res2 = Optim.optimize(f2, g2!, init0, LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ 77d2fdf2-e0b2-45f4-ab80-bd03c871baf0
plot([a.value for a in res2.trace], yscale=:log10)

# ╔═╡ 3355590b-00d3-4f37-ae4c-8c69dd5dac5f
[simshow(res2.minimizer) simshow(res.minimizer) simshow(img_bp) simshow(img) ]

# ╔═╡ d91046bd-f5cb-4924-b518-1446a0029b80
md"# Apply Anscombe transform
The [Anscombe transform](https://en.wikipedia.org/wiki/Anscombe_transform) helps in the case of Poisson shot noise (as in our case).
Visually the Anscombe transform results in the best reconstructions.
"

# ╔═╡ bc38105d-ce12-400f-97a4-7059fa0ca72e
function make_fg_anscombe!(fwd_operator, measurement; λ=0.01f0, regularizer=x -> zero(eltype(x)))

	f = let measurement=measurement
		# apply sqrt for anscombe
		f(x) = sum(abs2, sqrt.(max.(0, fwd_operator(x)) .+ 3f0/8f0) .- sqrt.(3f0 / 8f0 .+ measurement)) + λ * regularizer(x)
	end

	g! = let f=f
		g!(G, x) = begin
			if !isnothing(G)
				G .= Zygote.gradient(f, x)[1]
			end
		end
	end

	return f, g!
end

# ╔═╡ 0f73f624-3fd9-42a9-b623-d1be11bbe5af
f3, g3! = make_fg_anscombe!(x -> radon(x, angles), 500 .* measurement, regularizer=reg, λ=0.0002f0 * 500)

# ╔═╡ 2428fb0d-c911-4ac4-9dec-0fb19d9ad466
@time res3 = Optim.optimize(f3, g3!, init0, LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ 40bff2bd-fbbb-4840-92f5-7319202afeb5
[simshow(res3.minimizer) simshow(res2.minimizer) simshow(img_bp) simshow(img) ]

# ╔═╡ bdd283a2-2ecd-4e66-927b-d961842af434
compare(a, b) = sum(abs2, a ./ mean(a) .- b ./ mean(b))

# ╔═╡ 3ccb13a4-d2d3-41f1-bf3a-24fe657c8b07
compare(res.minimizer, img)

# ╔═╡ 9e100052-07b2-4bce-b4e7-e85e2c2f3c7c
compare(res2.minimizer, img)

# ╔═╡ 35fb48f6-5a80-4eee-bbb8-c2baa98043fe
compare(res3.minimizer, img)

# ╔═╡ 2692dda1-fcd7-408e-bf33-0511597513fc
md"# Try with CUDA"

# ╔═╡ 4f432699-3b0a-4275-8594-57904cd5d7ba
angles_c = togoc(angles);

# ╔═╡ 443f9400-3cbd-4e41-b761-1c15c5bb537d
f_cuda, g_cuda! = make_fg!(x -> radon(x, angles_c), togoc(measurement))

# ╔═╡ 7c12b32f-1098-4050-b3f0-a89a53bb569e
g_cuda!(togoc(init0), togoc(init0));

# ╔═╡ 9060c58c-95fb-4616-b474-2193b41aab4f
@mytime res4 = Optim.optimize(f_cuda, g_cuda!, togoc(init0), LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ 559a8866-5094-4e59-ac09-222a002e052b
f_cuda2, g_cuda2! = make_fg_anscombe!(x -> radon(x, angles_c), togoc(measurement))

# ╔═╡ 3b2599ea-a8db-4e7d-a51f-d9b0db6389c4
@mytime res7 = Optim.optimize(f_cuda2, g_cuda2!, togoc(init0), LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ 34ec68ba-6130-448c-9bb0-04a4c3ed734c
[simshow(Array(res4.minimizer)) simshow(Array(res7.minimizer))]

# ╔═╡ e72103d4-2940-4a60-addd-6d8990d8f0cf
[simshow(Array(res4.minimizer)) simshow(res2.minimizer) simshow(res.minimizer) simshow(img_bp) simshow(img) ]

# ╔═╡ 2b71a257-b3d1-4df1-83d2-4e0c7eeb4af8
md"# Try with Optimization"

# ╔═╡ e2628d52-641d-4dcc-bfb0-6c3731d6a1c5
measurement_c = togoc(measurement);

# ╔═╡ f86ef3f3-1c24-4437-80c7-8ad27df23396
opt_f(x, p) = sum(abs2, sqrt.(max.(0, radon(x, angles_c)) .+ 3f0/8f0) .- sqrt.(3f0 / 8f0 .+ measurement_c))

# ╔═╡ 1745cb68-efd6-429a-8d0e-748b483ef1dd
opt_fun = OptimizationFunction(opt_f, AutoZygote())

# ╔═╡ fee409fd-1ee0-45c9-b012-1d69fef34957
init0_c = togoc(init0);

# ╔═╡ e494d255-0886-46ee-be80-5fd565eaf1b4
opt_f(init0_c, angles_c)

# ╔═╡ 48d0c679-1dfd-4629-bfaa-f6ef0350722a
problem = OptimizationProblem(opt_fun, init0_c, angles_c);

# ╔═╡ fb3b8d74-341b-4627-b361-a28f7ccaf75b
@mytime res5 = solve(problem, OptimizationOptimisers.Adam(0.01), maxiters=500)

# ╔═╡ 68df9047-a898-4445-992a-cdfd9b4dc910
@mytime res6 = solve(problem, OptimizationOptimJL.LBFGS(), maxiters=20)

# ╔═╡ b054e438-d3f2-4df9-b504-5281e48287d4
[simshow(Array(res5.u)) simshow(Array(res6.u)) simshow(Array(res4.minimizer))]

# ╔═╡ Cell order:
# ╠═72f79d9e-a0dc-11ee-3eee-979a9f94c302
# ╠═1a255b6f-e83a-481b-8017-ecc996bc4929
# ╠═8d73ca31-2b63-4543-a5e5-82fd08917140
# ╠═a0057ab8-5c84-4f63-882f-1ac6f84fbc5e
# ╟─063d95c7-d150-48f6-b04a-b2026ac4ba69
# ╠═f4ca400d-2405-40bf-95d8-35580b9871c3
# ╠═9310a824-ee3b-4e51-b870-57401c411809
# ╠═a351afa1-33e6-4f03-ae06-2215646b70db
# ╟─0b49cba6-d506-422e-8bba-35a6e3476f37
# ╠═5fc5b324-90ca-4e4a-8b5b-35b9f276c47d
# ╠═274e71b3-d4b5-406b-848b-6c8847179125
# ╟─b36b1679-d4bd-4a7d-b55e-c0ad930be9d5
# ╠═e0043bf0-c55e-4160-a97b-41e2acceb19f
# ╠═8b8a4aa2-e583-4f50-9082-06b3511b853e
# ╠═d2f035c7-8bc4-4c2b-aeb5-56ce955b8f4c
# ╟─144d3c73-8b7e-4049-924e-a2e20423f3f7
# ╠═464f022d-0773-43dd-81a7-ca2f6fc91634
# ╠═49e59001-0e8d-4872-ae50-47c38486b3fd
# ╠═6fac5606-2350-4f22-9f35-120936114d5a
# ╟─1feccdec-cc35-4cc8-9a76-0e0e99bc7be3
# ╠═e501ede0-62d7-42a1-b258-518be737bd9c
# ╠═6a6b53c8-b5f2-46e6-ada1-ee53d99db4d2
# ╠═8f7688e7-208e-466b-b58d-b3e92aae87b3
# ╠═ae7a0720-78b4-472a-97d7-301c27c2b877
# ╠═482ae232-b188-4f34-90ea-797406b518ed
# ╠═fc5aae11-793a-47f0-9759-2cc5d856d6a4
# ╠═3149e668-7479-42e4-969a-f2901a2be9d7
# ╠═27488594-0dfd-419d-be5c-0047b2ebdb59
# ╠═9f3d751e-6576-420d-8e5f-a4245be1bb0f
# ╠═8b9c5804-877c-4679-a732-8042031892de
# ╠═13695e7c-b91e-4443-a811-d705de1478e2
# ╟─538fec79-94c8-4374-9176-c4dc2c149a74
# ╠═8a66c5e3-fa8c-42ec-82c3-7342dc9ca330
# ╠═7224f60a-c29a-48e3-8c99-689aa9587506
# ╠═70aa0bff-41f7-409b-a191-10f196dd9233
# ╠═a69281cf-f024-4cd6-9ae8-a6f52644956a
# ╠═a9d74730-2096-4e79-85b6-323ef8a2f54c
# ╠═77d2fdf2-e0b2-45f4-ab80-bd03c871baf0
# ╠═3355590b-00d3-4f37-ae4c-8c69dd5dac5f
# ╟─d91046bd-f5cb-4924-b518-1446a0029b80
# ╠═bc38105d-ce12-400f-97a4-7059fa0ca72e
# ╠═0f73f624-3fd9-42a9-b623-d1be11bbe5af
# ╠═2428fb0d-c911-4ac4-9dec-0fb19d9ad466
# ╠═40bff2bd-fbbb-4840-92f5-7319202afeb5
# ╠═bdd283a2-2ecd-4e66-927b-d961842af434
# ╠═3ccb13a4-d2d3-41f1-bf3a-24fe657c8b07
# ╠═9e100052-07b2-4bce-b4e7-e85e2c2f3c7c
# ╠═35fb48f6-5a80-4eee-bbb8-c2baa98043fe
# ╟─2692dda1-fcd7-408e-bf33-0511597513fc
# ╠═4f432699-3b0a-4275-8594-57904cd5d7ba
# ╠═443f9400-3cbd-4e41-b761-1c15c5bb537d
# ╠═7c12b32f-1098-4050-b3f0-a89a53bb569e
# ╠═9060c58c-95fb-4616-b474-2193b41aab4f
# ╠═559a8866-5094-4e59-ac09-222a002e052b
# ╠═3b2599ea-a8db-4e7d-a51f-d9b0db6389c4
# ╠═34ec68ba-6130-448c-9bb0-04a4c3ed734c
# ╠═e72103d4-2940-4a60-addd-6d8990d8f0cf
# ╟─2b71a257-b3d1-4df1-83d2-4e0c7eeb4af8
# ╠═2f0ca285-fd1f-42bf-bb07-7fafb7ddd7f6
# ╠═e2628d52-641d-4dcc-bfb0-6c3731d6a1c5
# ╠═f86ef3f3-1c24-4437-80c7-8ad27df23396
# ╠═1745cb68-efd6-429a-8d0e-748b483ef1dd
# ╠═e494d255-0886-46ee-be80-5fd565eaf1b4
# ╠═fee409fd-1ee0-45c9-b012-1d69fef34957
# ╠═48d0c679-1dfd-4629-bfaa-f6ef0350722a
# ╠═fb3b8d74-341b-4627-b361-a28f7ccaf75b
# ╠═68df9047-a898-4445-992a-cdfd9b4dc910
# ╠═b054e438-d3f2-4df9-b504-5281e48287d4
