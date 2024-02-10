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
using Zygote, Optim, RadonKA, TestImages, ImageShow, Noise, Plots,PlutoUI,Statistics

# ╔═╡ a0057ab8-5c84-4f63-882f-1ac6f84fbc5e
using CUDA

# ╔═╡ 2f0ca285-fd1f-42bf-bb07-7fafb7ddd7f6
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# ╔═╡ d2d039dd-b584-4f72-b16b-1d54841ee767
md"# Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ f4ca400d-2405-40bf-95d8-35580b9871c3
begin
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ 063d95c7-d150-48f6-b04a-b2026ac4ba69
md""" ## CUDA
Thanks to Julia multiple dispatch and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) our code can run on CUDA GPUs.
Big reconstructions can run 5-20 times faster on a CUDA GPU!

Your CUDA is functional: **$(use_CUDA[])**
"""

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
md"""# Simple Backprojection

Typical filtered backprojection which does not perform super well with noise
"""

# ╔═╡ 464f022d-0773-43dd-81a7-ca2f6fc91634
img_bp = backproject_filtered(measurement, angles);

# ╔═╡ 49e59001-0e8d-4872-ae50-47c38486b3fd
img_backproject = backproject(measurement, angles);

# ╔═╡ 6fac5606-2350-4f22-9f35-120936114d5a
[simshow(img_bp) simshow(img_backproject)]

# ╔═╡ 1feccdec-cc35-4cc8-9a76-0e0e99bc7be3
md"# Optimization with gradient descent

We construct the loss function `f` and its gradient. 
This format is typically used with [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/).
The gradient is calculated by automatic differentiation and the reverse rules of RadonKA.jl
"

# ╔═╡ cdf0f1fb-c95f-43ab-a41f-159f4963f3af
function make_fg!(fwd_operator, measurement; λ=0.01f0, regularizer=x -> zero(eltype(x)))

	f(x) = mean(abs2, fwd_operator(x) .- measurement) + λ * regularizer(x)

 	# some Optim boilerplate to get gradient and loss as fast as possible
    fg! = let f=f
        function fg!(F, G, x)  
            # Zygote calculates both derivative and loss
            if G !== nothing
                y, back = Zygote.withgradient(f, x)
                # calculate gradient
                G .= back[1]
                if F !== nothing
                    return y
                end
            end
            if F !== nothing
                return f(x)
            end
        end
    end 
	return fg!
end

# ╔═╡ 6a6b53c8-b5f2-46e6-ada1-ee53d99db4d2
fg! = make_fg!(x -> radon(x, angles), measurement)

# ╔═╡ 8f7688e7-208e-466b-b58d-b3e92aae87b3
init0 = ones(Float32, size(img));

# ╔═╡ d65efdb0-4c80-4119-9fcd-185d13c37b2a
@time fg!(copy(init0), copy(init0), init0)

# ╔═╡ fc5aae11-793a-47f0-9759-2cc5d856d6a4
@time res = Optim.optimize(Optim.only_fg!(fg!), init0, LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ 27488594-0dfd-419d-be5c-0047b2ebdb59
plot([a.value for a in res.trace], xlabel="Iterations", ylabel="loss value", yscale=:log10)

# ╔═╡ 9f3d751e-6576-420d-8e5f-a4245be1bb0f
[simshow(res.minimizer) simshow(img_bp) simshow(img)]

# ╔═╡ 538fec79-94c8-4374-9176-c4dc2c149a74
md"# Add a TV regularizer
We add a [Total Variation](https://en.wikipedia.org/wiki/Total_variation_denoising#2D_signal_images) regularizer.
"

# ╔═╡ 4b6a0cfe-a0a1-486f-ad56-4cf2bf5db430
reg(x) = sum(sqrt.((circshift(x, (1,0)) .- x).^2 .+ (circshift(x, (0,1)) .- x).^2 .+ 1f-8))

# ╔═╡ 70aa0bff-41f7-409b-a191-10f196dd9233
@time reg(init0)

# ╔═╡ a69281cf-f024-4cd6-9ae8-a6f52644956a
fg2! = make_fg!(x -> radon(x, angles), measurement, regularizer=reg, λ=0.002f0)

# ╔═╡ a9d74730-2096-4e79-85b6-323ef8a2f54c
@time res2 = Optim.optimize(Optim.only_fg!(fg2!), init0, LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ 77d2fdf2-e0b2-45f4-ab80-bd03c871baf0
plot([a.value for a in res2.trace], yscale=:log10, xlabel="iterations", ylabel="loss")

# ╔═╡ ef15aef6-06c2-4a57-8ac0-86e1253ce895
md"
----------------with TV ---------------------- without TV ---------------- filtered backprojection ----------- ground truth
"

# ╔═╡ 3355590b-00d3-4f37-ae4c-8c69dd5dac5f
[simshow(res2.minimizer) simshow(res.minimizer) simshow(img_bp) simshow(img) ]

# ╔═╡ d91046bd-f5cb-4924-b518-1446a0029b80
md"# Apply Anscombe transform
The [Anscombe transform](https://en.wikipedia.org/wiki/Anscombe_transform) helps in the case of Poisson shot noise (as in our case).
Visually the Anscombe transform results in the best reconstructions.
"

# ╔═╡ aa13314c-aa98-4952-a5be-02fb02c50709
function make_fg_anscombe!(fwd_operator, measurement)

	f(x) = sum(abs2, sqrt.(max.(0, fwd_operator(x)) .+ 3f0/8f0) .- sqrt.(3f0 / 8f0 .+ measurement))

 	# some Optim boilerplate to get gradient and loss as fast as possible
    fg! = let f=f
        function fg!(F, G, x)  
            # Zygote calculates both derivative and loss
            if G !== nothing
                y, back = Zygote.withgradient(f, x)
                # calculate gradient
                G .= back[1]
                if F !== nothing
                    return y
                end
            end
            if F !== nothing
                return f(x)
            end
        end
    end 
	return fg!
end

# ╔═╡ 0f73f624-3fd9-42a9-b623-d1be11bbe5af
fg_ans! = make_fg_anscombe!(x -> radon(x, angles), 2000 .* measurement)

# ╔═╡ 2428fb0d-c911-4ac4-9dec-0fb19d9ad466
@time res_ans = Optim.optimize(Optim.only_fg!(fg_ans!), init0, LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ b66fcc8f-3f01-4163-9e49-d46b6370390d
md"
-------- Anscombe transform ------------ with TV ---------------- filtered backprojection ----------- ground truth
"

# ╔═╡ 40bff2bd-fbbb-4840-92f5-7319202afeb5
[simshow(res_ans.minimizer) simshow(res2.minimizer) simshow(img_bp) simshow(img) ]

# ╔═╡ 2692dda1-fcd7-408e-bf33-0511597513fc
md"# Try with CUDA
On my multithreaded CPU it takes around 4 seconds.
With CUDA it takes 0.2 seconds!
"

# ╔═╡ 4f432699-3b0a-4275-8594-57904cd5d7ba
angles_c = togoc(angles);

# ╔═╡ 443f9400-3cbd-4e41-b761-1c15c5bb537d
fg_cuda! = make_fg!(x -> radon(x, angles_c), togoc(measurement))

# ╔═╡ 7c12b32f-1098-4050-b3f0-a89a53bb569e
fg_cuda!(togoc(init0), togoc(init0), togoc(init0));

# ╔═╡ 9060c58c-95fb-4616-b474-2193b41aab4f
@mytime res_cuda = Optim.optimize(Optim.only_fg!(fg_cuda!), togoc(init0), LBFGS(),
                                 Optim.Options(iterations = 20,  
                                               store_trace=true))

# ╔═╡ 7107f76e-7033-4e2f-ad44-d8486072c3a1
md"
-------------------- CUDA ------------------- without TV ---------------- filtered backprojection ----------- ground truth
"

# ╔═╡ e72103d4-2940-4a60-addd-6d8990d8f0cf
[simshow(Array(res_cuda.minimizer)) simshow(res.minimizer) simshow(img_bp) simshow(img) ]

# ╔═╡ 2b71a257-b3d1-4df1-83d2-4e0c7eeb4af8
md"# Try with Optimization.jl"

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
@mytime res5 = solve(problem, OptimizationOptimisers.Adam(0.01), maxiters=500);

# ╔═╡ 68df9047-a898-4445-992a-cdfd9b4dc910
@mytime res6 = solve(problem, OptimizationOptimJL.LBFGS(), maxiters=20)

# ╔═╡ b054e438-d3f2-4df9-b504-5281e48287d4
[simshow(Array(res5.u)) simshow(Array(res6.u)) simshow(Array(res_cuda.minimizer))]

# ╔═╡ Cell order:
# ╟─d2d039dd-b584-4f72-b16b-1d54841ee767
# ╠═72f79d9e-a0dc-11ee-3eee-979a9f94c302
# ╠═1a255b6f-e83a-481b-8017-ecc996bc4929
# ╠═063d95c7-d150-48f6-b04a-b2026ac4ba69
# ╠═a0057ab8-5c84-4f63-882f-1ac6f84fbc5e
# ╠═f4ca400d-2405-40bf-95d8-35580b9871c3
# ╠═a351afa1-33e6-4f03-ae06-2215646b70db
# ╟─0b49cba6-d506-422e-8bba-35a6e3476f37
# ╠═5fc5b324-90ca-4e4a-8b5b-35b9f276c47d
# ╟─274e71b3-d4b5-406b-848b-6c8847179125
# ╟─b36b1679-d4bd-4a7d-b55e-c0ad930be9d5
# ╠═e0043bf0-c55e-4160-a97b-41e2acceb19f
# ╠═8b8a4aa2-e583-4f50-9082-06b3511b853e
# ╟─d2f035c7-8bc4-4c2b-aeb5-56ce955b8f4c
# ╟─144d3c73-8b7e-4049-924e-a2e20423f3f7
# ╠═464f022d-0773-43dd-81a7-ca2f6fc91634
# ╠═49e59001-0e8d-4872-ae50-47c38486b3fd
# ╟─6fac5606-2350-4f22-9f35-120936114d5a
# ╟─1feccdec-cc35-4cc8-9a76-0e0e99bc7be3
# ╠═cdf0f1fb-c95f-43ab-a41f-159f4963f3af
# ╠═6a6b53c8-b5f2-46e6-ada1-ee53d99db4d2
# ╠═d65efdb0-4c80-4119-9fcd-185d13c37b2a
# ╠═8f7688e7-208e-466b-b58d-b3e92aae87b3
# ╠═fc5aae11-793a-47f0-9759-2cc5d856d6a4
# ╠═27488594-0dfd-419d-be5c-0047b2ebdb59
# ╠═9f3d751e-6576-420d-8e5f-a4245be1bb0f
# ╟─538fec79-94c8-4374-9176-c4dc2c149a74
# ╠═4b6a0cfe-a0a1-486f-ad56-4cf2bf5db430
# ╠═70aa0bff-41f7-409b-a191-10f196dd9233
# ╠═a69281cf-f024-4cd6-9ae8-a6f52644956a
# ╠═a9d74730-2096-4e79-85b6-323ef8a2f54c
# ╟─77d2fdf2-e0b2-45f4-ab80-bd03c871baf0
# ╟─ef15aef6-06c2-4a57-8ac0-86e1253ce895
# ╟─3355590b-00d3-4f37-ae4c-8c69dd5dac5f
# ╟─d91046bd-f5cb-4924-b518-1446a0029b80
# ╠═aa13314c-aa98-4952-a5be-02fb02c50709
# ╠═0f73f624-3fd9-42a9-b623-d1be11bbe5af
# ╠═2428fb0d-c911-4ac4-9dec-0fb19d9ad466
# ╠═b66fcc8f-3f01-4163-9e49-d46b6370390d
# ╠═40bff2bd-fbbb-4840-92f5-7319202afeb5
# ╟─2692dda1-fcd7-408e-bf33-0511597513fc
# ╠═4f432699-3b0a-4275-8594-57904cd5d7ba
# ╠═443f9400-3cbd-4e41-b761-1c15c5bb537d
# ╠═7c12b32f-1098-4050-b3f0-a89a53bb569e
# ╠═9060c58c-95fb-4616-b474-2193b41aab4f
# ╟─7107f76e-7033-4e2f-ad44-d8486072c3a1
# ╟─e72103d4-2940-4a60-addd-6d8990d8f0cf
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
