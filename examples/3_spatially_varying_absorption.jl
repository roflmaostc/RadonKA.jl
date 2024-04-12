### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 009ad239-5f90-4c9e-9d3c-23d58cb5f4af
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ aa5e0f2d-8b11-447d-be55-c84a1421cafd
using Zygote, Optim, RadonKA, TestImages, ImageShow, Noise, Plots,PlutoUI,Statistics

# ╔═╡ 7cfebbbc-e36f-474c-879d-dc0a8e360f6d
using IndexFunArrays

# ╔═╡ 059f357a-b1fc-4f6d-af5f-72ffd47cb64c
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# ╔═╡ 3e7e2e30-cb85-4eb8-b4f5-b8b19ecade8c
md"# Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ 8abe2914-922f-4ea5-8d85-b435a22c897f
TableOfContents()

# ╔═╡ dad29ebd-c69b-42ed-aef4-6bfc76d59a32
md"# 1. Load Image"

# ╔═╡ 604ed50d-616c-4fc9-8adc-d2e29595c504
img = Float32.(testimage("resolution_test_512"))[200:421, 200:421];

# ╔═╡ a4e4eb4e-da5f-4f91-bbf9-73b983968c16
simshow(img)

# ╔═╡ 490f4bb3-6ab5-47be-b2cc-f30f13cc2721
md"# 2. Specify Absorption map
This is a ring of high absorption.

Hence, a sample in the inner part of the absorption can not be measured well.
"

# ╔═╡ a89bbe2a-341c-4005-8243-f2c0cdcbd722
μ = ((rr(img) .< 60) .-  (rr(img) .< 59)) * 1;

# ╔═╡ b1b301d4-abab-4d07-9c00-3782db4a5ee4
simshow(μ)

# ╔═╡ 7e2d4d16-2ac5-4d44-b5f5-67f9343e59e0
md"# 3. Create Sinogram"

# ╔═╡ f82bfaa5-b394-4159-99d4-9cc859f12f32
angles = range(0, 2π, 200)

# ╔═╡ e9a404d0-e9bb-468c-9490-ec2d12039f20
sinogram = radon(img, angles, μ=μ)

# ╔═╡ 0bbef52a-e191-46c7-8aa0-ce105d07b54b
simshow(sinogram)

# ╔═╡ 32fa73fa-726d-4b03-bae1-f8094b7c4a5c
md"# 4. Backproject"

# ╔═╡ 32b2b1ca-aa80-4931-9d03-9f19340c1ca8
img_b = backproject(sinogram, angles, μ=μ);

# ╔═╡ 4e5e1bb1-eeb9-43f7-91a8-ba7b8b76dccd
simshow(img_b)

# ╔═╡ b7892246-8fe0-437c-a224-f1ca4661ef33
md"# 5. Try with Optimization.jl
The optimization is able to take this into account and can reconstruct a decent image.
"

# ╔═╡ 8a8e536a-c6a3-412b-b986-f33f2620429a
measurement = (sinogram);

# ╔═╡ d8edea79-7775-4718-9637-127df8b68715
opt_f(x, p) = sum(abs2, radon(x, angles, μ=μ) .- measurement)

# ╔═╡ 83c920b7-e932-4deb-ab4c-08ec4412f0c9
opt_fun = OptimizationFunction(opt_f, AutoZygote())

# ╔═╡ 06680604-1426-4f2f-963d-412b73f31c33
init0 = zeros(size(img));

# ╔═╡ 26af8e6e-4afa-4d73-9bbc-630e07bd6694
opt_f(init0, angles)

# ╔═╡ fd3a4f13-27c8-4766-9e58-07050aa1db2f
problem = OptimizationProblem(opt_fun, init0, angles);

# ╔═╡ 26015d23-e708-4ebc-b718-18bd5104000b
@time res6 = solve(problem, OptimizationOptimJL.LBFGS(), maxiters=20)

# ╔═╡ 5f5fb511-4b55-43bb-b8ac-f520e8b3bbc7
simshow(res6)

# ╔═╡ Cell order:
# ╠═3e7e2e30-cb85-4eb8-b4f5-b8b19ecade8c
# ╠═8abe2914-922f-4ea5-8d85-b435a22c897f
# ╠═009ad239-5f90-4c9e-9d3c-23d58cb5f4af
# ╠═aa5e0f2d-8b11-447d-be55-c84a1421cafd
# ╠═7cfebbbc-e36f-474c-879d-dc0a8e360f6d
# ╟─dad29ebd-c69b-42ed-aef4-6bfc76d59a32
# ╠═604ed50d-616c-4fc9-8adc-d2e29595c504
# ╠═a4e4eb4e-da5f-4f91-bbf9-73b983968c16
# ╟─490f4bb3-6ab5-47be-b2cc-f30f13cc2721
# ╠═a89bbe2a-341c-4005-8243-f2c0cdcbd722
# ╠═b1b301d4-abab-4d07-9c00-3782db4a5ee4
# ╟─7e2d4d16-2ac5-4d44-b5f5-67f9343e59e0
# ╠═f82bfaa5-b394-4159-99d4-9cc859f12f32
# ╠═e9a404d0-e9bb-468c-9490-ec2d12039f20
# ╠═0bbef52a-e191-46c7-8aa0-ce105d07b54b
# ╟─32fa73fa-726d-4b03-bae1-f8094b7c4a5c
# ╠═32b2b1ca-aa80-4931-9d03-9f19340c1ca8
# ╠═4e5e1bb1-eeb9-43f7-91a8-ba7b8b76dccd
# ╟─b7892246-8fe0-437c-a224-f1ca4661ef33
# ╠═059f357a-b1fc-4f6d-af5f-72ffd47cb64c
# ╠═8a8e536a-c6a3-412b-b986-f33f2620429a
# ╠═d8edea79-7775-4718-9637-127df8b68715
# ╠═83c920b7-e932-4deb-ab4c-08ec4412f0c9
# ╠═26af8e6e-4afa-4d73-9bbc-630e07bd6694
# ╠═06680604-1426-4f2f-963d-412b73f31c33
# ╠═fd3a4f13-27c8-4766-9e58-07050aa1db2f
# ╠═26015d23-e708-4ebc-b718-18bd5104000b
# ╠═5f5fb511-4b55-43bb-b8ac-f520e8b3bbc7
