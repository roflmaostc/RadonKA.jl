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

# ╔═╡ fa97e620-1011-11ee-06dc-5d1044e8f54a
begin
	begin
		using Pkg
		Pkg.activate(".")
		Pkg.develop(path="../.")
		using Revise
	end
end

# ╔═╡ 998f1ff8-3cc2-4919-89da-d015f884dc0c
using RadonKA, ImageShow, IndexFunArrays, PlutoUI, CUDA

# ╔═╡ 507fc6bb-9314-432b-9d19-0f0959e93491
md"# Load Package"

# ╔═╡ b6b3a84a-5ae1-4e8f-b7f3-1fac92bceac0
TableOfContents()

# ╔═╡ b34a7de8-ee95-4836-a29e-5dd0a8c673c2
md"# Simple Image"

# ╔═╡ 1c9d828d-eb69-4792-9fdb-9698939ba69f
begin
	img = zeros(Float32, (768, 768, 3))
	
	img .+= box(img, (60,30, 300), offset=(300, 250, 10))
	img .+= box(img, (100,300, 300), offset=(150, 250, 10))

	img .+= rr2(img, offset=(300, 350, 10)) .< 60^2
	img .-= rr2(img, offset=(300, 350, 10)) .< 40^2
	img .+= rr2(img, offset=(300, 350, 10)) .< 20^2

	#img .+= box(img, (20,10), offset=(140, 100))
end;

# ╔═╡ 105f9f3e-9af8-415c-b191-2c03da508c01
@bind depth Slider(1:size(img, 3), show_value=true)

# ╔═╡ 129389ea-e5c6-4eb4-b790-3ce9d7a8ec26
simshow(img[:, :, depth])

# ╔═╡ db6f96e5-cbad-40f6-abda-09f64ce988ff
md"# Radon Transform"

# ╔═╡ 937e9211-e37d-4424-82d3-7542d9ce934d
θs = range(0f0, Float32(π), 1000)

# ╔═╡ c3fa0010-b173-482d-b1f8-84683ed4927f
@time sinogram = radon(img, θs, 0.003f0);

# ╔═╡ b79839d8-c0c7-4534-9140-ee1efc624f88
img_c = CuArray(img);

# ╔═╡ e5daf24c-c814-45bc-b00c-d0a0080ed60c
CUDA.@time CUDA.@sync sinogram_c = radon(img_c, θs, 0.003f0, backend=CUDABackend());

# ╔═╡ 4e039f6a-dfa8-4fe2-8680-2ec2646e6788
@bind depth2 Slider(1:size(img, 3), show_value=true)

# ╔═╡ 80c0c8da-87ec-4d2f-b389-198e0ddb49ac
simshow(sinogram[:, :, depth2])

# ╔═╡ f88a45d5-181b-4afd-94f3-0584ce14af1f
md"# Iradon Transform"

# ╔═╡ b39d0bed-13d5-4761-b23a-ddbe361f63ec
@time img_i = iradon(sinogram, θs, 0.003f0);

# ╔═╡ 38f0614a-3548-4ccb-a139-b609fac879da
CUDA.@time CUDA.@sync img_i_c = iradon(sinogram_c, θs, 0.003f0, backend=CUDABackend());

# ╔═╡ ae62edef-f0fc-4467-a60c-ceaf632fb742
0.63 / 0.003

# ╔═╡ ac354458-b688-4ba8-9425-36fbd835a3ec
@bind depth3 Slider(1:size(img, 3), show_value=true)

# ╔═╡ 3c560485-9479-422e-a064-b5fe7ae3fcc7
simshow(img_i[:, :, depth3])

# ╔═╡ 74cb060f-4ed5-4c80-a9a3-5ff286f56fec
simshow(Array(img_i_c)[:, :, depth])

# ╔═╡ 332987aa-6003-41cc-8fb1-4bdc5e3f5ad4
simshow(abs.(Array(img_c .- img_i_c)[:, :, depth]))

# ╔═╡ 79a3fdd5-495f-4d3c-9d8b-178bce112d45
sum(img_c)

# ╔═╡ 46b14542-9b25-40cc-9ad9-0568a9bed6de
sum(img_i_c) / 768 / 768 * π /4

# ╔═╡ 86baebe1-f1a7-496a-98ec-f6d3ac0ab63f
md"# Filtered Backprojection"

# ╔═╡ 449de2b7-c8db-4d6e-a4fb-2ac82b218cc4
img_filtered_c = RadonKA.filtered_backprojection(sinogram_c, θs, backend=CUDABackend());

# ╔═╡ 1c742967-7b6c-4ad8-b4ea-dcb6bb2774ef
simshow(Array(img_filtered_c)[:, :, depth3])

# ╔═╡ 90467d6e-acbb-450f-82fe-9f46096d1b0a
simshow(Array(img_filtered_c)[:, :, depth3])

# ╔═╡ Cell order:
# ╠═fa97e620-1011-11ee-06dc-5d1044e8f54a
# ╟─507fc6bb-9314-432b-9d19-0f0959e93491
# ╟─b6b3a84a-5ae1-4e8f-b7f3-1fac92bceac0
# ╠═998f1ff8-3cc2-4919-89da-d015f884dc0c
# ╟─b34a7de8-ee95-4836-a29e-5dd0a8c673c2
# ╠═1c9d828d-eb69-4792-9fdb-9698939ba69f
# ╟─105f9f3e-9af8-415c-b191-2c03da508c01
# ╠═129389ea-e5c6-4eb4-b790-3ce9d7a8ec26
# ╟─db6f96e5-cbad-40f6-abda-09f64ce988ff
# ╠═937e9211-e37d-4424-82d3-7542d9ce934d
# ╠═c3fa0010-b173-482d-b1f8-84683ed4927f
# ╠═b79839d8-c0c7-4534-9140-ee1efc624f88
# ╠═e5daf24c-c814-45bc-b00c-d0a0080ed60c
# ╟─4e039f6a-dfa8-4fe2-8680-2ec2646e6788
# ╠═80c0c8da-87ec-4d2f-b389-198e0ddb49ac
# ╟─f88a45d5-181b-4afd-94f3-0584ce14af1f
# ╠═b39d0bed-13d5-4761-b23a-ddbe361f63ec
# ╠═38f0614a-3548-4ccb-a139-b609fac879da
# ╠═ae62edef-f0fc-4467-a60c-ceaf632fb742
# ╠═ac354458-b688-4ba8-9425-36fbd835a3ec
# ╠═3c560485-9479-422e-a064-b5fe7ae3fcc7
# ╠═74cb060f-4ed5-4c80-a9a3-5ff286f56fec
# ╠═332987aa-6003-41cc-8fb1-4bdc5e3f5ad4
# ╠═79a3fdd5-495f-4d3c-9d8b-178bce112d45
# ╠═46b14542-9b25-40cc-9ad9-0568a9bed6de
# ╟─86baebe1-f1a7-496a-98ec-f6d3ac0ab63f
# ╠═449de2b7-c8db-4d6e-a4fb-2ac82b218cc4
# ╠═1c742967-7b6c-4ad8-b4ea-dcb6bb2774ef
# ╠═90467d6e-acbb-450f-82fe-9f46096d1b0a
