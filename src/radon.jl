export radon

"""
    radon(I, θs; backend=CPU())

Calculates the parallel radon transform of the three dimensional AbstractArray `I`.
The first two dimensions are y and x. The third dimension is z, the rotational axis.

`θs` is a vector or range storing the angles in radians.

`backend` can be either `CPU()` for multithreaded CPU execution or
`CUDABackend()` for CUDA execution. 


Please note: the implementation is not quite optimize for cache efficiency and 
it is a very naive algorithm. But still, it is quite fast.
"""
function radon(I::AbstractArray{T, 3},  θs; backend=CPU()) where T			
	sz = size(I)
	@assert sz[1] == sz[2]
    @assert iseven(sz[1]) "Array needs to have a even number along x and y"
    
	kernel! = radon_kernel!(backend)

	I_itp = let
		if I isa CuArray
			I_itp = interpolate(I, (BSpline(Linear()), BSpline(Linear()), NoInterp()))
			I_itp = adapt(CuArray{eltype(I)}, I_itp);
		else
			I_itp = interpolate(I, (BSpline(Linear()), BSpline(Linear()), NoInterp()))
		end
	end
	
	# center coordinate
	cc = sz[1] ÷ 2 + 1
	# radius of the object, no rays outside of the radius are considered
	R = (sz[1] - 3) ÷ 2
	
	# all rays, in index steps
	rays_y = -R:one(T):R

	prop = 0:2*R

    sinogram = similar(I, size(I, 1), length(θs), size(I, 3))
    fill!(sinogram, 0)
	
	kernel!(sinogram, I_itp, θs, rays_y, prop, cc, R, ndrange=(length(rays_y), length(θs), sz[3]))

    sinogram ./= maximum(sinogram)

	return sinogram
end

@kernel function radon_kernel!(sinogram::AbstractArray{T}, I_itp,
			θs, rays_y, prop, cc, R) where T
    i_rays_y, iθ, i_z = @index(Global, NTuple)

	θ = θs[iθ]
	sθ = -sin(θ)
	cθ = cos(θ)

	y_ray = rays_y[i_rays_y]

	# x is the coordinate being on the circle
	x_s = R * cos(asin(T(y_ray) / R))
	x = cc + x_s * cθ - y_ray * sθ
	y = cc + x_s * sθ + y_ray * cθ
	#factor = exp(T(-0.01))
	#value = sinogram[round(Int, y_ray) + cc - 1 , i_z, iθ]

	tmp = zero(T)
	for i_prop = 1:round(Int, (2 * x_s) / prop[end] * length(prop))
		prop_dist = prop[i_prop]
		Δy = - sθ * prop_dist
		Δx = - cθ * prop_dist

		#d = distance_to_boundary(x + Δx - cc, y + Δy - cc, θ, R)
        #@show (d," ", prop_dist, "\n")
        #break
		tmp += I_itp(y + Δy, x + Δx, i_z)# * exp(-T(0.02) * d)
	end

	sinogram[round(Int, y_ray) + cc - 1, iθ, i_z] = tmp
end


