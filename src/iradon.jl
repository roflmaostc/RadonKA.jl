export iradon

function filtered_backprojection(sinogram::AbstractArray{T, 3}, θs; backend=CPU()) where T
    filter = similar(sinogram, (size(sinogram, 1),))
    filter .= rr(T, (size(sinogram, 1), ))


    p = plan_fft(sinogram, (1,))
    sinogram = real(inv(p) * (p * sinogram .* ifftshift(filter)))
    return iradon(sinogram, θs, backend=backend)
end



function iradon(sinogram::AbstractArray{T},  θs; backend=CPU()) where T
	sz = (size(sinogram, 1), size(sinogram, 1), size(sinogram, 3))

	kernel! = iradon_kernel!(backend)

	sinogram_itp = let
		if sinogram isa CuArray
			sinogram_itp = interpolate(sinogram, (BSpline(Linear()), NoInterp(), NoInterp()))
			sinogram_itp = adapt(CuArray{eltype(sinogram)}, sinogram_itp);
		else
			sinogram_itp = interpolate(sinogram, (BSpline(Linear()), NoInterp(), NoInterp()))
		end
	end
	
	# center coordinate
	cc = sz[1] ÷ 2 + 1
	# radius of the object, no rays outside of the radius are considered
	R = (sz[1] - 1) ÷ 2
	

	# output
    I_s = similar(sinogram, (sz[1], sz[2], sz[3]))
    fill!(I_s, 0)
	# maybe sz[1] - 1
	kernel!(I_s, sinogram_itp, θs, cc, R, ndrange=(sz[1], sz[2], sz[3]))
	
	return I_s
end


@kernel function iradon_kernel!(I_s::AbstractArray{T}, sinogram_itp,
			θs, cc, R) where T
    i_y, i_x, i_z = @index(Global, NTuple)


	y = i_y - cc
	x = i_x - cc

	r2 = x^2 + y^2
    
    # only consider everything inside the radius
	if r2 <= R^2
		tmp = zero(T)
		for iθ = 1:length(θs)
			θ = θs[iθ]
			xn = +cos(θ) * y - x * sin(θ) + cc - 1#size(I_s, 1) ÷ 2 + 1
			#d = distance_to_boundary(x, y, θ, R)
			tmp += sinogram_itp(xn, iθ, i_z)# * exp(-0.02 * d)
		end

		I_s[i_y, i_x, i_z] = tmp
	end
end
