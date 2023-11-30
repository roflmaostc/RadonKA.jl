export iradon




"""
    iradon(sinogram, θs; backend=CPU())

`backend` can be either `CPU()` for multithreaded CPU execution or
`CUDABackend()` for CUDA execution. 


"""
function iradon(sinogram::AbstractArray{T, 3}, angles::AbstractArray{T, 1};
			    backend=CPU()) where T
    #@assert isodd(size(sinogram, 1)) && size(sinogram, 2) == length(angles)
    # this is the actual size we are using for calculation
	N = size(sinogram, 1)
	N_angles = size(angles, 1)

    # the only significant allocation
	img = similar(sinogram, (N + 1, N + 1, size(sinogram, 3)))
	fill!(img, 0)

    # radius of the cylinder we are projecting through
	radius = size(img, 1) ÷ 2 - 1
    # mid point, it is actually N ÷ 2 + 1
    # but because of how adress the indices, we need 1.5 instead of +1
    mid = size(img, 1) ÷ 2 + 1#T(1)
    # the y_dists samplings, in principle we can add this as function parameter 
	y_dists = similar(img, (size(img, 1) - 1, ))
	y_dists .= -radius:radius

	#@show typeof(sinogram), typeof(img), typeof(y_dists), typeof(angles)
	kernel! = iradon_kernel!(backend)
	kernel!(sinogram::AbstractArray{T}, img, y_dists, angles, mid, radius,
					ndrange=(N, N, size(img, 3)))
	
	return img 
end

"""
    iradon_kernel!(sinogram, img,
                  y_dists, angles, mid, radius)

"""
@kernel function iradon_kernel!(sinogram::AbstractArray{T},
			img, y_dists, angles, mid, radius) where T
    # r is the index of the angles
    # k is the index of the detector spatial coordinate
    # i_z is the index for the z dimension
	i_i, i_j, i_z = @index(Global, NTuple)

    if (i_i - mid)^2 + (i_j - mid)^2 > radius^2
    
    else
	    l = 1
        # acculumators of the intensity
	    tmp = zero(T)
        
        for i_angle in 1:length(angles)
            angle = angles[i_angle]
#            if 0 ≤ angle ≤ π / 2 
                # we need to find out which ray potentially intersects those positions
                contact_x = sin(angle)
                contact_y = -cos(angle)
                dir_x = cos(angle)
                dir_y = sin(angle)
                if 0 ≤ angle < π
                    y_intersect = mid + floor(Int, (dir_x * (i_i - mid - contact_x * radius) + dir_y * (i_j - mid - contact_y * radius)))
                elseif π <= angle
                    y_intersect = mid + floor(Int, (dir_x * (i_i - mid - contact_x * radius) + dir_y * (i_j - mid - contact_y * radius)))
                end

                #println(angle, " ", y_intersect)

            tmp += sinogram[y_intersect, i_angle, i_z] 
	    end
        img[i_j,i_i, i_z] = tmp
    end
end





function filtered_backprojection(sinogram::AbstractArray{T, 3}, θs, μ=nothing; backend=CPU()) where T
    filter = similar(sinogram, (size(sinogram, 1),))
    filter .= rr(T, (size(sinogram, 1), )) 

    p = plan_fft(sinogram, (1,))
    sinogram = real(inv(p) * (p * sinogram .* ifftshift(filter)))
    return iradon(sinogram, θs, μ, backend=backend)
end

