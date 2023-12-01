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




function iradon2(sinogram::AbstractArray{T, 3}, angles::AbstractArray{T, 1};
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
    mid = size(img, 1) ÷ 2 + T(1.5)
    # the y_dists samplings, in principle we can add this as function parameter
	y_dists = similar(img, (size(img, 1) - 1, ))
	y_dists .= -radius:radius

	#@show typeof(sinogram), typeof(img), typeof(y_dists), typeof(angles)
	kernel! = iradon_kernel2!(backend)
	kernel!(sinogram::AbstractArray{T}, img, y_dists, angles, mid, radius,
					ndrange=(N_angles, N, size(img, 3)))

	return img
end

"""
    iradon_kernel!(sinogram, img,
                  y_dists, angles, mid, radius)

"""
@kernel function iradon_kernel2!(sinogram::AbstractArray{T},
			img, y_dists, angles, mid, radius) where T
    # r is the index of the angles
    # k is the index of the detector spatial coordinate
    # i_z is the index for the z dimension
	r, k, i_z = @index(Global, NTuple)

	angle = angles[r]
	sinα, cosα = sincos(angle)

    # x0, y0, x1, y1 beginning and end point of each ray
	a, b, c, d = next_ray_on_circle(img, angle, y_dists[k], mid, radius, sinα, cosα)

    # different comparisons depending which direction the ray is propagating
	cac = a <= c ? (a,c) -> a<=c : (a,c) -> a>=c
	cbd = b <= d ? (b,d) -> b<=d : (b,d) -> b>=d

	l = 1
    # acculumators of the intensity
	tmp = zero(T)
	while cac(a, c) && cbd(b, d)
		a_old, b_old = a, b
        # would be good to move this branch outside of the while loop
        # but maybe branch prediction is doing a good job here
		if a ≈ c && b ≈ d
			break
		end

        # find the next intersection for the ray
		@inline a, b = find_next_intersection(a,b,c,d)
		l += 1

        # find the cell it is cutting through
		@inline cell_i, cell_j = find_cell((a_old, b_old),
								   (a, b))

        # distance travelled through that cell
		distance = sqrt((a_old - a) ^2 +
						(b_old - b) ^2)
        # cell value times distance travelled through
        Atomix.@atomic img[cell_i, cell_j, i_z] += distance * sinogram[k, r, i_z]
	end
end

