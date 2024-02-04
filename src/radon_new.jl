export radon2
export RadonCircle

@inline function next_cell_intersection(x₀, y₀, x₁, y₁)
	# inspired by 
	floor_or_ceilx = x₁ ≥ x₀ ? ceil : floor
	floor_or_ceily = y₁ ≥ y₀ ? ceil : floor
	# https://cs.stackexchange.com/questions/132887/determining-the-intersections-of-a-line-segment-and-grid
	txx = (floor_or_ceilx(x₀)-x₀)
    txx = ifelse(txx != 0, txx, txx + sign(x₁ - x₀))
	tyy = (floor_or_ceily(y₀)-y₀)
    tyy = ifelse(tyy != 0, tyy, tyy + sign(y₁ - y₀))

	tx = txx / (x₁ - x₀)
	ty = tyy / (y₁ - y₀)
	# decide which t is smaller, and hence the next step
	t = ifelse(tx > ty, ty, tx)

	# calculate new coordinates
	x = x₀ + t * (x₁ - x₀)
	y = y₀ + t * (y₁ - y₀)
	return x, y, tx, ty
end

@inline function find_cell(x, y, xnew, ynew)
	x = floor(Int, (x + xnew) / 2)
	y = floor(Int, (y + ynew) / 2)
	return x, y
end

abstract type RadonGeometry end

struct RadonCircle{ToN, T} <: RadonGeometry
    μ::ToN
    detector::AbstractVector{T}
end



function radon2(img::AbstractArray{T, 3}, angles::AbstractArray{T, 1},
        geometry=RadonCircle(nothing, -(size(img,1)-1)÷2:(size(img,1)-1)÷2)) where T
    @assert size(img, 1) == size(img, 2) "Arrays has to be quadratically shaped"
    backend = KernelAbstractions.get_backend(img)

    N = size(img, 1) - 1
    # we only propagate inside this in circle
    radius = (size(img, 1) - 1) ÷ 2
    # the midpoint of the array
    mid = size(img, 1) ÷ 2 + 1 +  1 // 2 
    N_angles = length(angles)

    sinogram = similar(img, (N, N_angles, size(img, 3)))
    fill!(sinogram, 0)

    d_points = geometry.detector
    kernel! = radon_kernel2!(backend)

    kernel!(sinogram::AbstractArray{T}, img, d_points, angles, mid, radius,
    		ndrange=(N, N_angles, size(img, 3)))

    KernelAbstractions.synchronize(backend)    
    return sinogram
end


@kernel function radon_kernel2!(sinogram::AbstractArray{T}, img, d_points, angles, mid, radius) where T
    i, iangle, i_z = @index(Global, NTuple)
    
    sinα, cosα = sincos(angles[iangle])
    inside_arr(ii, jj, img) = (1 ≤ ii ≤ size(img, 1)) && (1 ≤ jj ≤ size(img, 2))
    inside_circ(ii, jj, img) = (ii - mid)^2 + (jj - mid)^2 ≤ radius ^2 


    xend, yend = size(img, 2), d_points[i]


    x_dist_rot, y_dist_rot, x_dist_end_rot, y_dist_end_rot = 
        next_ray_on_circle(yend, yend, mid, radius, sinα, cosα)


    xnew = x_dist_rot
    ynew = y_dist_rot
    xend = x_dist_end_rot
    yend = y_dist_end_rot
    xold, yold = xnew, ynew


    tmp = zero(T)
    # do while loop
    l = 1
    # we store old_direction
    # if the sign of old_direction changes, we have to stop tracing
    # because then we hit the end point 
    _, _, txold, tyold = next_cell_intersection(xnew, ynew, xend, yend)

    while l < 10_000 
        xnew, ynew, tx, ty = next_cell_intersection(xnew, ynew, xend, yend)
        inside_circ(xnew, ynew, img) && (xnew != xold || ynew != yold) && 
         (sign(tx) == sign(txold) && sign(ty) == sign(tyold)) || break
        if l > 9995
            @show xnew, ynew, xend, yend
        end
        
        jcell, icell = find_cell(xnew, ynew, xold, yold)

        distance = sqrt((xnew - xold)^2 + (ynew - yold) ^2)
        tmp += distance * img[icell, jcell, i_z]
        l += 1
        xold, yold = xnew, ynew
    end 
    sinogram[i, iangle, i_z] = tmp
end

