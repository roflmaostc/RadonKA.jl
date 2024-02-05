export iradon2

# handle 2D
function iradon2(sinogram::AbstractArray{T, 2}, angles::AbstractArray{T2, 1};
        geometry=RadonParallelCircle(size(sinogram,1) + 1,-(size(sinogram + 1,1)-1)÷2:(size(img,1)-1)÷2), μ=nothing) where {T, T2}
    view(iradon2(reshape(sinogram, (size(sinogram)..., 1)), angles_T; geometry, μ), :, :, 1)
end


"""
    iradon(sinogram, θs; <kwargs>)

`even`

See also [`radon`](@ref).

# Example
```
"""
function iradon2(sinogram::AbstractArray{T, 3}, angles_T::AbstractVector;
        geometry=RadonParallelCircle(size(sinogram,1) + 1, -(size(sinogram,1))÷2:(size(sinogram,1))÷2),
        μ=nothing) where T
    @assert size(sinogram, 2) == length(angles_T) "size of angles does not match sinogram size"
    @assert size(sinogram, 1) == size(geometry.in_height, 1)
    backend = KernelAbstractions.get_backend(sinogram)
 
    # angles_T might be a normal vector instead of Cuvector. fix it. 
    angles = similar(sinogram, (size(angles_T, 1),))
    angles .= typeof(angles)(angles_T) 


    # geometry can be very densely sampled, hence the sinogram depends on geometry size
    N = geometry.N
    # we only propagate inside this in circle
    # convert radius to correct float type, very important for performance!
    radius = T((N - 1) ÷ 2)
    # the midpoint of the array
    # convert to good type
    mid = T(N ÷ 2 + 1 +  1 // 2)
    N_angles = length(angles)

    @show N_angles

    img = similar(sinogram, (N, N,  size(sinogram, 3)))
    fill!(img, 0)

    # create an absorption function, maps just to 1 in case isnothing(μ)
    absorb_f = make_absorption_f(μ, T)

    # of the kernel goes
    kernel! = iradon_kernel2!(backend)
    kernel!(img, sinogram, geometry.in_height, angles, mid, radius, absorb_f,
            ndrange=(size(sinogram, 1), N_angles, size(img, 3)))
    KernelAbstractions.synchronize(backend)    
    return img 
end

@kernel function iradon_kernel2!(img, sinogram, in_height, angles, mid, radius, absorb_f) where {T, IT}
    i, iangle, i_z = @index(Global, NTuple)
    
    @inbounds sinα, cosα = sincos(angles[iangle])

    @inbounds xend, yend = T(size(img, 2)), T(in_height[i])

    # map the detector positions on a circle
    # also rotate according to the angle
    x_dist_rot, y_dist_rot, x_dist_end_rot, y_dist_end_rot = 
        next_ray_on_circle(yend, yend, mid, radius, sinα, cosα)

    # new will be always the current coordinates
    # end the final destination
    # and old is the previous one
    xnew = x_dist_rot
    ynew = y_dist_rot
    xend = x_dist_end_rot
    yend = y_dist_end_rot
    xold, yold = xnew, ynew



    tmp = zero(T)
    # we store old_direction
    # if the sign of old_direction changes, we have to stop tracing
    # because then we hit the end point 
    _, _, sxold, syold = next_cell_intersection(xnew, ynew, xend, yend)
    while true
        # find next intersection point with integer grid
        xnew, ynew, sx, sy = next_cell_intersection(xnew, ynew, xend, yend)

        # if we leave the circle or the direction of marching changes, this is the end
        inside_circ(xnew, ynew, mid, radius) && 
         (sx == sxold && sy == syold) || break
        
        # switch of i and j intentional to keep it consistent with existing code
        icell, jcell = find_cell(xnew, ynew, xold, yold)

        # calculate intersection distance
        distance = sqrt((xnew - xold)^2 + (ynew - yold) ^2)
        # add value to ray, potentially attenuate by attenuated exp factor
        if iangle > 99 
            @show iangle
        end
        img[icell, jcell, i_z] += distance * sinogram[i, iangle, i_z] * absorb_f(xnew, ynew, x_dist_rot, y_dist_rot)
        xold, yold = xnew, ynew
    end 
end
