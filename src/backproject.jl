export backproject

# handle 2D
function backproject(sinogram::AbstractArray{T, 2}, angles::AbstractArray{T2, 1};
        geometry=RadonParallelCircle(size(sinogram,1) + 1,-(size(sinogram,1))÷2:(size(sinogram,1))÷2), μ=nothing) where {T, T2}
    view(backproject(reshape(sinogram, (size(sinogram)..., 1)), angles; geometry, μ), :, :, 1)
end


"""
    backproject(sinogram, θs; <kwargs>)

Conceptually the adjoint operation of [`radon`](@ref). Intuitively, the `backproject` smears rays back into the space.
See also [`radon`](@ref).

For filtered backprojection see [`backproject_filtered`](@ref).

# Example
```jldoctest
julia> arr = zeros((5,2)); arr[2,:] .= 1; arr[4, :] .= 1
2-element view(::Matrix{Float64}, 4, :) with eltype Float64:
 1.0
 1.0

julia> backproject(arr, [0, π/2])
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0  0.0       0.0  0.0       0.0
 0.0  0.0  0.0       0.0  0.0       0.0
 0.0  0.0  2.0       1.0  2.0       0.732051
 0.0  0.0  1.0       0.0  1.0       0.0
 0.0  0.0  2.0       1.0  2.0       0.732051
 0.0  0.0  0.732051  0.0  0.732051  0.0

julia> arr = ones((2,1)); 

julia> backproject(arr, [0], geometry=RadonFlexibleCircle(10, [-3, 3], [0,0]))
10×10 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0  0.0       0.0       0.0        0.0      0.0        0.0       0.0       0.0
 0.0  0.0  0.0       0.0       0.0        0.0      0.0        0.0       0.0       0.0
 0.0  0.0  0.0       0.0       0.335172   1.49876  0.335172   0.0       0.0       0.0
 0.0  0.0  0.0       0.0       1.08455    0.0      1.08455    0.0       0.0       0.0
 0.0  0.0  0.0       0.0       1.08455    0.0      1.08455    0.0       0.0       0.0
 0.0  0.0  0.0       1.00552   0.0790376  0.0      0.0790376  1.00552   0.0       0.0
 0.0  0.0  0.0       1.08455   0.0        0.0      0.0        1.08455   0.0       0.0
 0.0  0.0  0.591307  0.493247  0.0        0.0      0.0        0.493247  0.591307  0.0
 0.0  0.0  0.700352  0.0       0.0        0.0      0.0        0.0       0.700352  0.0
 0.0  0.0  0.0       0.0       0.0        0.0      0.0        0.0       0.0       0.0
```
"""
function backproject(sinogram::AbstractArray{T, 3}, angles_T::AbstractVector;
        geometry=RadonParallelCircle(size(sinogram,1) + 1,-(size(sinogram,1))÷2:(size(sinogram,1))÷2), μ=nothing) where {T}
    return _backproject(sinogram::AbstractArray{T, 3}, angles_T::AbstractVector, geometry, μ)
end


function _backproject(sinogram::AbstractArray{T, 3}, angles_T::AbstractVector, geometry::Union{RadonParallelCircle, RadonFlexibleCircle}, μ) where T
    @assert size(sinogram, 2) == length(angles_T) "size of angles does not match sinogram size"
    @assert size(sinogram, 1) == size(geometry.in_height, 1)
    if geometry isa RadonFlexibleCircle
        @assert size(sinogram, 1) == size(geometry.out_height, 1)
    end

    backend = KernelAbstractions.get_backend(sinogram)
 
    # angles_T might be a normal vector instead of Cuvector. fix it. 
    angles = similar(sinogram, (size(angles_T, 1),))
    angles .= typeof(angles)(angles_T) 
    
    weights = similar(sinogram, (size(geometry.in_height, 1),))
    weights .= typeof(weights)(geometry.weights) 

    in_height = similar(sinogram, (size(geometry.in_height, 1),))
    in_height .= typeof(in_height)(geometry.in_height) 

    if geometry isa RadonFlexibleCircle
        out_height = similar(sinogram, (size(geometry.out_height, 1),))
        out_height .= typeof(out_height)(geometry.out_height) 
    else
        out_height = in_height
    end

    # geometry can be very densely sampled, hence the sinogram depends on geometry size
    N = geometry.N
    # we only propagate inside this in circle
    # convert radius to correct float type, very important for performance!
    radius = T((N - 1) ÷ 2)
    # the midpoint of the array
    # convert to good type
    mid = T(N ÷ 2 + 1 +  1 // 2)
    N_angles = length(angles)

    img = similar(sinogram, (N, N,  size(sinogram, 3)))
    fill!(img, 0)

    # create an absorption function, maps just to 1 in case isnothing(μ)
    absorb_f = make_absorption_f(μ, T)

    # of the kernel goes
    kernel! = backproject_kernel2!(backend)
    kernel!(img, sinogram, weights, in_height, out_height, angles, mid, radius, absorb_f,
            ndrange=(size(sinogram, 1), N_angles, size(img, 3)))
    KernelAbstractions.synchronize(backend)    
    return img 
end

@kernel function backproject_kernel2!(img::AbstractArray{T}, sinogram::AbstractArray{T}, weights, in_height,
                                 out_height, angles, mid, radius, absorb_f) where {T}
    i, iangle, i_z = @index(Global, NTuple)
    
    @inbounds sinα, cosα = sincos(angles[iangle])

    @inbounds ybegin = T(in_height[i])
    @inbounds yend = T(out_height[i])
    
     # weights
    @inbounds weight = weights[i]

    # map the detector positions on a circle
    # also rotate according to the angle
    x_dist_rot, y_dist_rot, x_dist_end_rot, y_dist_end_rot = 
        next_ray_on_circle(ybegin, yend, mid, radius, sinα, cosα)
    

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
        inside_circ(xnew, ynew, mid, radius + T(0.5)) && 
         (sx == sxold && sy == syold) || break
        
        # switch of i and j intentional to keep it consistent with existing code
        icell, jcell = find_cell(xnew, ynew, xold, yold)

        # calculate intersection distance
        distance = sqrt((xnew - xold)^2 + (ynew - yold) ^2)
        # add value to ray, potentially attenuate by attenuated exp factor
        @inbounds Atomix.@atomic img[icell, jcell, i_z] += weight * distance * 
                sinogram[i, iangle, i_z] * absorb_f(xnew, ynew, x_dist_rot, y_dist_rot)
        xold, yold = xnew, ynew
    end 
end


 # define adjoint rules
function ChainRulesCore.rrule(::typeof(_backproject), array::AbstractArray, angles,
                              geometry, μ) 
    res = _backproject(array, angles, geometry, μ)
    function pb_backproject(ȳ)
        ad = _radon(unthunk(ȳ), angles, geometry, μ)
        return NoTangent(), ad, NoTangent(), NoTangent(), NoTangent()
    end
    return res, pb_backproject 
end
