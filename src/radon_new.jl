export radon2


function make_absorption_f(μ, ::Type{T}) where T
    if isnothing(μ)
        @inline (x, y, x_start, y_start) -> one(T)
    else
        @inline (x, y, x_start, y_start) -> exp(-T(μ) * sqrt((x - x_start)^2 + (y - y_start)^2))
    end
end


# handle 2D
function radon(img::AbstractArray{T, 2}, angles::AbstractArray{T2, 1};
        geometry=RadonParallelCircle(-(size(img,1)-1)÷2:(size(img,1)-1)÷2), μ=nothing) where {T, T2}
    view(radon(reshape(img, (size(img)..., 1)), angles; geometry, μ), :, :, 1)
end


"""
    radon(img, θs; <kwargs>)

Calculates the parallel Radon transform of the AbstractArray `I`.
The first two dimensions are y and x. The third dimension is z, the rotational axis.
`size(I, 1)` and `size(I, 2)` have to be equally sized.
The Radon transform is rotated around the pixel `size(I, 1) ÷ 2 + 1`, so there
is always an integer center pixel!
Works either with a `AbstractArray{T, 3}` or `AbstractArray{T, 2}`.

`θs` is a vector or range storing the angles in radians.


In principle, all backends of KernelAbstractions.jl should work but are not tested. CUDA and CPU arrays are actively tested.


# Keywords
## `μ=nothing` - Exponential IRadon Transform
If `μ != nothing`, then the rays are attenuated with `exp(-μ * dist)` where `dist` 
is the distance to the circular boundary of the field of view.
`μ` is in units of pixel length. So `μ=1` corresponds to an attenuation of `exp(-1)` if propagated through one pixel.

## `geometry=geometry=RadonParallelCircle(-(size(img,1)-1)÷2:(size(img,1)-1)÷2)`
This corresponds to a parallel Radon transform. 
See `?RadonGeometries` for a full list of geometries. There is also the very flexible `RadonFlexibleCircle`.


Please note: the implementation is not quite optimized for cache efficiency and 
it is a very naive algorithm. But still, it is fast!

See also [`iradon`](@ref).

# Example
The reason the sinogram has the value `1.41421` for the diagonal ray `π/4` is,
that such a diagonal travels a longer distance through the pixel.
```jldoctest
julia> arr = zeros((4,4)); arr[3,3] = 1;

julia> radon(arr, [0, π/4, π/2])
3×3 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0      0.0
 1.0  1.41421  1.0
 0.0  0.0      0.0
```
"""
function radon2(img::AbstractArray{T, 3}, angles_T::AbstractArray{T, 1};
        geometry=RadonParallelCircle(-(size(img,1)-1)÷2:(size(img,1)-1)÷2),
        μ=nothing) where T
    @assert size(img, 1) == size(img, 2) "Arrays has to be quadratically shaped"
    backend = KernelAbstractions.get_backend(img)
 
    angles = angles_T
    angles = similar(img, (size(angles_T, 1),))
    angles .= typeof(angles)(angles_T) 

    N = length(geometry.in_height)#size(img, 1) - 1
    # we only propagate inside this in circle
    radius = T((size(img, 1) - 1) ÷ 2)
    # the midpoint of the array
    mid = T(size(img, 1) ÷ 2 + 1 +  1 // 2)
    N_angles = length(angles)

    sinogram = similar(img, (N, N_angles, size(img, 3)))
    fill!(sinogram, 0)

    # create an absorption function, maps just to 1 in case isnothing(μ)
    absorb_f = make_absorption_f(μ, T)
    kernel! = radon_kernel2!(backend)

    kernel!(sinogram::AbstractArray{T}, img, geometry.in_height, angles, mid, radius, absorb_f,
    		ndrange=(N, N_angles, size(img, 3)))

    KernelAbstractions.synchronize(backend)    
    return sinogram
end

@inline inside_circ(ii, jj, mid, radius) = (ii - mid)^2 + (jj - mid)^2 ≤ radius ^2 

@kernel function radon_kernel2!(sinogram::AbstractArray{T}, img, in_height, angles, mid, radius, absorb_f, ::Type{IT}) where {T, IT}
    i, iangle, i_z = @index(Global, NTuple)
    
    sinα, cosα = sincos(angles[iangle])

    xend, yend = T(size(img, 2)), T(in_height[i])

    # map the detector positions on a circle
    x_dist_rot, y_dist_rot, x_dist_end_rot, y_dist_end_rot = 
        next_ray_on_circle(yend, yend, mid, radius, sinα, cosα)

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
        xnew, ynew, sx, sy = next_cell_intersection(xnew, ynew, xend, yend)
        inside_circ(xnew, ynew, mid, radius) && 
         (sx == sxold && sy == syold) || break
        
        # switch of i and j intentional to keep it consistent with existing code
        icell, jcell = find_cell(xnew, ynew, xold, yold)

        distance = sqrt((xnew - xold)^2 + (ynew - yold) ^2)
        @inbounds tmp += distance * img[icell, jcell, i_z] * absorb_f(xnew, ynew, x_dist_rot, y_dist_rot)
        xold, yold = xnew, ynew
    end 
    @inbounds sinogram[i, iangle, i_z] = tmp
end