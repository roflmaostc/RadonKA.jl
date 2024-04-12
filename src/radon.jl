export radon


_get_μ_array(μ::Nothing) = nothing 
_get_μ_array(μ::Number) = nothing
_get_μ_array(μ::AbstractArray) = μ




function make_absorption_f(μ, ::Type{T}) where T
    @inline (i, j, iz, x, y, x_start, y_start) -> one(T)
end

function make_absorption_f(μ::Number, ::Type{T}) where T
    @inline (i, j, iz, x, y, x_start, y_start) -> exp(-T(μ) * sqrt((x - x_start)^2 + (y - y_start)^2))
end



# handle 2D
function radon(img::AbstractArray{T, 2}, angles::AbstractArray{T2, 1};
        geometry=RadonParallelCircle(size(img, 1), -(size(img,1)-1)÷2:(size(img,1)-1)÷2), μ=nothing) where {T, T2}
    view(radon(reshape(img, (size(img)..., 1)), angles; geometry, μ), :, :, 1)
end


"""
    radon(I, θs; <kwargs>)

Calculates the parallel Radon transform of the AbstractArray `I`.
Intuitively, the `radon` sums array entries  of `I` along ray paths.

The first two dimensions are y and x. The third dimension is z, the rotational axis.
`size(I, 1)` and `size(I, 2)` have to be equal.
The Radon transform is rotated around the pixel `size(I, 1) ÷ 2 + 1`, so there
is always an integer center pixel!
Works either with a `AbstractArray{T, 3}` or `AbstractArray{T, 2}`.

`θs` is a vector or range storing the angles in radians.

In principle, all backends of KernelAbstractions.jl should work but are not tested. CUDA and CPU arrays are actively tested.
Both `radon` and [`backproject`](@ref) are differentiable with respect to `I`.

# Keywords
## `μ=nothing` - Attenuated Radon Transform
If `μ != nothing`, then the rays are attenuated with `exp(-μ * dist)` where `dist` 
is the distance to the circular boundary of the field of view.
`μ` is in units of pixel length. So `μ=1` corresponds to an attenuation of `exp(-1)` if propagated through one pixel.
If `isnothing(μ)`, then the rays are not attenuated.

## `geometry = RadonParallelCircle(-(size(img,1)-1)÷2:(size(img,1)-1)÷2)`
This corresponds to a parallel Radon transform. 
See `?RadonGeometries` for a full list of geometries. There is also the very flexible `RadonFlexibleCircle`.



See also [`backproject`](@ref).

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

 ## Choose different detector
```jldoctest
julia> array = ones((6,6))
6×6 Matrix{Float64}:
 1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0

julia> radon(array, [0])
5×1 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 1.0
 3.7320508075688767
 5.0
 3.7320508075688767
 1.0

julia> radon(array, [0], geometry=RadonParallelCircle(6, -2:2))
5×1 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 1.0
 3.7320508075688767
 5.0
 3.7320508075688767
 1.0

julia> radon(array, [0], geometry=RadonParallelCircle(6, -2:2:2))
3×1 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 1.0
 5.0
 1.0
```

 ## Apply some weights on the rays
```jldoctest
julia> array = ones((6,6));

julia> radon(array, [0], geometry=RadonParallelCircle(6, -2:2, [2,1,0,1,2]))
5×1 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 2.0
 3.7320508075688767
 0.0
 3.7320508075688767
 2.0
```
"""
function radon(img::AbstractArray{T, 3}, angles_T::AbstractVector;
        geometry=RadonParallelCircle(size(img, 1), -(size(img,1)-1)÷2:(size(img,1)-1)÷2),
        μ=nothing) where  {T}
    return _radon(img::AbstractArray{T, 3}, angles_T::AbstractVector, geometry, μ)
end

"""

internal method which handles the different dispatches.
"""
function _radon(img::AbstractArray{T, 3}, angles_T::AbstractVector,
        geometry::Union{RadonParallelCircle, RadonFlexibleCircle}, μ) where  {T}

    @assert size(img, 1) == size(img, 2) "Arrays has to be quadratically shaped"
    @assert size(img, 1) == geometry.N
    if geometry isa RadonFlexibleCircle
        @assert size(geometry.in_height, 1) == size(geometry.out_height, 1)
    end
    backend = KernelAbstractions.get_backend(img)
 
    # angles_T might be a normal vector instead of Cuvector. fix it. 
    angles = similar(img, (size(angles_T, 1),))
    angles .= typeof(angles)(angles_T) 
    
    # in_height could be a range or a CPU vector, so convert it to an array
    in_height = similar(img, (size(geometry.in_height, 1),))
    in_height .= typeof(in_height)(geometry.in_height) 
    

    # wieghts could be a range or a CPU vector, so convert it to an array
    weights = similar(img, (size(geometry.in_height, 1),))
    weights .= typeof(weights)(geometry.weights) 

    if geometry isa RadonFlexibleCircle
        out_height = similar(img, (size(geometry.out_height, 1),))
        out_height .= typeof(out_height)(geometry.out_height) 
    else
        out_height = in_height
    end

    # geometry can be very densely sampled, hence the sinogram depends on geometry size
    N_sinogram = length(geometry.in_height)
    # we only propagate inside this in circle
    # convert radius to correct float type, very important for performance!
    radius = T((size(img, 1) - 1) ÷ 2)
    # the midpoint of the array
    # convert to correct type
    mid = T(size(img, 1) ÷ 2 + 1 +  1 // 2)
    N_angles = length(angles)

    # final sinogram
    sinogram = similar(img, (N_sinogram, N_angles, size(img, 3)))
    fill!(sinogram, 0)

    # create an absorption function, maps just to 1 in case isnothing(μ)
    absorb_f = make_absorption_f(μ, T)
    
    μ_array = _get_μ_array(μ)

    # of the kernel goes
    kernel! = radon_kernel!(backend)
    kernel!(sinogram::AbstractArray{T}, img, weights, in_height, 
            out_height, angles, mid, radius, absorb_f, μ_array,
            ndrange=size(sinogram))
    KernelAbstractions.synchronize(backend)    
    return sinogram::typeof(img)
end

@inline inside_circ(ii, jj, mid, radius) = (ii - mid)^2 + (jj - mid)^2 ≤ radius ^2 

@kernel function radon_kernel!(sinogram::AbstractArray{T}, @Const(img), 
                                @Const(weights), @Const(in_height), 
                                @Const(out_height), @Const(angles), mid,
                                radius, absorb_f, μ_array) where {T}
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
    ray_intensity = one(T)
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

        @inbounds value_in = img[icell, jcell, i_z]
        value, ray_intensity = calc_deposit_value(value_in,  weight, sinogram, i, 
                                                  iangle, i_z, icell, jcell, xnew, ynew,
                                                  x_dist_rot, y_dist_rot, absorb_f,
                                                  xold, yold, ray_intensity, μ_array)
        # add value to ray, potentially attenuate by attenuated exp factor
        @inbounds tmp += value 
        xold, yold = xnew, ynew
    end 
    @inbounds sinogram[i, iangle, i_z] = tmp
end

# we have two different cases
# - one μ_array which is nothing in case the absorption is constant, then we use
#   the simple absorption function absorb_f
# - one μ_array which is not nothing, then we use the absorption array
#
# the function returns the value and the new ray_intensity
# so we can reuse the whole code but whether if μ is an array or not, we do 
# two different things
@inline function calc_deposit_value(value_in, weight, sinogram, i, iangle, i_z, icell, jcell, xnew, ynew, x_dist_rot, y_dist_rot, absorb_f, xold, yold, ray_intensity, μ_array::Nothing)
    # calculate intersection distance
    distance = sqrt((xnew - xold)^2 + (ynew - yold) ^2)
    value = weight * distance * value_in * absorb_f(icell, jcell, i_z, xnew, ynew, x_dist_rot, y_dist_rot)
    return value, ray_intensity
end

@inline function calc_deposit_value(value_in, weight, sinogram, i, iangle, i_z, icell, jcell, xnew, ynew, x_dist_rot, y_dist_rot, absorb_f, xold, yold, ray_intensity, μ_array)
    distance = sqrt((xnew - xold)^2 + (ynew - yold) ^2)
    
    @inbounds ray_intensity -= ray_intensity * (distance) * μ_array[icell, jcell, i_z] / 2
    ray_intensity = max(ray_intensity, zero(typeof(ray_intensity)))
    value = value_in * weight * distance * ray_intensity 
    @inbounds ray_intensity -= ray_intensity * (distance) * μ_array[icell, jcell, i_z] / 2
    ray_intensity = max(ray_intensity, zero(typeof(ray_intensity)))
    return value, ray_intensity
end




 # define adjoint rules
function ChainRulesCore.rrule(::typeof(_radon), array::AbstractArray, angles,
                              geometry, μ) 
    res = _radon(array, angles, geometry, μ)
    function pb_radon(ȳ)
        ad = _backproject(unthunk(ȳ), angles, geometry, μ)
        return NoTangent(), ad, NoTangent(), NoTangent(), NoTangent()
    end
    return res, pb_radon 
end
