export radon_old
export iradon_old


# handle 2D
iradon_old(sinogram::AbstractArray{T, 2}, angles::AbstractArray{T2, 1}, μ=nothing;
       ray_startpoints=nothing, ray_endpoints=nothing) where {T, T2} = begin
    angles_T = similar(sinogram, (size(angles, 1),))
    angles_T .= angles 
    view(iradon_old(reshape(sinogram, (size(sinogram)..., 1)), angles_T, μ; ray_startpoints, ray_endpoints), :, :, 1)
end


iradon_old(sinogram::AbstractArray{T, 3}, angles::AbstractArray{T2, 1}, μ=nothing;
       ray_startpoints=nothing, ray_endpoints=nothing) where {T, T2} = begin
    angles_T = similar(sinogram, (size(angles, 1),))
    angles_T .= angles 
    iradon_old(sinogram, angles_T, μ; ray_startpoints, ray_endpoints)
end
"""
    iradon_old(sinogram, θs, μ=nothing; <kwargs>)

Calculates the parallel inverse radon_old transform of the `sinogram`.
The first two dimensions are y and x. The third dimension is z, the rotational axis.
Works either with a `AbstractArray{T, 3}` or `AbstractArray{T, 2}`.
`size(sinogram, 1)` has to be an odd number. And `size(sinogram, 2)` has to be equal to
`length(angles)`.
The inverse radon_old transform is rotated around the pixel `size(sinogram, 1) ÷ 2`, so there
is always a real center pixel!

`θs` is a vector or range storing the angles in radians.

# Exponential Iradon_old Transform
If `μ != nothing`, then the rays are attenuated with `exp(-μ * dist)` where `dist` 
is the distance to the circular boundary of the field of view.
`μ` is in units of pixel length. So `μ=1` corresponds to an attenuation of `exp(-1)` if propagated through one pixel.


# Keywords
* `ray_startpoints = -size(img, 1) ÷ 2 - 1: size(img, 1) ÷ 2 - 1`: this indicates the starting position of a ray at the entrance on the array 
* `ray_endpoints = -size(img, 1) ÷ 2 - 1: size(img, 1) ÷ 2 - 1`: this indicates the end position of a ray at the exit of the array 

Even though we trace everything inside the circle only, start and endpoint of the ray are all the first and last index position in the array.
So it's not the y position on the circle but rather the y position at the smallest and greatest x value.


In principle, all backends of KernelAbstractions.jl should work but are not tested. CUDA and CPU arrays are actively tested.

See also [`radon_old`](@ref).

# Examples
```jldoctest
julia> arr = zeros((5,2)); arr[2,:] .= 1; arr[4, :] .= 1
2-element view(::Matrix{Float64}, 4, :) with eltype Float64:
 1.0
 1.0

julia> iradon_old(arr, [0, π/2])
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0  0.0        0.0  0.0        0.0
 0.0  0.0  0.1        0.0  0.1        0.0
 0.0  0.1  0.2        0.1  0.2        0.0232051
 0.0  0.0  0.1        0.0  0.1        0.0
 0.0  0.1  0.2        0.1  0.2        0.0232051
 0.0  0.0  0.0232051  0.0  0.0232051  0.0

julia> iradon_old(arr, [0, π/2], 1) # exponential
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0         0.0         0.0        0.0         0.0
 0.0  0.0         0.00145226  0.0        0.00145226  0.0
 0.0  0.00145226  0.00789529  0.0107308  0.033117    0.0183994
 0.0  0.0         0.0107308   0.0        0.0107308   0.0
 0.0  0.00145226  0.033117    0.0107308  0.0583388   0.0183994
 0.0  0.0         0.0183994   0.0        0.0183994   0.0
```
"""
function iradon_old(sinogram::AbstractArray{T, 3}, angles::AbstractArray{T, 1}, μ=nothing;
                ray_startpoints=nothing, ray_endpoints=nothing) where T
    @assert isodd(size(sinogram, 1)) && size(sinogram, 2) == length(angles)
    backend=KernelAbstractions.get_backend(sinogram)
    absorption_f = let μ=μ
        if isnothing(μ)
            (x, y, x_start, y_start) -> one(T)
        else
            (x, y, x_start, y_start) -> exp(-T(μ) * sqrt((x - x_start)^2 + (y - y_start)^2))
        end
    end
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
    
    if isnothing(ray_endpoints)
        # the y_dists samplings, in principle we can add this as function parameter
        y_dists_end = similar(img, (size(img, 1) - 1, ))
        y_dists_end .= -radius:radius
    else
        y_dists_end = similar(img, (size(img, 1) - 1, ))
        y_dists_end .= ray_endpoints 
    end
        
    if isnothing(ray_startpoints)
        y_dists = similar(img, (size(img, 1) - 1, ))
        y_dists .= -radius:radius
    else
        y_dists = similar(img, (size(img, 1) - 1, ))
        y_dists .= ray_startpoints 
    end

    #@show typeof(sinogram), typeof(img), typeof(y_dists), typeof(angles)
    kernel! = iradon_old_kernel!(backend)
    kernel!(sinogram::AbstractArray{T}, img, y_dists, y_dists_end, angles, mid, radius,
            absorption_f,
    		ndrange=(N, N_angles, size(img, 3)))
    KernelAbstractions.synchronize(backend)    
    return img
end

"""
    iradon_old_kernel!(sinogram, img,
                  y_dists, angles, mid, radius,
                  absorption_f)
"""
@kernel function iradon_old_kernel!(sinogram::AbstractArray{T},
			img, y_dists, y_dists_end, angles, mid, radius,
            absorption_f) where T
    # r is the index of the angles
    # k is the index of the detector spatial coordinate
    # i_z is the index for the z dimension
    k, r, i_z = @index(Global, NTuple)
    
    angle = angles[r]
    sinα, cosα = sincos(angle)
    
    # x0, y0, x1, y1 beginning and end point of each ray
    a, b, c, d = next_ray_on_circle(y_dists[k], y_dists_end[k], mid, radius, sinα, cosα)
    a0, b0, c0, d0 = a, b, c, d 
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
        a, b = find_next_intersection(a,b,c,d)
        l += 1
        
        # find the cell it is cutting through
        cell_i, cell_j = find_cell((a_old, b_old),
        						   (a, b))
        
        # distance travelled through that cell
        distance = sqrt((a_old - a) ^2 +
        				(b_old - b) ^2)
        # cell value times distance travelled through
        @inbounds Atomix.@atomic img[cell_i, cell_j, i_z] += 
            distance * sinogram[k, r, i_z] * absorption_f(a, b, a0, b0)
    end
end


 # define adjoint rules
function ChainRulesCore.rrule(::typeof(iradon_old), sinogram::AbstractArray, angles, μ=nothing; ray_endpoints=nothing) 
    res = iradon_old(sinogram, angles, μ; ray_endpoints)
    function pb_iradon_old(ȳ)
        ad = radon_old(unthunk(ȳ), angles, μ; ray_endpoints)
        return NoTangent(), ad, NoTangent(), NoTangent()
    end
    return res, pb_iradon_old 
end


radon_old(img::AbstractArray{T, 3}, angles::AbstractArray{T2, 1}, μ=nothing;
      ray_startpoints=nothing, ray_endpoints=nothing) where {T, T2} = begin
    angles_T = similar(img, (size(angles, 1),))
    angles_T .= angles 
    radon_old(img, angles_T, μ; ray_startpoints, ray_endpoints)
end

# handle 2D
radon_old(img::AbstractArray{T, 2}, angles::AbstractArray{T2, 1}, μ=nothing;
      ray_startpoints=nothing, ray_endpoints=nothing) where {T, T2} = begin
    angles_T = similar(img, (size(angles, 1),))
    angles_T .= angles 
    view(radon_old(reshape(img, (size(img)..., 1)), angles_T, μ; 
               ray_startpoints, ray_endpoints), :, :, 1)
end
"""
    radon_old(I, θs, μ=nothing; <kwargs>)

Calculates the parallel radon_old transform of the AbstractArray `I`.
The first two dimensions are y and x. The third dimension is z, the rotational axis.
`size(I, 1)` and `size(I, 2)` have to be equal and a even number. 
The radon_old transform is rotated around the pixel `size(I, 1) ÷ 2 + 1`, so there
is always a real center pixel!
Works either with a `AbstractArray{T, 3}` or `AbstractArray{T, 2}`.

`θs` is a vector or range storing the angles in radians.

# Exponential Iradon_old Transform
If `μ != nothing`, then the rays are attenuated with `exp(-μ * dist)` where `dist` 
is the distance to the circular boundary of the field of view.
`μ` is in units of pixel length. So `μ=1` corresponds to an attenuation of `exp(-1)` if propagated through one pixel.

In principle, all backends of KernelAbstractions.jl should work but are not tested. CUDA and CPU arrays are actively tested.


# Keywords
* `ray_startpoints = -size(img, 1) ÷ 2 - 1: size(img, 1) ÷ 2 - 1`: this indicates the starting position of a ray at the entrance on the array 
* `ray_endpoints = -size(img, 1) ÷ 2 - 1: size(img, 1) ÷ 2 - 1`: this indicates the end position of a ray at the exit of the array 

Even though we trace everything inside the circle only, start and endpoint of the ray are all the first and last index position in the array.
So it's not the y position on the circle but rather the y position at the smallest and greatest x value.



Please note: the implementation is not quite optimized for cache efficiency and 
it is a very naive algorithm. But still, it is fast!

See also [`iradon_old`](@ref).

# Example
The reason the sinogram has the value `1.41421` for the diagonal ray `π/4` is,
that such a diagonal travels a longer distance through the pixel.
```jldoctest
julia> arr = zeros((4,4)); arr[3,3] = 1;

julia> radon_old(arr, [0, π/4, π/2])
3×3 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0      0.0
 1.0  1.41421  1.0
 0.0  0.0      0.0
```
"""
function radon_old(img::AbstractArray{T, 3}, angles::AbstractArray{T, 1}, 
               μ=nothing; ray_startpoints=nothing, ray_endpoints=nothing) where T
    @assert iseven(size(img, 1)) && iseven(size(img, 2)) && size(img, 1) == size(img, 2) "Arrays has to be quadratic and even sized shape"
    
    backend = KernelAbstractions.get_backend(img)

    absorption_f = let μ=μ
        if isnothing(μ)
            (x, y, x_start, y_start) -> one(T)
        else
            (x, y, x_start, y_start) -> exp(-T(μ) * sqrt((x - x_start)^2 + (y - y_start)^2))
        end
    end

    # this is the actual size we are using for calculation
    N = size(img, 1) - 1
    N_angles = size(angles, 1)
    
    # the only significant allocation
    sinogram = similar(img, (N, N_angles, size(img, 3)))
    fill!(sinogram, 0)
    
    # radius of the cylinder we are projecting through
    radius = size(img, 1) ÷ 2 - 1
    # mid point, it is actually N ÷ 2 + 1
    # but because of how adress the indices, we need 1.5 instead of +1
    mid = size(img, 1) ÷ 2 + T(1.5)


    if isnothing(ray_endpoints)
        # the y_dists samplings, in principle we can add this as function parameter
        y_dists_end = similar(img, (size(img, 1) - 1, ))
        y_dists_end .= -radius:radius
    else
        y_dists_end = similar(img, (size(img, 1) - 1, ))
        y_dists_end .= ray_endpoints 
    end
        
    if isnothing(ray_startpoints)
        y_dists = similar(img, (size(img, 1) - 1, ))
        y_dists .= -radius:radius
    else
        y_dists = similar(img, (size(img, 1) - 1, ))
        y_dists .= ray_startpoints 
    end


    # the y_dists samplings, in principle we can add this as function parameter 
    y_dists = similar(img, (size(img, 1) - 1, ))
    y_dists .= -radius:radius
    
    #@show typeof(sinogram), typeof(img), typeof(y_dists), typeof(angles)
    kernel! = radon_old_kernel!(backend)
    kernel!(sinogram::AbstractArray{T}, img, y_dists, y_dists_end, angles, mid, radius,
            absorption_f,
    		ndrange=(N, N_angles, size(img, 3)))
    KernelAbstractions.synchronize(backend)    
    return sinogram
end

"""
    radon_old_kernel!(sinogram, img,
                  y_dists, angles, mid, radius,
                  absorption_f)

"""
@kernel function radon_old_kernel!(sinogram::AbstractArray{T},
			img, y_dists, y_dists_end, angles, mid, radius, absorption_f) where T
    # r is the index of the angles
    # k is the index of the detector spatial coordinate
    # i_z is the index for the z dimension
    k, r, i_z = @index(Global, NTuple)
    
    angle = angles[r]
    sinα, cosα = sincos(angle)
    
    # x0, y0, x1, y1 beginning and end point of each ray
    a, b, c, d = next_ray_on_circle(y_dists[k], y_dists_end[k], mid, radius, sinα, cosα)
    a0, b0, c0, d0 = a, b, c, d 
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
        a, b = find_next_intersection(a,b,c,d)
        l += 1
        
        # find the cell it is cutting through
        cell_i, cell_j = find_cell((a_old, b_old),
        						   (a, b))
        
        # distance travelled through that cell
        distance = sqrt((a_old - a) ^2 +
        				(b_old - b) ^2)
        # cell value times distance travelled through
        @inbounds tmp += distance * img[cell_i, cell_j, i_z] * absorption_f(a, b, a0, b0)
    end
    @inbounds sinogram[k, r, i_z] = tmp
end


 # define adjoint rules
function ChainRulesCore.rrule(::typeof(radon_old), array::AbstractArray, angles, μ=nothing; 
                              ray_startpoints=nothing, ray_endpoints=nothing) 
    res = radon_old(array, angles, μ; ray_startpoints, ray_endpoints)
    function pb_radon_old(ȳ)
        ad = iradon_old(unthunk(ȳ), angles, μ; ray_startpoints, ray_endpoints)
        return NoTangent(), ad, NoTangent(), NoTangent()
    end
    return res, pb_radon_old 
end
