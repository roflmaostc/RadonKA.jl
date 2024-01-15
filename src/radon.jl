export radon



radon(img::AbstractArray{T, 3}, angles::AbstractArray{T2, 1}, μ=nothing; ray_endpoints=nothing) where {T, T2} =
    radon(img, T.(angles), μ; ray_endpoints)

# handle 2D
radon(img::AbstractArray{T, 2}, angles::AbstractArray{T2, 1}, μ=nothing; ray_endpoints=nothing) where {T, T2} =
    view(radon(reshape(img, (size(img)..., 1)), T.(angles), μ; ray_endpoints), :, :, 1)

"""
    radon(I, θs, μ=nothing)

Calculates the parallel Radon transform of the AbstractArray `I`.
The first two dimensions are y and x. The third dimension is z, the rotational axis.
`size(I, 1)` and `size(I, 2)` have to be equal and a even number. 
The Radon transform is rotated around the pixel `size(I, 1) ÷ 2 + 1`, so there
is always a real center pixel!
Works either with a `AbstractArray{T, 3}` or `AbstractArray{T, 2}`.

`θs` is a vector or range storing the angles in radians.

# Exponential IRadon Transform
If `μ != nothing`, then the rays are attenuated with `exp(-μ * dist)` where `dist` 
is the distance to the circular boundary of the field of view.
`μ` is in units of pixel length. So `μ=1` corresponds to an attenuation of `exp(-1)` if propagated through one pixel.

In principle, all backends of KernelAbstractions.jl should work but are not tested. CUDA and CPU arrays are actively tested.


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
function radon(img::AbstractArray{T, 3}, angles::AbstractArray{T, 1}, 
               μ=nothing; ray_endpoints=nothing) where T
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



    # the y_dists samplings, in principle we can add this as function parameter 
    y_dists = similar(img, (size(img, 1) - 1, ))
    y_dists .= -radius:radius
    
    #@show typeof(sinogram), typeof(img), typeof(y_dists), typeof(angles)
    kernel! = radon_kernel!(backend)
    kernel!(sinogram::AbstractArray{T}, img, y_dists, y_dists_end, angles, mid, radius,
            absorption_f,
    		ndrange=(N, N_angles, size(img, 3)))
    KernelAbstractions.synchronize(backend)    
    return sinogram
end

"""
    radon_kernel!(sinogram, img,
                  y_dists, angles, mid, radius,
                  absorption_f)

"""
@kernel function radon_kernel!(sinogram::AbstractArray{T},
			img, y_dists, y_dists_end, angles, mid, radius, absorption_f) where T
    # r is the index of the angles
    # k is the index of the detector spatial coordinate
    # i_z is the index for the z dimension
    k, r, i_z = @index(Global, NTuple)
    
    angle = angles[r]
    sinα, cosα = sincos(angle)
    
    # x0, y0, x1, y1 beginning and end point of each ray
    a, b, c, d = next_ray_on_circle(img, angle, y_dists[k], y_dists_end[k], mid, radius, sinα, cosα)
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
function ChainRulesCore.rrule(::typeof(radon), array::AbstractArray, angles, μ=nothing) 
    res = radon(array, angles, μ)
    function pb_radon(ȳ)
        ad = iradon(unthunk(ȳ), angles, μ)
        return NoTangent(), ad, NoTangent(), NoTangent()
    end
    return res, pb_radon 
end
