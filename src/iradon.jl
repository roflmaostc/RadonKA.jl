export iradon, filtered_backprojection

"""
    filtered_backprojection(sinogram, θs)

Calculates the simple Filtered Backprojection in CT with applying a ramp filter
in Fourier space.

"""
function filtered_backprojection(sinogram::AbstractArray{T}, θs::AbstractVector, μ=nothing; 
                                 backend=KernelAbstractions.get_backend(sinogram)) where T
    filter = similar(sinogram, (size(sinogram, 1),))
    filter .= rr(T, (size(sinogram, 1), )) 

    p = plan_fft(sinogram, (1,))
    sinogram = real(inv(p) * (p * sinogram .* ifftshift(filter)))
    return iradon(sinogram, θs, μ, backend=backend)
end

# handle 2D
iradon(sinogram::AbstractArray{T, 2}, angles::AbstractArray{T2, 1}, μ=nothing; 
       backend=KernelAbstractions.get_backend(sinogram)) where {T, T2} =
    view(iradon(reshape(sinogram, (size(sinogram)..., 1)), angles, μ; backend), :, :, 1)

iradon(sinogram::AbstractArray{T, 3}, angles::AbstractArray{T2, 1}, μ=nothing; 
       backend=KernelAbstractions.get_backend(sinogram)) where {T, T2} =
    iradon(sinogram, T.(angles), μ; backend)

"""
    iradon(sinogram, θs, μ=nothing)

Calculates the parallel inverse Radon transform of the `sinogram`.
The first two dimensions are y and x. The third dimension is z, the rotational axis.
Works either with a `AbstractArray{T, 3}` or `AbstractArray{T, 2}`.
`size(sinogram, 1)` has to be an odd number. And `size(sinogram, 2)` has to be equal to
`length(angles)`.
The inverse Radon transform is rotated around the pixel `size(sinogram, 1) ÷ 2`, so there
is always a real center pixel!

`θs` is a vector or range storing the angles in radians.

# Exponential IRadon Transform
If `μ != nothing`, then the rays are attenuated with `exp(-μ * dist)` where `dist` 
is the distance to the circular boundary of the field of view.
`μ` is in units of pixel length. So `μ=1` corresponds to an attenuation of `exp(-1)` if propagated through one pixel.

# Keywords 
* `backend` can be either `CPU()` for multithreaded CPU execution or `CUDABackend()` for CUDA execution. In principle, all backends of KernelAbstractions.jl should work but are not tested.

See also [`radon`](@ref).


# Examples
```jldoctest
julia> arr = zeros((5,2)); arr[2,:] .= 1; arr[4, :] .= 1
2-element view(::Matrix{Float64}, 4, :) with eltype Float64:
 1.0
 1.0

julia> iradon(arr, [0, π/2])
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0  0.0        0.0  0.0        0.0
 0.0  0.0  0.1        0.0  0.1        0.0
 0.0  0.1  0.2        0.1  0.2        0.0232051
 0.0  0.0  0.1        0.0  0.1        0.0
 0.0  0.1  0.2        0.1  0.2        0.0232051
 0.0  0.0  0.0232051  0.0  0.0232051  0.0

julia> iradon(arr, [0, π/2], 1) # exponential
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0         0.0         0.0        0.0         0.0
 0.0  0.0         0.00145226  0.0        0.00145226  0.0
 0.0  0.00145226  0.00789529  0.0107308  0.033117    0.0183994
 0.0  0.0         0.0107308   0.0        0.0107308   0.0
 0.0  0.00145226  0.033117    0.0107308  0.0583388   0.0183994
 0.0  0.0         0.0183994   0.0        0.0183994   0.0
```
"""
function iradon(sinogram::AbstractArray{T, 3}, angles::AbstractArray{T, 1}, μ=nothing;
                backend=KernelAbstractions.get_backend(sinogram)) where T
    @assert isodd(size(sinogram, 1)) && size(sinogram, 2) == length(angles)

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
    # the y_dists samplings, in principle we can add this as function parameter
    y_dists = similar(img, (size(img, 1) - 1, ))
    y_dists .= -radius:radius
    
    #@show typeof(sinogram), typeof(img), typeof(y_dists), typeof(angles)
    kernel! = iradon_kernel!(backend)
    kernel!(sinogram::AbstractArray{T}, img, y_dists, angles, mid, radius,
            absorption_f,
    		ndrange=(N, N_angles, size(img, 3)))
    
    img ./= N .* length(angles)
    return img
end

"""
    iradon_kernel!(sinogram, img,
                  y_dists, angles, mid, radius,
                  absorption_f)
"""
@kernel function iradon_kernel!(sinogram::AbstractArray{T},
			img, y_dists, angles, mid, radius,
            absorption_f) where T
    # r is the index of the angles
    # k is the index of the detector spatial coordinate
    # i_z is the index for the z dimension
    k, r, i_z = @index(Global, NTuple)
    
    angle = angles[r]
    sinα, cosα = sincos(angle)
    
    # x0, y0, x1, y1 beginning and end point of each ray
    a, b, c, d = next_ray_on_circle(img, angle, y_dists[k], mid, radius, sinα, cosα)
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
        @inline a, b = find_next_intersection(a,b,c,d)
        l += 1
        
        # find the cell it is cutting through
        @inline cell_i, cell_j = find_cell((a_old, b_old),
        						   (a, b))
        
        # distance travelled through that cell
        distance = sqrt((a_old - a) ^2 +
        				(b_old - b) ^2)
        # cell value times distance travelled through
        Atomix.@atomic img[cell_i, cell_j, i_z] += 
            distance * sinogram[k, r, i_z] * absorption_f(a, b, a0, b0)
    end
end

