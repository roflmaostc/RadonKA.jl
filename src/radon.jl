export radon



radon(img::AbstractArray{T, 3}, angles::AbstractArray{T2, 1}; backend=CPU()) where {T, T2} =
    radon(img, T.(angles); backend)

# handle 2D
radon(img::AbstractArray{T, 2}, angles::AbstractArray{T2, 1}; backend=CPU()) where {T, T2} =
    view(radon(reshape(img, (size(img)..., 1)), angles; backend), :, :, 1)

"""
    radon(I, θs; backend=CPU())

Calculates the parallel Radon transform of the AbstractArray `I`.
The first two dimensions are y and x. The third dimension is z, the rotational axis.
`size(I, 1)` and `size(I, 2)` have to be equal and a even number. 
The Radon transform is rotated around the pixel `size(I, 1) ÷ 2 + 1`, so there
is always a real center pixel!
Works either with a `AbstractArray{T, 3}` or `AbstractArray{T, 2}`.

`θs` is a vector or range storing the angles in radians.

`backend` can be either `CPU()` for multithreaded CPU execution or
`CUDABackend()` for CUDA execution. In principle, all backends of KernelAbstractions.jl should work
but are not tested


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
function radon(img::AbstractArray{T, 3}, angles::AbstractArray{T, 1};
			   backend=CPU()) where T
    @assert iseven(size(img, 1)) && iseven(size(img, 2)) && size(img, 1) == size(img, 2) "Arrays has to be quadratic and even sized shape"
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
    # the y_dists samplings, in principle we can add this as function parameter 
    y_dists = similar(img, (size(img, 1) - 1, ))
    y_dists .= -radius:radius
    
    #@show typeof(sinogram), typeof(img), typeof(y_dists), typeof(angles)
    kernel! = radon_kernel!(backend)
    kernel!(sinogram::AbstractArray{T}, img, y_dists, angles, mid, radius,
    				ndrange=(N, N_angles, size(img, 3)))
    
    return sinogram
end

"""
    radon_kernel!(sinogram, img,
                  y_dists, angles, mid, radius)

"""
@kernel function radon_kernel!(sinogram::AbstractArray{T},
			img, y_dists, angles, mid, radius) where T
    # r is the index of the angles
    # k is the index of the detector spatial coordinate
    # i_z is the index for the z dimension
    k, r, i_z = @index(Global, NTuple)
    
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
        @inbounds tmp += distance * img[cell_i, cell_j, i_z]
    end
    @inbounds sinogram[k, r, i_z] = tmp
end
