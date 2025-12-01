export backproject_filtered

"""
    backproject_filtered(sinogram, θs; 
                            geometry, μ, filter)

Reconstruct the image from the `sinogram` using the filtered backprojection algorithm.

* `filter=nothing`: The filter to be applied in Fourier space. If `nothing`, a ramp filter is used. `filter` should be a 1D array of the same length as the sinogram.

See [`radon`](@ref) for the explanation of the keyword arguments
"""
function backproject_filtered(sinogram::AbstractArray{T}, θs::AbstractVector;
        geometry=RadonParallelCircle(size(sinogram,1) + 1,-(size(sinogram,1))÷2:(size(sinogram,1))÷2), μ=nothing,
        filter=nothing) where {T}
    _backproject(sinogram, θs, geometry, μ, filter, T)
end


function _backproject(sinogram, θs, geometry, μ, filter::Nothing, ::Type{T}) where {T}
    filter = similar(sinogram, (size(sinogram, 1),))
    filter .= rr(T, (size(sinogram, 1), )) 

    p = plan_fft(sinogram, (1,))
    sinogram = real(inv(p) * (p * sinogram .* ifftshift(filter)))
    return backproject(sinogram, θs; geometry, μ)
end


function _backproject(sinogram, θs, geometry, μ, filter::AbstractArray, ::Type{T}) where {T}
    p = plan_fft(sinogram, (1,))
    sinogram = real(inv(p) * (p * sinogram .* ifftshift(filter)))
    return backproject(sinogram, θs; geometry, μ)
end
