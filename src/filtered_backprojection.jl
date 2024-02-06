export filtered_backprojection

"""
    filtered_backprojection(sinogram, θs; 
                            geometry, μ)

Calculates the simple Filtered Backprojection in CT with applying a ramp filter
in Fourier space.

See [`radon`](@ref) for the explanation of the keyword arguments
"""
function filtered_backprojection(sinogram::AbstractArray{T}, θs::AbstractVector;
        geometry=RadonParallelCircle(size(sinogram,1) + 1,-(size(sinogram,1))÷2:(size(sinogram,1))÷2), μ=nothing) where {T}

    filter = similar(sinogram, (size(sinogram, 1),))
    filter .= rr(T, (size(sinogram, 1), )) 

    p = plan_fft(sinogram, (1,))
    sinogram = real(inv(p) * (p * sinogram .* ifftshift(filter)))
    return iradon(sinogram, θs; geometry, μ)
end
