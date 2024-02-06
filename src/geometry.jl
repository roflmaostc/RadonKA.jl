export RadonGeometry, RadonParallelCircle, RadonFlexibleCircle

abstract type RadonGeometry end

"""
    RadonParallelCircle(N, in_height)

`N` is the size of the first and second dimension of the array.
`in_height` is a vector or range indicating the positions of the detector
with respect to the midpoint which is located at `N ÷ 2 + 1`.

So an array of size `N=10` the default definition is: `RadonParallelCircle(10, -4:4)`
So the resulting sinogram has the shape: `(9, length(angles), size(array, 3))`
"""
struct RadonParallelCircle{T} <: RadonGeometry
    N::Int
    in_height::AbstractVector{T}
end


"""
    RadonFlexibleCircle(N, in_height, out_height)

`N` is the size of the first and second dimension of the array.
`in_height` is a vector or range indicating the vertical positions of the rays entering the circle 
with respect to the midpoint which is located at `N ÷ 2 + 1`.
`out_height` is a vector or range indicating the vertical positions of the rays exiting the circle 
with respect to the midpoint which is located at `N ÷ 2 + 1`.

One definition could be: `RadonFlexibleCircle(10, -4:4, zeros((9,)))`
It would describe rays which enter the circle at positions `-4:4` but all of them would focus at the position 0 when leaving the circle.
This is an extreme form of cone beam tomography.
"""
struct RadonFlexibleCircle{T} <: RadonGeometry
    N::Int
    in_height::AbstractVector{T}
    out_height::AbstractVector{T}
end
