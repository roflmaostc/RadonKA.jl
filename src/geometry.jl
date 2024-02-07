export RadonGeometry, RadonParallelCircle, RadonFlexibleCircle

"""
    abstract type RadonGeometry end

List of geometries:
* [`RadonParallelCircle`](@ref)
* [`RadonFlexibleCircle`](@ref)
"""
abstract type RadonGeometry end

"""
    RadonParallelCircle(N, in_height, weights)

`N` is the size of the first and second dimension of the array.
`in_height` is a vector or range indicating the positions of the detector
with respect to the midpoint which is located at `N ÷ 2 + 1`.

So an array of size `N=10` the default definition is: `RadonParallelCircle(10, -4:4)`
So the resulting sinogram has the shape: `(9, length(angles), size(array, 3))`

`weights` can weight each of the rays with different strength.
Per default `weights = 0 .* in_height .+ 1`
"""
struct RadonParallelCircle{T, T2} <: RadonGeometry
    N::Int
    in_height::AbstractVector{T}
    weights::AbstractVector{T2}

    function RadonParallelCircle(N, in_height)
        return new{eltype(in_height),eltype(in_height)}(N, in_height, in_height .* 0 .+ 1) 
    end

    function RadonParallelCircle(N, in_height, weights)
        return new{eltype(in_height),eltype(weights)}(N, in_height, weights) 
    end
end


"""
    RadonFlexibleCircle(N, in_height, out_height, weights)

`N` is the size of the first and second dimension of the array.
`in_height` is a vector or range indicating the vertical positions of the rays entering the circle 
with respect to the midpoint which is located at `N ÷ 2 + 1`.
`out_height` is a vector or range indicating the vertical positions of the rays exiting the circle 
with respect to the midpoint which is located at `N ÷ 2 + 1`.

One definition could be: `RadonFlexibleCircle(10, -4:4, zeros((9,)))`
It would describe rays which enter the circle at positions `-4:4` but all of them would focus at the position 0 when leaving the circle.
This is an extreme form of fan beam tomography.


`weights` can weight each of the rays with different strength.
Per default `weights = 0 .* in_height .+ 1`
"""
struct RadonFlexibleCircle{T, T2, T3} <: RadonGeometry
    N::Int
    in_height::AbstractVector{T}
    out_height::AbstractVector{T2}
    weights::AbstractVector{T3}
    function RadonFlexibleCircle(N, in_height, out_height)
        return new{eltype(in_height), eltype(out_height), eltype(in_height)}(N, in_height, out_height, in_height .* 0 .+ 1) 
    end

    function RadonFlexibleCircle(N, in_height, out_height, weights)
        return new{eltype(in_height), eltype(out_height), eltype(weights)}(N, in_height, out_height, weights) 
    end
end
