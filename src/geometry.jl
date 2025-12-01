export RadonGeometry, RadonParallelCircle, RadonFlexibleCircle

"""
    abstract type RadonGeometry end

Abstract supertype for all geometries which are supported by `radon` and `backproject`.

List of geometries:
* [`RadonParallelCircle`](@ref)
* [`RadonFlexibleCircle`](@ref)

See [`radon`](@ref) and [`backproject`](@ref) how to apply.
"""
abstract type RadonGeometry end

"""
    RadonParallelCircle(N, in_height, weights)

- `N` is the size of the first and second dimension of the input array for `radon`.

- `in_height` is a vector or a range indicating the positions of the detector with respect to the midpoint which is located at `N ÷ 2 + 1`. The rays travel along straight parallel paths through the array. Maximum and minimum values are `(N+1) ÷ 2 - 1` and `-(N+1) ÷ 2 + 1` respectively.

For example, for an array of size `N=10` the default definition is: `RadonParallelCircle(10, -4:4)`
So the resulting sinogram has the shape: `(9, length(angles), size(array, 3))`.

- `weights` can weight each of the rays with different strength. Per default `weights = 0 .* in_height .+ 1`

See [`radon`](@ref) and [`backproject`](@ref) how to apply.
"""
struct RadonParallelCircle{T, T2} <: RadonGeometry
    N::Int
    in_height::T
    weights::T2

    function RadonParallelCircle(N, in_height)
        @assert minimum(in_height) >= -(N+1) ÷ 2 + 1 && maximum(in_height) <= (N+1) ÷ 2 - 1 "The in_height values are out of bounds. Limits are: $(-(N+1) ÷ 2 + 1) and $((N+1) ÷ 2 - 1)"
        weights = in_height .* 0 .+ 1
        return new{typeof(in_height), typeof(weights)}(N, in_height, weights) 
    end

    function RadonParallelCircle(N, in_height, weights)
        @assert minimum(in_height) >= -(N+1) ÷ 2 + 1 && maximum(in_height) <= (N+1) ÷ 2 - 1 "The in_height values are out of bounds. Limits are: $(-(N+1) ÷ 2 + 1) and $((N+1) ÷ 2 - 1)"
        return new{typeof(in_height),typeof(weights)}(N, in_height, weights) 
    end
end


"""
    RadonFlexibleCircle(N, in_height, out_height, weights)

- `N` is the size of the first and second dimension of the input for `radon`.
- `in_height` is a vector or range indicating the vertical positions of the rays entering the circle with respect to the midpoint which is located at `N ÷ 2 + 1`. Maximum and minimum values are `(N+1) ÷ 2 - 1` and `-(N+1) ÷ 2 + 1` respectively.
- `out_height` is a vector or range indicating the vertical positions of the rays exiting the circle with respect to the midpoint which is located at `N ÷ 2 + 1`. Maximum and minimum values are `(N+1) ÷ 2 - 1` and `-(N+1) ÷ 2 + 1` respectively.


One definition could be: `RadonFlexibleCircle(10, -4:4, zeros((9,)))`
It would describe rays which enter the circle at positions `-4:4` but all of them would focus at the position 0 when leaving the circle.
This is an extreme form of fan beam tomography.

- `weights` can weight each of the rays with different strength. Per default `weights = 0 .* in_height .+ 1`

See [`radon`](@ref) and [`backproject`](@ref) how to apply.
"""
struct RadonFlexibleCircle{T, T2, T3} <: RadonGeometry
    N::Int
    in_height::T
    out_height::T2
    weights::T3
    function RadonFlexibleCircle(N, in_height, out_height)
        @assert minimum(in_height) >= -(N+1) ÷ 2 + 1 && maximum(in_height) <= (N+1) ÷ 2 - 1 "The in_height values are out of bounds. Limits are: $(-(N+1) ÷ 2 + 1) and $((N+1) ÷ 2 - 1)"
        @assert minimum(out_height) >= -(N+1) ÷ 2 + 1 && maximum(out_height) <= (N+1) ÷ 2 - 1 "The out_height values are out of bounds. Limits are: $(-(N+1) ÷ 2 + 1) and $((N+1) ÷ 2 - 1)"
        weights = in_height .* 0 .+ 1 
        return new{typeof(in_height), typeof(out_height), typeof(weights)}(N, in_height, out_height, weights) 
    end

    function RadonFlexibleCircle(N, in_height, out_height, weights)
        @assert minimum(in_height) >= -(N+1) ÷ 2 + 1 && maximum(in_height) <= (N+1) ÷ 2 - 1 "The in_height values are out of bounds. Limits are: $(-(N+1) ÷ 2 + 1) and $((N+1) ÷ 2 - 1)"
        @assert minimum(out_height) >= -(N+1) ÷ 2 + 1 && maximum(out_height) <= (N+1) ÷ 2 - 1 "The out_height values are out of bounds. Limits are: $(-(N+1) ÷ 2 + 1) and $((N+1) ÷ 2 - 1)"
        return new{typeof(in_height), typeof(out_height), typeof(weights)}(N, in_height, out_height, weights) 
    end
end
