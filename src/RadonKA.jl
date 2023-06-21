module RadonKA

using KernelAbstractions, CUDA, CUDA.CUDAKernels, Adapt, Interpolations
using IndexFunArrays
using FFTW

include("radon.jl")
include("iradon.jl")


# TODO
# * try allocating sin(θ) and cos(θ) since we only need those



function distance_to_boundary(x, y, θ, R)
 	# avoid division by zero
	if iszero(x) && iszero(y)
		return R
	end

	d = sqrt(x^2 + y^2)
	α = acos(x/d)
	β = y < 0 ? π - θ - α : π - θ + α

	function quadratic_solve(a,b,c)
		return (- b + sqrt(b^2 - 4 * a * c)) / 2 / a
	end


	res = quadratic_solve(1, -2 * d * cos(β), d^2 - R^2)

	return res
end

end
