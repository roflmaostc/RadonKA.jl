module RadonKA

using KernelAbstractions
using IndexFunArrays
using FFTW
using Atomix

include("utils.jl")
include("radon.jl")
include("iradon.jl")

end
