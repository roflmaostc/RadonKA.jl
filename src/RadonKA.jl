module RadonKA

using KernelAbstractions, CUDA, CUDA.CUDAKernels
using IndexFunArrays
using FFTW
using Atomix

include("utils.jl")
include("radon.jl")
include("iradon.jl")


end
