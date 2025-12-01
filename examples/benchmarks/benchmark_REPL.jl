using RadonKA, CUDA, BenchmarkTools




for sz in [(1920, 1920), (512,512,100)]
    for d in [Array, CuArray]
        @show d, sz
        angles = d(range(0f0, 2π, 500))
        arr = d(randn(Float32, sz))
        iarr = radon(arr, angles)

        if d === Array
            @btime radon($arr, $angles)
            @btime backproject($iarr, $angles)
        else
            @btime CUDA.@sync radon($arr, $angles)
            @btime CUDA.@sync backproject($iarr, $angles)
        end
    end
end
