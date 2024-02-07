using RadonKA, CUDA, BenchmarkTools




for sz in [(512,512,100), (256, 256)]
    for d in [Array, CuArray]
        @show d, sz
        angles = d(range(0f0, 360f0, 360))
        arr = d(randn(Float32, sz))
        iarr = radon(arr, angles)

        if d === Array
            @btime radon($arr, $angles)
            @btime iradon($iarr, $angles)
        else
            @btime CUDA.@sync radon($arr, $angles)
            @btime CUDA.@sync iradon($iarr, $angles)
        end
    end
end
