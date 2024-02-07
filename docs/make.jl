using RadonKA, Documenter 

DocMeta.setdocmeta!(RadonKA, :DocTestSetup, :(using RadonKA); recursive=true)
makedocs(modules = [RadonKA], 
         sitename = "RadonKA.jl", 
         pages = Any[
            "RadonKA.jl" => "index.md",
            "Simple Tutorial" =>  "tutorial.md",
            "Specifying different geometries and absorption" =>  "geometries.md",
            "Benchmark with Matlab and Astra" =>  "benchmark.md",
            "Function Docstrings" =>  "functions.md"
         ],
         warnonly=true,
        )

deploydocs(repo = "github.com/roflmaostc/RadonKA.jl.git", devbranch="main")
