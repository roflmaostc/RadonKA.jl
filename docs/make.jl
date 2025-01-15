using RadonKA, Documenter, DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "ref.bib");
    style=:numeric
)

DocMeta.setdocmeta!(RadonKA, :DocTestSetup, :(using RadonKA); recursive=true)
makedocs(modules = [RadonKA], 
         sitename = "RadonKA.jl", 
         pages = Any[
            "RadonKA.jl" => "index.md",
            "Mathematical Background" =>  "mathematical.md",
            "Simple Tutorial" =>  "tutorial.md",
            "Specifying different geometries and absorption" =>  "geometries.md",
            "Benchmark with Matlab and Astra" =>  "benchmark.md",
            "Function Docstrings" =>  "functions.md"
         ],
         warnonly=true,
         plugins=[bib,]
        )

deploydocs(repo = "github.com/roflmaostc/RadonKA.jl.git", devbranch="main")

