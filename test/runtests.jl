using Test
using GraphEpimodels
using Random

@testset "GraphEpimodels.jl" begin
    include("test_chasescape.jl")
    include("test_lattices.jl")
end
