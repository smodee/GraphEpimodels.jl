using Test
using GraphEpimodels
using Random

@testset "GraphEpimodels.jl" begin
    include("test_chasescape.jl")
    include("test_lattices.jl")
    include("test_erdos_renyi.jl")
    include("test_complete_graph.jl")
end
