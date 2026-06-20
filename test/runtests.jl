using Test
using GraphEpimodels
using Random

@testset "GraphEpimodels.jl" begin
    include("test_chasescape.jl")
    include("test_lattices.jl")
    include("test_erdos_renyi.jl")
    include("test_complete_graph.jl")
    include("test_structured_graphs.jl")
    include("test_regular_tree.jl")
    include("test_geograph.jl")
    include("test_visualization.jl")
    include("test_persistence.jl")
end
