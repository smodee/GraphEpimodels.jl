using Test
using GraphEpimodels
using Random

# Reference complete-graph neighbors: everyone except the node itself.
_ref_neighbors(n, i) = [j for j in 1:n if j != i]

# Generic (interface-level) neighbor-by-state count, used as an oracle for the
# CompleteGraph override.
function _generic_count(g, node_id, state)
    states = node_states_raw(g)
    target = state_to_int(state)
    # `count` (unlike `sum(... for ...)`) returns 0 on an empty neighbor set,
    # e.g. the single-node complete graph.
    count(j -> states[j] == target, get_neighbors(g, node_id))
end

@testset "CompleteGraph" begin
    @testset "type and construction" begin
        g = create_complete_graph(6)
        @test g isa CompleteGraph
        @test g isa AbstractImplicitGraph
        @test g isa AbstractEpidemicGraph
        # Implicit graphs are siblings of AdjacencyGraph, not subtypes.
        @test !(g isa AbstractLatticeGraph)
        @test num_nodes(g) == 6

        @test_throws ArgumentError create_complete_graph(0)
        @test_throws ArgumentError create_complete_graph(-3)
    end

    @testset "topology" begin
        n = 7
        g = create_complete_graph(n)
        for i in 1:n
            nb = get_neighbors(g, i)
            @test Set(nb) == Set(_ref_neighbors(n, i))   # all others
            @test i ∉ nb                                  # no self-loop
            @test length(nb) == length(unique(nb))        # no duplicates
            @test length(nb) == n - 1
            @test get_node_degree(g, i) == n - 1
        end
        # Every edge is mirrored (undirected).
        for i in 1:n, j in get_neighbors(g, i)
            @test i in get_neighbors(g, j)
        end
        @test_throws BoundsError get_neighbors(g, 0)
        @test_throws BoundsError get_neighbors(g, n + 1)

        # Single-node complete graph: valid, no neighbors.
        g1 = create_complete_graph(1)
        @test num_nodes(g1) == 1
        @test isempty(get_neighbors(g1, 1))
    end

    @testset "get_neighbors! reuses the buffer" begin
        g = create_complete_graph(5)
        buf = Int[]
        out = get_neighbors!(buf, g, 2)
        @test out === buf                        # filled in place, no allocation
        @test Set(out) == Set(_ref_neighbors(5, 2))
        # Reusing the same buffer for another node gives the right answer.
        out2 = get_neighbors!(buf, g, 4)
        @test Set(out2) == Set(_ref_neighbors(5, 4))
    end

    @testset "count_neighbors_by_state matches the generic oracle" begin
        rng = MersenneTwister(42)
        for n in (1, 2, 5, 20)
            g = create_complete_graph(n)
            # Random state assignment, including in-place mutation of the raw array
            # (the path the models actually use).
            states = node_states_raw(g)
            for i in 1:n
                states[i] = rand(rng, Int8(0):Int8(2))
            end
            for i in 1:n, s in (SUSCEPTIBLE, INFECTED, REMOVED)
                @test count_neighbors_by_state(g, i, s) == _generic_count(g, i, s)
            end
        end
    end

    @testset "geometry: nodes on the unit circle, no cells" begin
        n = 8
        g = create_complete_graph(n)
        @test has_layout(g) && layout_dim(g) == 2
        @test !has_cells(g)                       # not a space-filling tiling
        pos = node_positions(g)
        @test size(pos) == (2, n)
        for idx in 1:n
            @test isapprox(hypot(pos[1, idx], pos[2, idx]), 1.0; atol = 1e-9)
        end
    end

    @testset "memory: K_n is O(n), not O(n^2)" begin
        # A materialized K_50000 would need ~2.5e9 neighbor entries (tens of GB).
        # The implicit type must stay tiny — bounded by the n-length state vector.
        g = create_complete_graph(50_000)
        @test num_nodes(g) == 50_000
        @test Base.summarysize(g) < 10 * 50_000   # ~1 byte/node of state + overhead
    end

    @testset "SIR runs on a complete graph and conserves nodes" begin
        g = create_complete_graph(40)
        p = SIRProcess(g, 3.0, 1.0)
        reset!(p, [1]; rng_seed = 1)
        steps = 0
        while is_active(p) && steps < 5000
            step!(p); steps += 1
        end
        counts = count_states(g)
        @test counts[INFECTED] + counts[REMOVED] >= 1
        @test counts[SUSCEPTIBLE] + counts[INFECTED] + counts[REMOVED] == num_nodes(g)
    end
end
