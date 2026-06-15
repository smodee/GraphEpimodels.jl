using Test
using GraphEpimodels
using Random

# Reference neighbor sets (the topology each implicit type must reproduce).
_path_ref(n, i)  = filter(j -> 1 <= j <= n, [i - 1, i + 1])
_cycle_ref(n, i) = [i == 1 ? n : i - 1, i == n ? 1 : i + 1]
_star_ref(n, i)  = i == 1 ? collect(2:n) : [1]

# Generic (interface-level) neighbor-by-state count, used as an oracle for the
# per-type overrides. `count` returns 0 on an empty neighbor set.
function _oracle_count(g, node_id, state)
    states = node_states_raw(g)
    target = state_to_int(state)
    count(j -> states[j] == target, get_neighbors(g, node_id))
end

# Shared checks: topology matches the reference, plus the implicit-type invariants.
function _check_topology(g, ref, n)
    for i in 1:n
        nb = get_neighbors(g, i)
        @test Set(nb) == Set(ref(n, i))
        @test i ∉ nb                              # no self-loop
        @test length(nb) == length(unique(nb))    # no duplicates
        @test get_node_degree(g, i) == length(ref(n, i))
    end
    for i in 1:n, j in get_neighbors(g, i)        # undirected: every edge mirrored
        @test i in get_neighbors(g, j)
    end
end

function _check_count_oracle(g, n)
    rng = MersenneTwister(7)
    states = node_states_raw(g)
    for i in 1:n
        states[i] = rand(rng, Int8(0):Int8(2))    # mutate the raw array in place
    end
    for i in 1:n, s in (SUSCEPTIBLE, INFECTED, REMOVED)
        @test count_neighbors_by_state(g, i, s) == _oracle_count(g, i, s)
    end
end

function _check_sir_conserves(g)
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

@testset "Structured implicit graphs" begin
    @testset "PathGraph" begin
        g = create_path_graph(8)
        @test g isa PathGraph && g isa AbstractImplicitGraph
        @test !(g isa AbstractLatticeGraph)
        @test num_nodes(g) == 8
        @test_throws ArgumentError create_path_graph(0)

        _check_topology(g, _path_ref, 8)
        # Endpoints have degree 1, interior degree 2.
        @test get_node_degree(g, 1) == 1
        @test get_node_degree(g, 8) == 1
        @test get_node_degree(g, 4) == 2

        # Single-node path: valid, no neighbors.
        @test isempty(get_neighbors(create_path_graph(1), 1))

        _check_count_oracle(g, 8)

        @test has_layout(g) && layout_dim(g) == 2 && !has_cells(g)
        pos = node_positions(g)
        @test size(pos) == (2, 8)
        @test all(pos[2, :] .== 0.0)               # collinear on the x-axis
        @test pos[1, :] == collect(1.0:8.0)

        _check_sir_conserves(create_path_graph(30))
    end

    @testset "CycleGraph" begin
        g = create_cycle_graph(9)
        @test g isa CycleGraph && g isa AbstractImplicitGraph
        @test !(g isa AbstractLatticeGraph)
        @test num_nodes(g) == 9
        @test_throws ArgumentError create_cycle_graph(2)   # needs >= 3

        _check_topology(g, _cycle_ref, 9)
        @test all(get_node_degree(g, i) == 2 for i in 1:9)
        # Wraparound edge closes the ring.
        @test Set(get_neighbors(g, 1)) == Set([9, 2])

        _check_count_oracle(g, 9)

        @test has_layout(g) && layout_dim(g) == 2 && !has_cells(g)
        pos = node_positions(g)
        @test size(pos) == (2, 9)
        for idx in 1:9
            @test isapprox(hypot(pos[1, idx], pos[2, idx]), 1.0; atol = 1e-9)
        end

        _check_sir_conserves(create_cycle_graph(30))
    end

    @testset "StarGraph" begin
        n = 7
        g = create_star_graph(n)
        @test g isa StarGraph && g isa AbstractImplicitGraph
        @test !(g isa AbstractLatticeGraph)
        @test num_nodes(g) == n
        @test_throws ArgumentError create_star_graph(1)    # needs >= 2

        _check_topology(g, _star_ref, n)
        @test get_node_degree(g, 1) == n - 1               # center
        @test all(get_node_degree(g, i) == 1 for i in 2:n) # leaves
        @test Set(get_neighbors(g, 1)) == Set(2:n)
        @test get_neighbors(g, 3) == [1]

        _check_count_oracle(g, n)

        @test has_layout(g) && layout_dim(g) == 2 && !has_cells(g)
        pos = node_positions(g)
        @test size(pos) == (2, n)
        @test pos[:, 1] == [0.0, 0.0]                      # center at the origin
        for idx in 2:n
            @test isapprox(hypot(pos[1, idx], pos[2, idx]), 1.0; atol = 1e-9)
        end

        _check_sir_conserves(create_star_graph(30))
    end
end
