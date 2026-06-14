using Test
using GraphEpimodels
using Random
using Statistics

# Every directed neighbor relation is mirrored (undirected graph).
function _er_is_symmetric(g)
    for i in 1:num_nodes(g), j in get_neighbors(g, i)
        i in get_neighbors(g, j) || return false
    end
    return true
end

# No self-loops, no duplicate neighbors.
function _er_is_simple(g)
    for i in 1:num_nodes(g)
        nb = get_neighbors(g, i)
        i in nb && return false
        length(nb) == length(unique(nb)) || return false
    end
    return true
end

_edge_count(g) = sum(length(get_neighbors(g, i)) for i in 1:num_nodes(g)) ÷ 2

@testset "ErdosRenyiGraph" begin

    @testset "construction validation" begin
        @test_throws ArgumentError create_erdos_renyi(0; p = 0.1)        # n < 1
        @test_throws ArgumentError create_erdos_renyi(10)                # neither p nor m
        @test_throws ArgumentError create_erdos_renyi(10; p = 0.1, m = 5)  # both
        @test_throws ArgumentError create_erdos_renyi(10; p = -0.1)      # p < 0
        @test_throws ArgumentError create_erdos_renyi(10; p = 1.5)       # p > 1
        @test_throws ArgumentError create_erdos_renyi(10; m = -1)        # m < 0
        @test_throws ArgumentError create_erdos_renyi(10; m = 46)        # m > n(n-1)/2 = 45
    end

    @testset "G(n,p) topology" begin
        g = create_gnp(200, 0.05; rng = MersenneTwister(1))
        @test g isa ErdosRenyiGraph
        @test g isa AbstractEpidemicGraph
        @test num_nodes(g) == 200
        @test g.model == :gnp
        @test g.p == 0.05
        @test g.m == _edge_count(g)          # realized edge count is stored
        @test _er_is_symmetric(g)
        @test _er_is_simple(g)
    end

    @testset "G(n,p) statistics & reproducibility" begin
        n, p = 500, 0.04
        g = create_gnp(n, p; rng = MersenneTwister(42))
        degs = [length(get_neighbors(g, i)) for i in 1:n]
        # Mean degree should be close to p*(n-1).
        @test isapprox(mean(degs), p * (n - 1); rtol = 0.1)

        # Same seed → identical graph; different seed → (almost surely) different.
        g2 = create_gnp(n, p; rng = MersenneTwister(42))
        g3 = create_gnp(n, p; rng = MersenneTwister(43))
        @test g.graph.adjacency_list == g2.graph.adjacency_list
        @test g.graph.adjacency_list != g3.graph.adjacency_list
    end

    @testset "G(n,m) exact edge count" begin
        g = create_gnm(100, 250; rng = MersenneTwister(7))
        @test g.model == :gnm
        @test g.m == 250
        @test _edge_count(g) == 250          # exactly m edges
        @test isapprox(g.p, 250 / (100 * 99 ÷ 2); atol = 1e-12)
        @test _er_is_symmetric(g)
        @test _er_is_simple(g)

        # Reproducible.
        g2 = create_gnm(100, 250; rng = MersenneTwister(7))
        @test g.graph.adjacency_list == g2.graph.adjacency_list
    end

    @testset "edge cases" begin
        empty_p = create_gnp(50, 0.0)
        @test _edge_count(empty_p) == 0
        @test empty_p.m == 0

        empty_m = create_gnm(50, 0)
        @test _edge_count(empty_m) == 0

        total = 20 * 19 ÷ 2
        complete_p = create_gnp(20, 1.0)
        @test _edge_count(complete_p) == total
        @test all(length(get_neighbors(complete_p, i)) == 19 for i in 1:20)

        complete_m = create_gnm(20, total)
        @test _edge_count(complete_m) == total
        @test _er_is_simple(complete_m)

        single = create_gnp(1, 0.5)
        @test num_nodes(single) == 1
        @test _edge_count(single) == 0
    end

    @testset "interface forwarding" begin
        g = create_gnp(80, 0.1; rng = MersenneTwister(3))
        buf = Int[]
        for i in 1:num_nodes(g)
            nb = get_neighbors(g, i)
            @test get_neighbors!(buf, g, i) == nb                  # buffer variant agrees
            @test get_node_degree(g, i) == length(nb)              # degree agrees
        end
        # count_neighbors_by_state matches a manual count after seeding some infected.
        states = node_states_raw(g)
        states[1:10] .= Int8(1)
        set_node_states_raw!(g, states)
        for i in 1:num_nodes(g)
            manual = count(j -> node_states_raw(g)[j] == Int8(1), get_neighbors(g, i))
            @test count_neighbors_by_state(g, i, INFECTED) == manual
        end
    end

    @testset "epidemic models run on ER graphs" begin
        # p well above the giant-component threshold (1/n) so spread can happen.
        for g in (create_gnp(300, 0.03; rng = MersenneTwister(11)),
                  create_gnm(300, 900; rng = MersenneTwister(12)))
            # Seed the highest-degree node to give the epidemic a foothold.
            seed = argmax([length(get_neighbors(g, i)) for i in 1:num_nodes(g)])

            for p in (SIRProcess(g, 3.0, 1.0), ZIMProcess(g, 3.0, 1.0))
                reset!(p, [seed]; rng_seed = 1)
                steps = 0
                while is_active(p) && steps < 5000
                    step!(p); steps += 1
                end
                counts = count_states(g)
                @test counts[SUSCEPTIBLE] + counts[INFECTED] + counts[REMOVED] == num_nodes(g)
                @test counts[INFECTED] + counts[REMOVED] >= 1     # something happened
            end
        end
    end
end
