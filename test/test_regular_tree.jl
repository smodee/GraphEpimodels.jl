using Test
using GraphEpimodels
using Random

# Oracle: reference neighbor set for node i in a k-ary tree with n total nodes.
function _tree_ref(k, n, i)
    neighbors = Int[]
    i > 1 && push!(neighbors, (i - 2) ÷ k + 1)           # parent
    first_child = k * (i - 1) + 2
    if first_child <= n
        append!(neighbors, first_child:min(k * i + 1, n)) # children
    end
    return neighbors
end

function _oracle_count_tree(g, node_id, state)
    states = node_states_raw(g)
    target = state_to_int(state)
    count(j -> states[j] == target, get_neighbors(g, node_id))
end

@testset "RegularTree" begin

    @testset "Construction and node count" begin
        # n = (k^h - 1) ÷ (k - 1)
        @test num_nodes(create_regular_tree(2, 1)) == 1
        @test num_nodes(create_regular_tree(2, 2)) == 3
        @test num_nodes(create_regular_tree(2, 3)) == 7
        @test num_nodes(create_regular_tree(2, 4)) == 15
        @test num_nodes(create_regular_tree(3, 1)) == 1
        @test num_nodes(create_regular_tree(3, 2)) == 4
        @test num_nodes(create_regular_tree(3, 3)) == 13
        @test num_nodes(create_regular_tree(4, 3)) == 21

        t = create_regular_tree(2, 3)
        @test t isa RegularTree
        @test t isa AbstractImplicitGraph
        @test !(t isa AbstractLatticeGraph)
        @test t.branching_factor == 2
        @test t.height == 3

        @test_throws ArgumentError create_regular_tree(1, 3)   # k < 2
        @test_throws ArgumentError create_regular_tree(0, 3)
        @test_throws ArgumentError create_regular_tree(2, 0)   # h < 1
    end

    @testset "Topology: binary tree (k=2, h=3, n=7)" begin
        k, h = 2, 3
        g = create_regular_tree(k, h)
        n = num_nodes(g)
        for i in 1:n
            nb = get_neighbors(g, i)
            ref = _tree_ref(k, n, i)
            @test Set(nb) == Set(ref)
            @test i ∉ nb                              # no self-loop
            @test length(nb) == length(unique(nb))    # no duplicates
        end
        # Undirected: every edge is mirrored
        for i in 1:n, j in get_neighbors(g, i)
            @test i in get_neighbors(g, j)
        end
    end

    @testset "Topology: ternary tree (k=3, h=3, n=13)" begin
        k, h = 3, 3
        g = create_regular_tree(k, h)
        n = num_nodes(g)
        for i in 1:n
            @test Set(get_neighbors(g, i)) == Set(_tree_ref(k, n, i))
        end
        for i in 1:n, j in get_neighbors(g, i)
            @test i in get_neighbors(g, j)
        end
    end

    @testset "Degrees" begin
        # Binary tree, h=3: root deg 2, internal (2,3) deg 3, leaves (4-7) deg 1
        g = create_regular_tree(2, 3)
        @test get_node_degree(g, 1) == 2   # root
        @test get_node_degree(g, 2) == 3   # internal
        @test get_node_degree(g, 3) == 3
        @test get_node_degree(g, 4) == 1   # leaf
        @test get_node_degree(g, 7) == 1

        # Degree matches actual neighbor count for all nodes
        for k in [2, 3, 4], h in [1, 2, 3, 4]
            tree = create_regular_tree(k, h)
            n = num_nodes(tree)
            for i in 1:n
                @test get_node_degree(tree, i) == length(get_neighbors(tree, i))
            end
        end

        # Single-node tree (h=1): root has degree 0
        solo = create_regular_tree(2, 1)
        @test get_node_degree(solo, 1) == 0
        @test isempty(get_neighbors(solo, 1))
    end

    @testset "get_neighbors! buffer reuse" begin
        g = create_regular_tree(2, 4)
        buf = Int[]
        for i in 1:num_nodes(g)
            result = get_neighbors!(buf, g, i)
            @test result === buf          # same buffer object returned
            @test Set(buf) == Set(_tree_ref(2, num_nodes(g), i))
        end
    end

    @testset "State counting oracle" begin
        for (k, h) in [(2, 4), (3, 3)]
            g = create_regular_tree(k, h)
            n = num_nodes(g)
            rng = MersenneTwister(42)
            states = node_states_raw(g)
            for i in 1:n
                states[i] = rand(rng, Int8(0):Int8(2))
            end
            for i in 1:n, s in (SUSCEPTIBLE, INFECTED, REMOVED)
                @test count_neighbors_by_state(g, i, s) == _oracle_count_tree(g, i, s)
            end
        end
    end

    @testset "Geometry" begin
        for (k, h) in [(2, 1), (2, 3), (3, 3)]
            g = create_regular_tree(k, h)
            n = num_nodes(g)
            @test has_layout(g)
            @test layout_dim(g) == 2
            @test !has_cells(g)

            pos = node_positions(g)
            @test size(pos) == (2, n)

            # Root at origin
            @test pos[1, 1] ≈ 0.0 atol=1e-12
            @test pos[2, 1] ≈ 0.0 atol=1e-12

            # Level-l nodes lie on circle of radius l
            node_idx = 1
            for level in 0:(h - 1)
                level_count = k^level
                r = Float64(level)
                for _ in 1:level_count
                    dist = sqrt(pos[1, node_idx]^2 + pos[2, node_idx]^2)
                    @test dist ≈ r atol=1e-12
                    node_idx += 1
                end
            end
        end
    end

    @testset "SIR conservation" begin
        for (k, h) in [(2, 5), (3, 4)]
            g = create_regular_tree(k, h)
            p = SIRProcess(g, 3.0, 1.0)
            reset!(p, [1]; rng_seed = 1)
            steps = 0
            while is_active(p) && steps < 10_000
                step!(p); steps += 1
            end
            counts = count_states(g)
            @test counts[SUSCEPTIBLE] + counts[INFECTED] + counts[REMOVED] == num_nodes(g)
            @test counts[INFECTED] + counts[REMOVED] >= 1
        end
    end

end
