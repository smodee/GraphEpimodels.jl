using Test
using GraphEpimodels
using Random

# Oracle: reference neighbor set for node i in a tree whose root has `R` children
# and whose other internal nodes have `b` children, with `n` total nodes (1-indexed
# BFS order). Covers both conventions: d-ary uses R == b == k; the graph-theory
# regular (Cayley) tree uses R == d, b == d - 1.
function _tree_ref(R, b, n, i)
    neighbors = Int[]
    if i > 1
        push!(neighbors, i <= R + 1 ? 1 : 2 + (i - R - 2) ÷ b)   # parent
    end
    first_child = i == 1 ? 2 : R + 2 + (i - 2) * b
    if first_child <= n
        nc = i == 1 ? R : b
        append!(neighbors, first_child:min(first_child + nc - 1, n))  # children
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
        # d-ary: n = (k^h - 1) ÷ (k - 1)
        @test num_nodes(create_dary_tree(2, 1)) == 1
        @test num_nodes(create_dary_tree(2, 3)) == 7
        @test num_nodes(create_dary_tree(2, 4)) == 15
        @test num_nodes(create_dary_tree(3, 3)) == 13
        @test num_nodes(create_dary_tree(4, 3)) == 21

        # Cayley (graph-theory regular): root degree d, internal degree d.
        @test num_nodes(create_regular_tree(2, 1)) == 1
        @test [num_nodes(create_regular_tree(2, h)) for h in 1:4] == [1, 3, 5, 7]  # d=2 ⇒ path
        @test [num_nodes(create_regular_tree(3, h)) for h in 1:4] == [1, 4, 10, 22]
        @test num_nodes(create_regular_tree(4, 3)) == 17

        # Shared struct, distinct child counts.
        t = create_dary_tree(2, 3)
        @test t isa RegularTree
        @test t isa AbstractImplicitGraph
        @test !(t isa AbstractLatticeGraph)
        @test t.root_children == 2 && t.branching == 2 && t.height == 3

        c = create_regular_tree(3, 3)
        @test c.root_children == 3 && c.branching == 2

        @test_throws ArgumentError create_dary_tree(1, 3)      # k < 2
        @test_throws ArgumentError create_regular_tree(1, 3)   # d < 2
        @test_throws ArgumentError create_dary_tree(2, 0)      # h < 1
        @test_throws ArgumentError create_regular_tree(3, 0)   # h < 1
    end

    @testset "Topology vs oracle" begin
        cases = [(create_dary_tree(2, 3),    2, 2),
                 (create_dary_tree(3, 3),    3, 3),
                 (create_regular_tree(3, 4), 3, 2),   # Cayley: R=3, b=2
                 (create_regular_tree(4, 3), 4, 3),
                 (create_regular_tree(2, 4), 2, 1)]   # Cayley path
        for (g, R, b) in cases
            n = num_nodes(g)
            for i in 1:n
                nb = get_neighbors(g, i)
                @test Set(nb) == Set(_tree_ref(R, b, n, i))
                @test i ∉ nb                              # no self-loop
                @test length(nb) == length(unique(nb))    # no duplicates
            end
            for i in 1:n, j in get_neighbors(g, i)        # undirected: edges mirrored
                @test i in get_neighbors(g, j)
            end
        end
    end

    @testset "Degrees" begin
        # d-ary binary tree, h=3: root deg 2, internal (2,3) deg 3, leaves deg 1.
        g = create_dary_tree(2, 3)
        @test get_node_degree(g, 1) == 2
        @test get_node_degree(g, 2) == 3
        @test get_node_degree(g, 4) == 1

        # Cayley d=3: root degree 3, every internal vertex degree 3, leaves deg 1.
        c = create_regular_tree(3, 3)
        @test get_node_degree(c, 1) == 3
        @test get_node_degree(c, 2) == 3            # 1 parent + 2 children
        @test get_node_degree(c, num_nodes(c)) == 1 # leaf

        # Degree matches actual neighbor count for all nodes, both conventions.
        trees = vcat([create_dary_tree(k, h) for k in [2, 3, 4] for h in [1, 2, 3, 4]],
                     [create_regular_tree(d, h) for d in [2, 3, 4] for h in [1, 2, 3, 4]])
        for tree in trees
            for i in 1:num_nodes(tree)
                @test get_node_degree(tree, i) == length(get_neighbors(tree, i))
            end
        end

        # Single-node tree (h=1): root has degree 0.
        for solo in (create_dary_tree(2, 1), create_regular_tree(3, 1))
            @test get_node_degree(solo, 1) == 0
            @test isempty(get_neighbors(solo, 1))
        end
    end

    @testset "get_neighbors! buffer reuse" begin
        g = create_dary_tree(2, 4)
        buf = Int[]
        for i in 1:num_nodes(g)
            result = get_neighbors!(buf, g, i)
            @test result === buf          # same buffer object returned
            @test Set(buf) == Set(_tree_ref(2, 2, num_nodes(g), i))
        end
    end

    @testset "State counting oracle" begin
        for g in (create_dary_tree(2, 4), create_regular_tree(3, 4))
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
        # Both conventions, both layout dimensions: root at origin, every node on
        # the shell of radius = its level.
        cases = [create_dary_tree(2, 1), create_dary_tree(2, 3), create_dary_tree(3, 3),
                 create_regular_tree(3, 3), create_regular_tree(4, 3), create_regular_tree(2, 4)]
        for g in cases
            n = num_nodes(g)
            @test has_layout(g)
            @test layout_dim(g) == 2
            @test supported_layout_dims(g) == (2, 3)
            @test !has_cells(g)

            R, b, h = g.root_children, g.branching, g.height
            for dim in (2, 3)
                pos = node_positions(g; dim = dim)
                @test size(pos) == (dim, n)
                @test all(iszero, @view pos[:, 1])           # root at origin

                node_idx = 1
                for level in 0:(h - 1)
                    level_count = level == 0 ? 1 : R * b^(level - 1)
                    for _ in 1:level_count
                        @test sqrt(sum(abs2, @view pos[:, node_idx])) ≈ Float64(level) atol = 1e-9
                        node_idx += 1
                    end
                end
            end
        end

        # 2D rings are equidistant: all angular gaps within a level are equal.
        p = node_positions(create_dary_tree(3, 3); dim = 2)
        ang(i) = atan(p[2, i], p[1, i])
        gaps = sort([mod(ang(i + 1) - ang(i), 2π) for i in 5:12])  # ternary level 2 = nodes 5..13
        @test gaps[end] - gaps[1] < 1e-9
    end

    @testset "SIR conservation" begin
        for g in (create_dary_tree(2, 5), create_regular_tree(3, 4))
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
