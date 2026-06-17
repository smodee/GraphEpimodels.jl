using Test
using GraphEpimodels

# Helper: every directed neighbor relation is mirrored.
function _is_symmetric(g)
    for i in 1:num_nodes(g), j in get_neighbors(g, i)
        i in get_neighbors(g, j) || return false
    end
    return true
end

# Euclidean distance between columns i and j of a 2 × N position matrix.
_coldist(pos, i, j) = hypot(pos[1, i] - pos[1, j], pos[2, i] - pos[2, j])

@testset "HypercubicLattice" begin
    @testset "dimension aliases" begin
        @test SquareLattice === HypercubicLattice{2}
        @test CubeLattice === HypercubicLattice{3}
        @test create_square_lattice(4, 4) isa HypercubicLattice{2}
        @test create_cube_lattice(3, 3, 3) isa HypercubicLattice{3}
        @test create_hypercubic_lattice(2, 2, 2, 2) isa HypercubicLattice{4}
    end

    @testset "2D non-square (legacy was buggy here)" begin
        g = create_square_lattice(3, 5, :absorbing)   # width=3, height=5
        @test num_nodes(g) == 15
        @test _is_symmetric(g)                          # legacy 3×5 was NOT symmetric
        degs = [get_node_degree(g, i) for i in 1:num_nodes(g)]
        @test maximum(degs) == 4                        # interior is 4-regular
        @test minimum(degs) == 2                        # corners
        @test count(==(2), degs) == 4                   # exactly four corners
        @test length(get_boundary_nodes(g)) == 2 * (3 + 5) - 4   # perimeter = 12
        # get_node_degree must agree with the actual neighbour list everywhere.
        @test all(get_node_degree(g, i) == length(get_neighbors(g, i)) for i in 1:num_nodes(g))
    end

    @testset "index ↔ coordinate round trip" begin
        g = create_square_lattice(6, 4)
        for i in 1:num_nodes(g)
            r, c = index_to_coord(g, i)
            @test coord_to_index(g, r, c) == i          # legacy (row, col) API
        end
        c = create_cube_lattice(4, 5, 3)
        for i in 1:num_nodes(c)
            @test coord_to_index(c, index_to_coord(c, i)) == i   # NTuple API
        end
        @test get_center_node(create_square_lattice(25, 25)) == 313  # matches legacy
    end

    @testset "3D cube topology + geometry" begin
        c = create_cube_lattice(5, 5, 5, :absorbing)
        @test num_nodes(c) == 125
        @test _is_symmetric(c)
        degs = [get_node_degree(c, i) for i in 1:num_nodes(c)]
        @test maximum(degs) == 6                        # interior is 6-regular
        @test get_node_degree(c, get_center_node(c)) == 6
        # Geometry: a 3D layout, no 2D cell tiling.
        @test supported_layout_dims(c) == (3,)
        @test layout_dim(c) == 3 && has_layout(c)
        @test !has_cells(c)
        pos = node_positions(c)
        @test size(pos) == (3, 125)
        # Neighbours sit at unit Euclidean distance in the 3D embedding.
        ctr = get_center_node(c)
        for j in get_neighbors(c, ctr)
            d = sqrt(sum(abs2, pos[:, ctr] .- pos[:, j]))
            @test isapprox(d, 1.0; atol = 1e-9)
        end
    end

    @testset "4D topology" begin
        h = create_hypercubic_lattice((4, 4, 4, 4), boundary = :absorbing)
        @test num_nodes(h) == 256
        @test _is_symmetric(h)
        @test get_node_degree(h, get_center_node(h)) == 8   # interior is 2D = 8-regular
        @test supported_layout_dims(h) == ()                # no intrinsic layout in d≥4
        @test !has_layout(h)
        @test_throws ErrorException node_positions(h)       # nothing to draw
    end

    @testset "periodic wrap (torus)" begin
        for g in (create_torus(8), create_cube_lattice(4, 4, 4, :periodic))
            @test _is_symmetric(g)
            D = g isa CubeLattice ? 3 : 2
            @test all(get_node_degree(g, i) == 2D for i in 1:num_nodes(g))  # every node full degree
            @test isempty(get_boundary_nodes(g))
            @test !has_boundary(g)
            @test distance_to_boundary(g, 1) == Inf
        end
    end

    @testset "2D geometry unchanged" begin
        g = create_square_lattice(7, 7)
        @test supported_layout_dims(g) == (2,) && has_cells(g)
        @test size(node_positions(g)) == (2, 49)
        cells = cell_polygons(g)
        @test length(cells) == 49 && size(cells[1], 2) == 4   # unit-square cells
    end

    @testset "SIR runs on a 3D cube" begin
        c = create_cube_lattice(10, 10, 10, :absorbing)
        p = create_sir_process(c, 3.0, 1.0; initial_infected = :center, rng_seed = 1)
        steps = 0
        while is_active(p) && steps < 50_000
            step!(p); steps += 1
        end
        counts = count_states(c)
        @test counts[INFECTED] + counts[REMOVED] >= 1
        @test counts[SUSCEPTIBLE] + counts[INFECTED] + counts[REMOVED] == num_nodes(c)
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError create_square_lattice(0, 5)
        @test_throws ArgumentError create_hypercubic_lattice((10, -1, 3))
        @test_throws ArgumentError create_square_lattice(5, 5, :bogus)
    end
end

@testset "TriangularLattice" begin
    tri = create_triangular_lattice(7, 7)

    @testset "topology" begin
        @test num_nodes(tri) == 49
        @test _is_symmetric(tri)
        degs = [length(get_neighbors(tri, i)) for i in 1:num_nodes(tri)]
        @test maximum(degs) == 6                # interior nodes are 6-regular
        @test count(==(6), degs) == 25          # the inner 5×5 block
        # No node lists itself or a duplicate.
        for i in 1:num_nodes(tri)
            nb = get_neighbors(tri, i)
            @test i ∉ nb
            @test length(nb) == length(unique(nb))
        end
        @test length(get_boundary_nodes(tri)) == 24   # perimeter of 7×7
    end

    @testset "geometry" begin
        @test has_layout(tri) && layout_dim(tri) == 2 && has_cells(tri)
        pos = node_positions(tri)
        @test size(pos) == (2, 49)
        # All 6 neighbors of an interior node are at unit distance.
        ci = findfirst(i -> length(get_neighbors(tri, i)) == 6, 1:num_nodes(tri))
        for j in get_neighbors(tri, ci)
            @test isapprox(_coldist(pos, ci, j), 1.0; atol = 1e-9)
        end
        cells = cell_polygons(tri)
        @test length(cells) == 49
        @test size(cells[ci], 2) == 6           # dual cell is a hexagon
    end

    @test_throws ArgumentError create_triangular_lattice(4, 4, :periodic)
end

@testset "HexagonalLattice" begin
    hex = create_hexagonal_lattice(7, 7)

    @testset "topology" begin
        @test num_nodes(hex) == 49
        @test _is_symmetric(hex)
        degs = [length(get_neighbors(hex, i)) for i in 1:num_nodes(hex)]
        @test maximum(degs) == 3                # honeycomb is 3-regular
        for i in 1:num_nodes(hex)
            nb = get_neighbors(hex, i)
            @test i ∉ nb
            @test length(nb) == length(unique(nb))
        end
        @test length(get_boundary_nodes(hex)) == 24
    end

    @testset "geometry" begin
        @test has_layout(hex) && layout_dim(hex) == 2 && has_cells(hex)
        pos = node_positions(hex)
        @test size(pos) == (2, 49)
        hi = findfirst(i -> length(get_neighbors(hex, i)) == 3, 1:num_nodes(hex))
        # Unit-length edges, mutually 120° apart.
        for j in get_neighbors(hex, hi)
            @test isapprox(_coldist(pos, hi, j), 1.0; atol = 1e-9)
        end
        angs = sort([atand(pos[2, j] - pos[2, hi], pos[1, j] - pos[1, hi])
                     for j in get_neighbors(hex, hi)])
        @test isapprox(angs[2] - angs[1], 120.0; atol = 1e-6)
        @test isapprox(angs[3] - angs[2], 120.0; atol = 1e-6)
        cells = cell_polygons(hex)
        @test length(cells) == 49
        @test size(cells[hi], 2) == 3           # dual cell is a triangle
    end

    @test_throws ArgumentError create_hexagonal_lattice(4, 4, :periodic)
end

@testset "epidemic models run on new lattices" begin
    # The generic count_neighbors_by_state fallback must carry SIR through.
    for g in (create_triangular_lattice(15, 15), create_hexagonal_lattice(15, 15))
        p = SIRProcess(g, 3.0, 1.0)
        center = (num_nodes(g) + 1) ÷ 2
        reset!(p, [center]; rng_seed = 1)
        steps = 0
        while is_active(p) && steps < 500
            step!(p); steps += 1
        end
        counts = count_states(g)
        @test counts[INFECTED] + counts[REMOVED] >= 1   # something spread
        @test counts[SUSCEPTIBLE] + counts[INFECTED] + counts[REMOVED] == num_nodes(g)
    end
end

# Regression for the seeding over-count bug: when seeds are mutually adjacent
# (center + all its neighbors), the active tracker's per-node susceptible-neighbor
# counts must equal the freshly recomputed truth — i.e. the active_tracker
# invariant "every active node has >=1 susceptible neighbor" holds from step 0.
# Counting susceptible neighbors in the same pass that marks seeds infected used
# to leave earlier seeds stale-high, firing the "marked active" warning.
@testset "active tracker boundary counts are exact after overlapping seeding" begin
    # Number of active-tracker entries whose stored count disagrees with truth.
    function _stale_count(p)
        sum(p.active_tracker.active_nodes; init = 0) do (node, stored)
            count_neighbors_by_state(p.graph, node, SUSCEPTIBLE) == stored ? 0 : 1
        end
    end

    # Seed the center plus every one of its neighbors so the seed set overlaps.
    seeds_of(g) = (c = num_nodes(g) ÷ 2; unique(vcat(c, get_neighbors(g, c))))

    for g in (create_triangular_lattice(25, 25),
              create_hexagonal_lattice(25, 25),
              create_square_lattice(25, 25, :absorbing))
        seeds = seeds_of(g)

        @testset "SIR" begin
            p = SIRProcess(g, 3.0, 1.0)
            reset!(p, seeds; rng_seed = 42)

            # Every stored boundary count matches the recomputed truth...
            @test _stale_count(p) == 0
            # ...and so does the aggregate the infection rate is built from.
            @test GraphEpimodels.get_total_boundary(p.active_tracker) ==
                  sum(n -> count_neighbors_by_state(g, n, SUSCEPTIBLE), seeds)

            # A seed fully surrounded by other seeds has zero susceptible
            # neighbors: it must be absent from the active tracker yet still
            # tracked for spontaneous recovery.
            center = num_nodes(g) ÷ 2
            @test count_neighbors_by_state(g, center, SUSCEPTIBLE) == 0
            @test center ∉ keys(p.active_tracker.active_nodes)
            @test center ∈ p.infected_nodes

            # The invariant must survive a full run, not just the seeding.
            steps = 0
            worst = 0
            while is_active(p) && steps < 200_000
                step!(p); steps += 1
                worst = max(worst, _stale_count(p))
            end
            @test worst == 0
        end

        @testset "ZIM" begin
            # ZIM shares the seeding loop and the same active_tracker invariant.
            p = ZIMProcess(g, 3.0, 1.0)
            reset!(p, seeds; rng_seed = 42)
            @test _stale_count(p) == 0
            @test GraphEpimodels.get_total_boundary(p.active_tracker) ==
                  sum(n -> count_neighbors_by_state(g, n, SUSCEPTIBLE), seeds)

            # The invariant must survive a full run, not just the seeding. ZIM has
            # a second event type (kill) that SIR lacks, so it needs its own
            # full-run check; previously only the post-reset state was tested.
            steps = 0
            worst = 0
            while is_active(p) && steps < 200_000
                step!(p); steps += 1
                worst = max(worst, _stale_count(p))
            end
            @test worst == 0
        end
    end
end

# Weighted sampling must not depend on Dict iteration order. The active-node
# tracker is a Dict, whose iteration order depends on insertion/deletion/rehash
# history. If sampling walked that order, two trackers with identical contents
# but different histories (a fresh process vs. one reused across simulations, or
# per-thread processes that ran different work) would map the same random draw
# onto different nodes — making results irreproducible between serial and
# threaded runs. The sampler sorts into a canonical order to prevent this.
@testset "weighted sampling is independent of Dict insertion order" begin
    pairs = [(10, 3), (25, 1), (7, 4), (42, 2), (13, 5), (99, 1), (56, 3), (3, 2)]

    t1 = GraphEpimodels.DictActiveTracker()
    for (nd, w) in pairs
        GraphEpimodels.add_active_node!(t1, nd, w)
    end

    # Same contents, but built in reverse order and churned to force a different
    # internal hash-table layout.
    t2 = GraphEpimodels.DictActiveTracker()
    for (nd, w) in reverse(pairs)
        GraphEpimodels.add_active_node!(t2, nd, w)
    end
    GraphEpimodels.add_active_node!(t2, 1000, 9)
    GraphEpimodels.remove_active_node!(t2, 1000)

    @test t1.active_nodes == t2.active_nodes  # equal contents...
    for seed in 1:100
        a = GraphEpimodels._weighted_sample_active(t1, MersenneTwister(seed))
        b = GraphEpimodels._weighted_sample_active(t2, MersenneTwister(seed))
        @test a == b  # ...therefore identical selection for the same draw
    end
end

# The sampler no longer sorts per call: it reads `sorted_nodes`, which the
# add/update/remove mutators must keep equal to sort(collect(keys(active_nodes)))
# at all times. A drifted invariant would silently corrupt weighted selection, so
# check it directly under a churn of every mutator path (insert, weight-only
# update, update-to-zero removal, explicit removal, re-insert, overwrite).
@testset "active tracker maintains sorted_nodes invariant" begin
    t = GraphEpimodels.DictActiveTracker()
    canonical(tr) = sort(collect(keys(tr.active_nodes)))
    @test t.sorted_nodes == canonical(t)

    GraphEpimodels.add_active_node!(t, 50, 2)
    GraphEpimodels.add_active_node!(t, 10, 1)
    GraphEpimodels.add_active_node!(t, 30, 3)
    GraphEpimodels.add_active_node!(t, 20, 1)
    @test t.sorted_nodes == canonical(t) == [10, 20, 30, 50]

    GraphEpimodels.update_active_node!(t, 30, 5)   # weight-only: order unchanged
    @test t.sorted_nodes == canonical(t) == [10, 20, 30, 50]

    GraphEpimodels.add_active_node!(t, 50, 4)       # overwrite existing: no dup insert
    @test t.sorted_nodes == canonical(t) == [10, 20, 30, 50]

    GraphEpimodels.update_active_node!(t, 20, 0)    # drop to inactive via update
    @test t.sorted_nodes == canonical(t) == [10, 30, 50]

    GraphEpimodels.remove_active_node!(t, 10)       # explicit removal
    GraphEpimodels.remove_active_node!(t, 999)      # absent: no-op, stays consistent
    @test t.sorted_nodes == canonical(t) == [30, 50]

    GraphEpimodels.update_active_node!(t, 5, 2)     # newly active via update inserts in order
    @test t.sorted_nodes == canonical(t) == [5, 30, 50]

    GraphEpimodels.clear_active_nodes!(t)
    @test isempty(t.sorted_nodes) && isempty(t.active_nodes)
end

# SIR recovery now samples from a dense Vector with O(1) swap-removal, backed by a
# node→index map. A stale index would not necessarily crash (indices can stay
# in-range), so check the bookkeeping invariant directly across a full run: the
# infected vector has no duplicates, its index map points each node to its own
# slot, and together they exactly equal the graph's INFECTED nodes.
@testset "SIR infected-set swap-remove bookkeeping invariant" begin
    g = create_square_lattice(25, 25, :absorbing)
    p = SIRProcess(g, 3.0, 1.0)
    reset!(p, [num_nodes(g) ÷ 2]; rng_seed = 7)

    function _infected_consistent(p)
        nodes = p.infected_nodes
        idx = p.infected_index
        length(nodes) == length(idx) || return false
        for (i, nd) in enumerate(nodes)
            get(idx, nd, 0) == i || return false
        end
        # The dense set must match the graph's actual INFECTED nodes exactly.
        Set(nodes) == Set(get_nodes_in_state(g, INFECTED))
    end

    @test _infected_consistent(p)
    steps = 0
    ok = true
    while is_active(p) && steps < 200_000
        step!(p); steps += 1
        ok &= _infected_consistent(p)
    end
    @test ok
    @test isempty(p.infected_nodes) && isempty(p.infected_index)  # ran to extinction
end
