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
        end
    end
end
