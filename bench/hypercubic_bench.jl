"""
Benchmark + equivalence check: the generic `HypercubicLattice{2}` vs a verbatim
snapshot of the legacy hand-unrolled `SquareLattice`.

Goals:
  1. Equivalence — on SQUARE lattices the new lattice must produce identical
     neighbours, boundary nodes, center node, and per-node susceptible counts.
  2. No performance regression — `get_neighbors!` and `count_neighbors_by_state`
     (the SIR/ZIM hot path), plus a full seeded SIR run, must be no slower.
  3. Document the legacy non-square bug the rewrite fixes.

Run:  julia --project=. bench/hypercubic_bench.jl
"""

using GraphEpimodels
using Random, Printf

const GE = GraphEpimodels

# =============================================================================
# Legacy SquareLattice snapshot (copied verbatim from the pre-rewrite lattice.jl)
# =============================================================================

@inline _leg_coord_to_index(row, col, height) = col + (row - 1) * height
@inline function _leg_index_to_coord(index, height)
    row, col = divrem(index - 1, height)
    return (row + 1, col + 1)
end

mutable struct LegacySquareLattice
    width::Int
    height::Int
    n_nodes::Int
    boundary::Symbol
    states::Vector{Int8}
    boundary_nodes::Vector{Int}
    function LegacySquareLattice(width, height, boundary = :absorbing)
        n = width * height
        bn = boundary == :absorbing ? _leg_boundary_nodes(width, height) : Int[]
        new(width, height, n, boundary, zeros(Int8, n), bn)
    end
end

function _leg_boundary_nodes(width, height)
    nodes = Int[]
    for col in 1:width
        push!(nodes, _leg_coord_to_index(1, col, height))
        height > 1 && push!(nodes, _leg_coord_to_index(height, col, height))
    end
    if width > 1
        for row in 2:(height - 1)
            push!(nodes, _leg_coord_to_index(row, 1, height))
            push!(nodes, _leg_coord_to_index(row, width, height))
        end
    end
    return nodes
end

function leg_get_neighbors!(nb::Vector{Int}, l::LegacySquareLattice, node_id::Int)
    row, col = _leg_index_to_coord(node_id, l.height)
    empty!(nb)
    if l.boundary == :absorbing
        row > 1       && push!(nb, _leg_coord_to_index(row - 1, col, l.height))
        row < l.height && push!(nb, _leg_coord_to_index(row + 1, col, l.height))
        col > 1       && push!(nb, _leg_coord_to_index(row, col - 1, l.height))
        col < l.width  && push!(nb, _leg_coord_to_index(row, col + 1, l.height))
    else
        nr = row == 1 ? l.height : row - 1
        sr = row == l.height ? 1 : row + 1
        wc = col == 1 ? l.width : col - 1
        ec = col == l.width ? 1 : col + 1
        push!(nb, _leg_coord_to_index(nr, col, l.height))
        push!(nb, _leg_coord_to_index(sr, col, l.height))
        push!(nb, _leg_coord_to_index(row, wc, l.height))
        push!(nb, _leg_coord_to_index(row, ec, l.height))
    end
    return nb
end

function leg_count(l::LegacySquareLattice, node_id::Int, target::Int8)
    row, col = _leg_index_to_coord(node_id, l.height)
    st = l.states
    c = 0
    if l.boundary == :absorbing
        row > 1        && (st[_leg_coord_to_index(row - 1, col, l.height)] == target && (c += 1))
        row < l.height && (st[_leg_coord_to_index(row + 1, col, l.height)] == target && (c += 1))
        col > 1        && (st[_leg_coord_to_index(row, col - 1, l.height)] == target && (c += 1))
        col < l.width  && (st[_leg_coord_to_index(row, col + 1, l.height)] == target && (c += 1))
    else
        nr = row == 1 ? l.height : row - 1
        sr = row == l.height ? 1 : row + 1
        wc = col == 1 ? l.width : col - 1
        ec = col == l.width ? 1 : col + 1
        for idx in (_leg_coord_to_index(nr, col, l.height), _leg_coord_to_index(sr, col, l.height),
                    _leg_coord_to_index(row, wc, l.height), _leg_coord_to_index(row, ec, l.height))
            st[idx] == target && (c += 1)
        end
    end
    return c
end

# =============================================================================
# Equivalence (square lattices)
# =============================================================================

function check_equivalence(n, boundary)
    new = create_square_lattice(n, n, boundary)
    leg = LegacySquareLattice(n, n, boundary)
    # Randomize states identically.
    Random.seed!(7)
    s = rand(Int8.(0:2), n * n)
    set_node_states_raw!(new, copy(s)); leg.states .= s

    nb_new = Int[]; nb_leg = Int[]
    ok = true
    for i in 1:(n * n)
        get_neighbors!(nb_new, new, i)
        leg_get_neighbors!(nb_leg, leg, i)
        if Set(nb_new) != Set(nb_leg)
            @warn "neighbour mismatch" i n boundary nb_new nb_leg; ok = false; break
        end
        if count_neighbors_by_state(new, i, SUSCEPTIBLE) != leg_count(leg, i, GE.STATE_SUSCEPTIBLE)
            @warn "count mismatch" i; ok = false; break
        end
    end
    bn_ok = Set(get_boundary_nodes(new)) == Set(leg.boundary_nodes)
    return ok && bn_ok
end

# =============================================================================
# Timing
# =============================================================================

# Robust microbenchmark: report the MINIMUM elapsed time over `trials` (the
# minimum is the standard noise-resistant estimator — it filters GC pauses and
# scheduler jitter that inflate single-shot timings).
const TRIALS = 7

function time_neighbors(get_nb!, lat, reps)
    nb = Int[]
    n = lat isa LegacySquareLattice ? lat.n_nodes : num_nodes(lat)
    best = Inf
    for _ in 1:TRIALS
        acc = 0
        t = @elapsed for _ in 1:reps, i in 1:n
            get_nb!(nb, lat, i)
            acc += length(nb)
        end
        acc == 0 && error("dead code eliminated")
        best = min(best, t)
    end
    return best, 0
end

function time_count(countfn, lat, target, reps)
    n = lat isa LegacySquareLattice ? lat.n_nodes : num_nodes(lat)
    best = Inf
    for _ in 1:TRIALS
        acc = 0
        t = @elapsed for _ in 1:reps, i in 1:n
            acc += countfn(lat, i, target)
        end
        acc < 0 && error("impossible")
        best = min(best, t)
    end
    return best, 0
end

function time_sir(make_lat)
    lat = make_lat()
    p = create_sir_process(lat, 3.0, 1.0; initial_infected = :center, rng_seed = 1)
    steps = 0
    t = @elapsed while is_active(p) && steps < 200_000
        step!(p); steps += 1
    end
    return t, steps
end

function main()
    println("="^70)
    println("EQUIVALENCE (square lattices, new vs legacy)")
    for n in (10, 25, 100), b in (:absorbing, :periodic)
        @printf "  %4dx%-4d %-10s : %s\n" n n b (check_equivalence(n, b) ? "OK" : "MISMATCH")
    end

    println("\nLEGACY NON-SQUARE BUG (new lattice is symmetric, legacy is not)")
    leg_neighbors(l, i) = leg_get_neighbors!(Int[], l, i)
    leg_symmetric(l) = all(i in leg_neighbors(l, j)
                           for i in 1:l.n_nodes for j in leg_neighbors(l, i))
    new_symmetric(g) = all(i in get_neighbors(g, j)
                           for i in 1:num_nodes(g) for j in get_neighbors(g, i))
    for (w, h) in ((3, 5), (7, 4))
        leg = LegacySquareLattice(w, h, :absorbing)
        new = create_square_lattice(w, h, :absorbing)
        @printf "  %dx%-3d : legacy symmetric=%s   new symmetric=%s\n" w h leg_symmetric(leg) new_symmetric(new)
    end

    n = 200
    reps = 50
    println("\nTIMING  ($(n)x$(n), $reps sweeps over all nodes)")
    for b in (:absorbing, :periodic)
        new = create_square_lattice(n, n, b)
        leg = LegacySquareLattice(n, n, b)
        Random.seed!(1); s = rand(Int8.(0:2), n * n)
        set_node_states_raw!(new, copy(s)); leg.states .= s

        # Warmup
        time_neighbors(get_neighbors!, new, 1); time_neighbors(leg_get_neighbors!, leg, 1)
        time_count((l, i, t) -> count_neighbors_by_state(l, i, t), new, SUSCEPTIBLE, 1)
        time_count(leg_count, leg, GE.STATE_SUSCEPTIBLE, 1)

        tn_new, _ = time_neighbors(get_neighbors!, new, reps)
        tn_leg, _ = time_neighbors(leg_get_neighbors!, leg, reps)
        tc_new, _ = time_count((l, i, t) -> count_neighbors_by_state(l, i, t), new, SUSCEPTIBLE, reps)
        tc_leg, _ = time_count(leg_count, leg, GE.STATE_SUSCEPTIBLE, reps)

        @printf "  [%-9s] get_neighbors!      new=%.3f ms  legacy=%.3f ms  (%.2fx)\n" b 1e3tn_new 1e3tn_leg tn_new/tn_leg
        @printf "  [%-9s] count_neighbors     new=%.3f ms  legacy=%.3f ms  (%.2fx)\n" b 1e3tc_new 1e3tc_leg tc_new/tc_leg
    end

    println("\nFULL SIR RUN (300x300 absorbing, seeded; min of $TRIALS)")
    time_sir(() -> create_square_lattice(300, 300, :absorbing))  # warmup
    best = Inf; st_new = 0
    for _ in 1:TRIALS
        t, st_new = time_sir(() -> create_square_lattice(300, 300, :absorbing))
        best = min(best, t)
    end
    @printf "  new HypercubicLattice{2}: %.1f ms  (%d steps)\n" 1e3best st_new
    println("="^70)
end

main()
