"""
Benchmark for the `AdjacencyGraph` / `ErdosRenyiGraph` simulation hot path.

`bench/hypercubic_bench.jl` only exercises the lattice (implicit, on-demand
neighbours). This benchmark covers the *stored-adjacency* path instead, which is
the one that reads `AdjacencyGraph.node_degrees` — see issue #29 (deciding
whether that precomputed-degree field earns its keep).

It reports, as the noise-resistant minimum over `TRIALS`:
  1. `count_neighbors_by_state` swept over every node (the per-Gillespie-step hot
     path for adjacency-backed graphs; this is the reader the field optimizes),
  2. `get_node_degree` swept over every node (the off-hot-path reader),
  3. a full seeded ZIM run on an Erdős–Rényi graph (end-to-end), with its total
     allocation.

The script uses only the public API (no reference to `node_degrees`), so the
identical file runs against both the baseline (field present) and the candidate
(field removed).

Run:  julia --project=. bench/adjacency_bench.jl
"""

using GraphEpimodels
using Random, Printf

# Robust microbenchmark: report the MINIMUM elapsed time over `TRIALS` (filters
# GC pauses and scheduler jitter that inflate single-shot timings).
const TRIALS = 9

# Fixed Erdős–Rényi graph: n nodes, mean degree ≈ MEANDEG. Seeded so the graph
# (and therefore every run below) is identical across baseline/candidate.
const N       = 20_000
const MEANDEG = 8.0
build_graph() = create_gnp(N, MEANDEG / (N - 1); rng = MersenneTwister(2024))

function time_count(g, target, reps)
    n = num_nodes(g)
    best = Inf
    for _ in 1:TRIALS
        acc = 0
        t = @elapsed for _ in 1:reps, i in 1:n
            acc += count_neighbors_by_state(g, i, target)
        end
        acc < 0 && error("dead code eliminated")
        best = min(best, t)
    end
    return best
end

function time_degree(g, reps)
    n = num_nodes(g)
    best = Inf
    for _ in 1:TRIALS
        acc = 0
        t = @elapsed for _ in 1:reps, i in 1:n
            acc += get_node_degree(g, i)
        end
        acc == 0 && error("dead code eliminated")
        best = min(best, t)
    end
    return best
end

# Full seeded ZIM run on the ER graph. Seeds a fixed block of nodes (so the run is
# large and reproducible) and goes to completion. ZIM is chosen because its death
# handler re-counts neighbours, hammering count_neighbors_by_state.
const SEEDS = collect(1:25)
function time_zim(g)
    best = Inf; steps = 0; bytes = 0
    for _ in 1:TRIALS
        p = create_zim_process(g, 3.0, 1.0; initial_infected = SEEDS, rng_seed = 7)
        local s = 0
        stats = @timed while is_active(p) && s < 5_000_000
            step!(p); s += 1
        end
        best = min(best, stats.time); steps = s; bytes = stats.bytes
    end
    return best, steps, bytes
end

function main()
    g = build_graph()
    # Randomize states for the count sweep (mix of S/I/R).
    Random.seed!(11)
    set_node_states_raw!(g, rand(Int8.(0:2), num_nodes(g)))

    println("="^70)
    @printf "ErdosRenyi G(n=%d, mean_deg≈%.1f), min of %d trials\n" N MEANDEG TRIALS

    reps = 200
    time_count(g, SUSCEPTIBLE, 1); time_degree(g, 1)         # warmup
    tc = time_count(g, SUSCEPTIBLE, reps)
    td = time_degree(g, reps)
    @printf "  count_neighbors_by_state  %.3f ms / sweep  (%d sweeps)\n" 1e3tc/reps reps
    @printf "  get_node_degree           %.3f ms / sweep  (%d sweeps)\n" 1e3td/reps reps

    # Fresh graph for the run (the count sweep left states dirty).
    gz = build_graph()
    time_zim(gz)                                              # warmup
    t, steps, bytes = time_zim(gz)
    @printf "  full ZIM run              %.1f ms  (%d steps)  alloc=%.3f MB  %.2f B/step\n" 1e3t steps bytes/2^20 bytes/max(steps,1)
    println("="^70)
end

main()
