#!/usr/bin/env julia
"""
Generate example animations for each epidemic model in GraphEpimodels.jl.

Run from the project root (the examples environment bundles CairoMakie, which
GraphEpimodels needs for plotting/animation via its package extension):
    julia --project=examples examples/make_animations.jl

Produces one GIF per model in this directory (examples/). Each animation uses
`TimeInterval` (sample-and-hold) sampling so playback is faithful to simulation
time. To keep a consistent frame count across models with very different natural
runtimes, we measure the end time with one run, then animate an identical
seeded run with dt = end_time / TARGET_FRAMES.

Note: each process is seeded with a small *patch* of nodes at the center rather
than a single node. A single supercritical seed still goes extinct with
non-negligible probability; a patch makes a full, spreading run essentially
certain, so the examples are reliably interesting.
"""

using GraphEpimodels
using CairoMakie   # activates the plotting/animation extension

const OUTDIR = @__DIR__
const TARGET_FRAMES = 120
const FPS = 24
const L = 121          # lattice side (odd -> well-defined center)
const SEED = 7

"""Node indices of the (2r+1)x(2r+1) square block centered on an L x L lattice.

Uses the lattice's column-major indexing: index = col + (row-1)*L (height == L).
"""
function center_patch(L::Int, r::Int)
    c = (L + 1) ÷ 2
    return [col + (row - 1) * L for row in (c - r):(c + r) for col in (c - r):(c + r)]
end

"""
Run `build()` twice with the same seed: once to measure the natural end time,
then once to record/animate at a dt that yields ~TARGET_FRAMES frames.
`build` must be a zero-arg closure returning a fresh, identically-seeded process.
"""
function animate_model(name, build; color_scheme, filename)
    # Measure natural end time on a throwaway, identically-seeded run.
    T = run_simulation(build())[:time]

    sampler = T <= 0 ? EveryStep() : TimeInterval(T / TARGET_FRAMES)

    path = joinpath(OUTDIR, filename)
    rec = animate_simulation(build();
                             sampler = sampler,
                             color_scheme = color_scheme,
                             fps = FPS,
                             filename = path)
    (_, nI, nR) = rec.counts[end]
    println("  $name: end_time=$(round(T, digits=2)), frames=$(num_frames(rec)), " *
            "final (I,R)=($nI,$nR)  ->  $filename")
    return rec
end

println("Generating example animations (lattice $L x $L) ...")

patch = center_patch(L, 2)   # 5x5 = 25-node center seed (robust takeoff)

# 1. SIR — infection (red) sweeps out leaving recovered (gray) behind.
animate_model("SIR",
    () -> create_sir_simulation(L, L, 3.0, 1.0; initial_infected=patch, rng_seed=SEED);
    color_scheme = :sir, filename = "sir.gif")

# 2. ZIM (Zombie Infection Model) — zombies (green) spread, killed (red) behind.
animate_model("ZIM",
    () -> create_zim_simulation(L, L, 3.0, 1.0; initial_infected=patch, rng_seed=SEED);
    color_scheme = :zim, filename = "zim.gif")

# 3. Maki-Thompson rumor spreading — spreaders (orange) advance, stiflers (purple) behind.
animate_model("Maki-Thompson",
    () -> create_maki_thompson_simulation(L, L, 3.0, 1.0; initial_infected=patch, rng_seed=SEED);
    color_scheme = :medical, filename = "maki_thompson.gif")

# 4. Chase-escape (predator-prey) — prey (red) escapes outward, predator (blue) chases from within.
#    Needs lambda > mu so prey can outrun the predator; ghost seeds the chase from the center patch.
animate_model("Chase-escape",
    () -> create_chase_escape_simulation(L, L, 3.0, 1.0; ghost=true, initial_red=patch, rng_seed=SEED);
    color_scheme = :chaseescape, filename = "chase_escape.gif")

# =============================================================================
# SIR on other graph topologies (triangular / hexagonal lattices, general graph)
# =============================================================================
#
# These reuse the same animation machinery via `visualizer_for`: lattices render
# as dual-tiling cells (triangular -> hexagons, hexagonal -> triangles), and a
# general adjacency graph renders as a node-link diagram with a force-directed
# layout.

"""Center node plus its neighbors, as a robust seed for an arbitrary graph."""
function center_seed(g)
    c = (num_nodes(g) + 1) ÷ 2
    return unique(vcat(c, get_neighbors(g, c)))
end

"""Build a fresh, identically-seeded SIR process on `graph` seeded at `seeds`."""
function sir_builder(graph_factory, seed_fn; β = 3.0, γ = 1.0)
    return function ()
        g = graph_factory()
        p = SIRProcess(g, β, γ; rng = Random.MersenneTwister(SEED))
        reset!(p, seed_fn(g); rng_seed = SEED)
        return p
    end
end

# 5. SIR on a triangular lattice (6-neighbor) — drawn with hexagonal cells.
animate_model("SIR (triangular)",
    sir_builder(() -> create_triangular_lattice(45, 45), center_seed);
    color_scheme = :sir, filename = "sir_triangular.gif")

# 6. SIR on a hexagonal/honeycomb lattice (3-neighbor) — drawn with triangular cells.
animate_model("SIR (hexagonal)",
    sir_builder(() -> create_hexagonal_lattice(45, 45), center_seed);
    color_scheme = :sir, filename = "sir_hexagonal.gif")

# 7. SIR on a general graph (Erdős–Rényi) — node-link diagram, spring layout.
animate_model("SIR (network)",
    sir_builder(() -> create_gnp(80, 0.06; rng = Random.MersenneTwister(SEED)),
                g -> [argmax([length(get_neighbors(g, i)) for i in 1:num_nodes(g)])]);
    color_scheme = :sir, filename = "sir_network.gif")

println("Done. GIFs written to: $OUTDIR")
