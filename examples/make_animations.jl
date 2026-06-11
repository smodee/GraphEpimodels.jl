#!/usr/bin/env julia
"""
Generate example animations for each epidemic model in GraphEpimodels.jl.

Run from the project root:
    julia --project examples/make_animations.jl

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

println("Done. GIFs written to: $OUTDIR")
