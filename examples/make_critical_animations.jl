#!/usr/bin/env julia
"""
Near-critical ZIM and SIR animations on a 200×200 square lattice.

Critical parameters:
  ZIM: λ_c ≈ 2.28 on Z² (contact process / Bezuidenhout–Grimmett).
  SIR: maps to bond percolation; β_c = γ on Z² (since p = β/(β+γ), p_c = 1/2).

For each model we search over RNG seeds until finding a run that escapes to the
boundary, then render it as a GIF with equal-time frame spacing.

Run from the project root:
    julia --project=examples examples/make_critical_animations.jl
"""

using GraphEpimodels
using CairoMakie

const L              = 200    # lattice side
const TARGET_FRAMES  = 150
const FPS            = 24
const MAX_ATTEMPTS   = 500    # seed search limit per model
const OUTDIR         = @__DIR__

# ZIM: slightly above critical (λ_c ≈ 2.28, μ = 1)
const ZIM_LAMBDA = 2.30
const ZIM_MU     = 1.0

# SIR: slightly above critical (β_c = γ = 1.0 on Z², since p_c = 1/2)
const SIR_BETA  = 1.20   # T = β/(β+γ) ≈ 0.545, above T_c ≈ 0.538 (Tomé & Ziff 2010)
const SIR_GAMMA = 1.0

"""
Try RNG seeds 1, 2, … until the process escapes to the lattice boundary.
Uses stop_on_escape=true so the simulation halts while a boundary node is still
INFECTED — for SIR, infected boundary nodes eventually recover (→ REMOVED), so
checking has_escaped only at the very end always returns false.
Returns the first winning seed, or errors after MAX_ATTEMPTS.
"""
function find_escaping_seed(process_factory, label)
    print("Searching for escaping $label seed")
    for seed in 1:MAX_ATTEMPTS
        p = process_factory(seed)
        run_simulation(p; stop_on_escape=true)
        if has_escaped(p)
            println(" → seed $seed")
            return seed
        end
        seed % 100 == 0 && print(".")
    end
    error("No escaping seed found in $MAX_ATTEMPTS attempts for $label. " *
          "Try increasing the rate slightly.")
end

"""
Animate a full run to completion, with transparent background and no title.
"""
function make_gif(label, process_factory, seed; color_scheme, filename)
    # Measure full end time
    p_measure = process_factory(seed)
    T = run_simulation(p_measure)[:time]

    dt = max(T / TARGET_FRAMES, 1e-3)

    p_anim = process_factory(seed)
    path   = joinpath(OUTDIR, filename)
    rec    = animate_simulation(p_anim;
                                sampler                = TimeInterval(dt),
                                color_scheme           = color_scheme,
                                fps                    = FPS,
                                filename               = path,
                                figure_size            = (800, 800),
                                show_title             = false,
                                transparent_background = true)
    (nS, nI, nR) = rec.counts[end]
    println("  $label: T=$(round(T, digits=1)), $(num_frames(rec)) frames, " *
            "final (S=$nS, I=$nI, R=$nR) → $filename")
    return rec
end

println("=== Near-critical animations, $(L)×$(L) lattice ===\n")

# ZIM — seed 1 escapes (found in previous run)
println("ZIM  λ=$(ZIM_LAMBDA)  μ=$(ZIM_MU)  seed=1")
make_gif("ZIM",
    seed -> create_zim_process(L, L, ZIM_LAMBDA, ZIM_MU;
                               initial_infected = :center,
                               rng_seed = seed),
    1;
    color_scheme = :zim,
    filename     = "zim_critical.gif")

println()

# SIR — search for escaping seed with new β
println("SIR  β=$(SIR_BETA)  γ=$(SIR_GAMMA)  (T=$(round(SIR_BETA/(SIR_BETA+SIR_GAMMA), digits=3)), T_c ≈ 0.538)")
sir_seed = find_escaping_seed(
    seed -> create_sir_process(L, L, SIR_BETA, SIR_GAMMA;
                               initial_infected = :center,
                               rng_seed = seed),
    "SIR")
make_gif("SIR",
    seed -> create_sir_process(L, L, SIR_BETA, SIR_GAMMA;
                               initial_infected = :center,
                               rng_seed = seed),
    sir_seed;
    color_scheme = :sir,
    filename     = "sir_critical.gif")

println("\nDone. GIFs written to: $OUTDIR")
