#!/usr/bin/env julia
"""
Demo script for issue #4 — transparent_background lattice plots.

Generates six comparison PNGs in examples/transparent_demo/:
  square_opaque.png        — square lattice, white background
  square_transparent.png   — square lattice, transparent background (fixed)
  tri_opaque.png           — triangular lattice, white background
  tri_transparent.png      — triangular lattice, transparent background
  hex_opaque.png           — hexagonal lattice, white background
  hex_transparent.png      — hexagonal lattice, transparent background

In each transparent PNG the susceptible cells carry alpha=0, so they are absent
from the image. Open in any viewer that shows alpha (e.g. browser, Preview on
macOS, IrfanView on Windows) to confirm susceptibles are truly transparent.

Run from the project root:
    julia --project examples/transparent_demo.jl
"""

using GraphEpimodels
using CairoMakie
using Random

const OUTDIR = joinpath(@__DIR__, "transparent_demo")
mkpath(OUTDIR)

const SEED = 42

# ---------------------------------------------------------------------------
# Helper: run SIR until a good fraction is infected/removed, then snapshot.
# ---------------------------------------------------------------------------

function run_to_mid(process; target_removed_frac = 0.15)
    n = num_nodes(get_graph(process))
    while is_active(process)
        step!(process)
        s = get_statistics(process)
        if s[:removed] >= target_removed_frac * n
            break
        end
    end
    return process
end

# ---------------------------------------------------------------------------
# Square lattice (40 x 40)
# ---------------------------------------------------------------------------

println("Square lattice ...")

sq = create_sir_simulation(40, 40, 4.0, 1.0; rng_seed = SEED)
run_to_mid(sq)

viz_sq = LatticeVisualizer(color_scheme = :sir, figure_size = (500, 500), show_grid = true)

fig = visualize_state(viz_sq, sq; transparent_background = false)
save(joinpath(OUTDIR, "square_opaque.png"), fig)

fig = visualize_state(viz_sq, sq; transparent_background = true)
save(joinpath(OUTDIR, "square_transparent.png"), fig)

println("  -> square_opaque.png, square_transparent.png")

# ---------------------------------------------------------------------------
# Triangular lattice (35 x 35)
# ---------------------------------------------------------------------------

println("Triangular lattice ...")

tri_graph = create_triangular_lattice(35, 35)
tri = SIRProcess(tri_graph, 4.0, 1.0; rng = Random.MersenneTwister(SEED))
reset!(tri, [num_nodes(tri_graph) ÷ 2]; rng_seed = SEED)
run_to_mid(tri)

viz_tri = LatticeVisualizer(color_scheme = :sir, figure_size = (500, 500), show_grid = true)

fig = visualize_state(viz_tri, tri; transparent_background = false)
save(joinpath(OUTDIR, "tri_opaque.png"), fig)

fig = visualize_state(viz_tri, tri; transparent_background = true)
save(joinpath(OUTDIR, "tri_transparent.png"), fig)

println("  -> tri_opaque.png, tri_transparent.png")

# ---------------------------------------------------------------------------
# Hexagonal lattice (35 x 35)
# ---------------------------------------------------------------------------

println("Hexagonal lattice ...")

hex_graph = create_hexagonal_lattice(35, 35)
hex = SIRProcess(hex_graph, 4.0, 1.0; rng = Random.MersenneTwister(SEED))
reset!(hex, [num_nodes(hex_graph) ÷ 2]; rng_seed = SEED)
run_to_mid(hex)

viz_hex = LatticeVisualizer(color_scheme = :sir, figure_size = (500, 500), show_grid = true)

fig = visualize_state(viz_hex, hex; transparent_background = false)
save(joinpath(OUTDIR, "hex_opaque.png"), fig)

fig = visualize_state(viz_hex, hex; transparent_background = true)
save(joinpath(OUTDIR, "hex_transparent.png"), fig)

println("  -> hex_opaque.png, hex_transparent.png")

println("\nAll plots written to: $OUTDIR")
println("In the *_transparent.png files the susceptible cells should be")
println("absent (alpha=0), leaving only infected (red) and removed (gray) visible.")
