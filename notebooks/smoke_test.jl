#!/usr/bin/env julia
"""
smoke_test.jl — headless validation of the Explorer engine (no Pluto UI).

Exercises exactly the code paths the notebook uses: it `include`s `engine.jl`
and, for a representative grid of models × graph families, auto-samples + records
a simulation, plans the preview, renders the preview frame cache, and builds the
client-side player — plus a couple of cases that export an animation. Catches
environment problems and wrong API usage before any manual UI testing.

Run from the repo root:
    julia --project=notebooks notebooks/smoke_test.jl
"""

import Pkg
Pkg.activate(@__DIR__)

using GraphEpimodels, CairoMakie
import Random, NetworkLayout, Base64, Printf

include(joinpath(@__DIR__, "engine.jl"))

# --- cfg builders mirroring the notebook's control defaults ------------------

function gsize_for(f)
    if f == "Square lattice"
        (width = 60, height = 60, boundary = :absorbing)
    elseif f in ("Triangular lattice", "Hexagonal lattice")
        (width = 45, height = 45)
    elseif f == "Cube lattice"
        (width = 10, height = 10, depth = 10, boundary = :absorbing)
    elseif f in ("Complete graph", "Path", "Cycle", "Star")
        (n = 40,)
    elseif f == "Regular tree"
        (degree = 3, height = 6)
    elseif f == "d-ary tree"
        (branching = 2, height = 7)
    elseif f == "Country graph"
        (;)   # country + layers come from cfg.country / cfg.country_edges, not gsize
    else  # Erdos-Renyi
        (n = 120, p = 0.04)
    end
end

function mparams_for(m)
    if m == "SIR"
        (beta = 3.0, gamma = 1.0)
    elseif m == "ZIM"
        (lambda = 3.0, mu = 1.0)
    elseif m == "Maki-Thompson"
        (alpha = 3.0, beta = 1.0, stifler = true)
    else  # Chase-Escape
        (lambda = 3.0, mu = 1.0, ghost = true)
    end
end

function make_cfg(model, family;
                  init_kind = "Center patch", patch_r = 2,
                  time_model = "Continuous", target_time = 10.0,
                  stop_escape = false, seed = 1,
                  country = "norway_mock", country_edges = Symbol[])
    (model = model, graph_family = family,
     gsize = gsize_for(family), mparams = mparams_for(model),
     init_kind = init_kind, patch_r = patch_r,
     time_model = time_model, target_time = target_time,
     stop_escape = stop_escape, seed = seed,
     country = country, country_edges = country_edges)
end

# Full notebook pipeline: record (single adaptive pass) → preview indices →
# render cache → build player.
function check(model, family; dim = 2, kwargs...)
    cfg = make_cfg(model, family; kwargs...)
    rec = build_recording(cfg, cfg.seed)
    @assert num_frames(rec) > 0 "no frames recorded for $model / $family"
    @assert num_frames(rec) <= MAX_FRAMES + 1 "frame cap exceeded for $model / $family ($(num_frames(rec)))"
    @assert issorted(rec.times) "times not nondecreasing for $model / $family"
    @assert issorted(rec.steps) "steps not nondecreasing for $model / $family"

    plan = preview_indices(rec)
    @assert !isempty(plan.indices) "empty preview plan for $model / $family"
    @assert length(plan.indices) <= PREVIEW_CAP "preview exceeds cap for $model / $family"
    @assert all(1 .<= plan.indices .<= num_frames(rec)) "preview indices out of range"

    g = rec.graph
    viz = visualizer_for(g; color_scheme = default_color_scheme(rec.process_name),
                         figure_size = (500, 500))
    pos = nothing
    if !has_cells(g)
        viz.dim = dim
        pos = frame_positions(g, dim)
    end

    t0 = time()
    cache = build_frame_cache(rec, viz, plan.indices; positions = pos)
    render_s = time() - t0
    @assert length(cache) == length(plan.indices)
    @assert all(p -> length(p) > 0 && p[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47], cache) "bad PNG"

    html = frame_player(cache, cfg.target_time)
    @assert occursin("data:image/png;base64", html.content) "player has no frames"

    println("  ok: $model on $family — captured $(num_frames(rec)), " *
            "preview $(length(plan.indices)) frames, " *
            "export @ $(export_fps(rec, cfg.target_time)) fps, " *
            "trivial=$(plan.trivial), render $(round(render_s, digits=1))s")
    return rec
end

println("== models on a square lattice ==")
for m in ("SIR", "ZIM", "Maki-Thompson", "Chase-Escape")
    check(m, "Square lattice")
end

println("== SIR across graph families ==")
for f in ("Triangular lattice", "Hexagonal lattice", "Cube lattice",
          "Complete graph", "Path", "Cycle", "Star",
          "Regular tree", "d-ary tree", "Erdos-Renyi")
    dim = f in ("Cube lattice",) ? 3 : 2
    check("SIR", f; dim = dim)
end

println("== country graph (GeoGraph) — full pipeline incl. basemap render ==")
check("SIR", "Country graph")
check("ZIM", "Country graph"; init_kind = "Center")
check("SIR", "Country graph"; country_edges = [:road])        # single-layer subset

println("== initial-condition variants ==")
check("SIR", "Square lattice"; init_kind = "Center")
check("SIR", "Square lattice"; init_kind = "Random")

println("== time-model + target-time variants ==")
check("SIR", "Square lattice"; time_model = "Discrete")
check("SIR", "Square lattice"; target_time = 3.0)
check("SIR", "Square lattice"; target_time = 16.0)

println("== centers are interior, not on the boundary ==")
for (name, g) in (("triangular", create_triangular_lattice(40, 40)),
                  ("hexagonal",  create_hexagonal_lattice(40, 40)),
                  ("tree",       create_regular_tree(3, 5)),
                  ("star",       create_star_graph(40)))
    c = center_node(g)
    bnd = Set(get_boundary_nodes(g))
    @assert !(c in bnd) "$name center $c is on the boundary"
    println("  ok: $name center = $c (interior)")
end

println("== extinct-early run is flagged trivial (subcritical β) ==")
let rec = build_recording(make_cfg("SIR", "Square lattice"; init_kind = "Center"), 3)
    # subcritical handled separately below; here just confirm the predicate runs
    println("  default-run trivial? ", is_trivial(rec))
end
let cfg = (model = "SIR", graph_family = "Square lattice",
           gsize = (width = 60, height = 60, boundary = :absorbing),
           mparams = (beta = 0.2, gamma = 1.0), init_kind = "Center", patch_r = 0,
           time_model = "Continuous", target_time = 10.0, stop_escape = false, seed = 1)
    rec = build_recording(cfg, 1)
    plan = preview_indices(rec)
    @assert plan.trivial "subcritical run should be flagged trivial"
    # A run that dies early keeps only its (few) real events — it is NOT inflated to
    # the survival floor; the client-side player holds the end frame to fill the time.
    @assert num_frames(rec) < MAX_FRAMES ÷ 2 "early-death run inflated past the survival floor ($(num_frames(rec)))"
    @assert all(1 .<= plan.indices .<= num_frames(rec)) "preview indices out of range"
    println("  ok: subcritical SIR flagged trivial (ever=$(ever_infected(rec)), " *
            "$(num_frames(rec)) frames, preview $(length(plan.indices)))")
end

println("== adaptive recorder: bounded frames + survival floor (high-event runs) ==")
for tm in (:continuous, :discrete)
    g = create_square_lattice(100, 100)
    p = create_sir_process(g, 5.0, 1.0; initial_infected = :center, rng_seed = 1)
    rec = record_simulation_adaptive(p; time_model = tm, max_frames = MAX_FRAMES)
    @assert num_frames(rec) <= MAX_FRAMES + 1 "frame cap exceeded ($tm): $(num_frames(rec))"
    @assert num_frames(rec) >= MAX_FRAMES ÷ 2 "survival floor missed ($tm): $(num_frames(rec))"
    @assert issorted(rec.times) "times not nondecreasing ($tm)"
    @assert issorted(rec.steps) "steps not nondecreasing ($tm)"
    if tm == :discrete && num_frames(rec) >= 3
        # Equal-event spacing: every frame but the appended final one sits on a
        # uniform step grid (decimation re-spaces the whole buffer each overflow).
        @assert allequal(diff(rec.steps[1:end - 1])) "discrete frames not equally step-spaced"
    end
    println("  ok: square(100×100) SIR $tm — $(num_frames(rec)) frames, " *
            "t_end=$(round(rec.times[end], digits = 2)), steps=$(rec.steps[end])")
end

println("== adaptive recorder: short run keeps every event, stays bounded ==")
let g = create_path_graph(30)
    p = create_sir_process(g, 3.0, 1.0; initial_infected = [15], rng_seed = 1)
    rec = record_simulation_adaptive(p; time_model = :discrete, max_frames = MAX_FRAMES)
    @assert num_frames(rec) <= MAX_FRAMES + 1
    @assert issorted(rec.steps) && issorted(rec.times)
    # The whole run is far shorter than the cap, so nothing was decimated.
    @assert num_frames(rec) < MAX_FRAMES "unexpectedly long path run ($(num_frames(rec)))"
    println("  ok: path(30) SIR discrete — $(num_frames(rec)) frames, steps=$(rec.steps[end])")
end

println("== square-lattice grid renders on the non-transparent path ==")
let g = create_square_lattice(20, 20)
    p = create_sir_process(g, 3.0, 1.0; initial_infected = :center, rng_seed = 1)
    run_simulation(p; max_steps = 500)
    plain = visualizer_for(g; color_scheme = :sir, figure_size = (300, 300))
    grid  = visualizer_for(g; color_scheme = :sir, figure_size = (300, 300)); grid.show_grid = true
    a = render_png(plain, g, node_states_raw(g))
    b = render_png(grid,  g, node_states_raw(g))
    @assert a != b "show_grid had no effect on the non-transparent square path"
    println("  ok: grid overlay changes the rendered square lattice")
end

println("== turntable rotates the 3D preview frames ==")
let cfg = make_cfg("SIR", "Regular tree"; target_time = 4.0)
    rec = build_recording(cfg, cfg.seed)
    g = rec.graph
    viz = visualizer_for(g; color_scheme = :sir, figure_size = (300, 300)); viz.dim = 3
    plan = preview_indices(rec)
    pos = frame_positions(g, 3)
    still = build_frame_cache(rec, viz, plan.indices; positions = pos, turntable = false)
    spin  = build_frame_cache(rec, viz, plan.indices; positions = pos, turntable = true)
    @assert still[end] != spin[end] "turntable did not rotate the 3D camera"
    println("  ok: turntable changes later 3D frames")
end

println("== caption width is stable across frames ==")
let rec = build_recording(make_cfg("SIR", "Square lattice"), 1)
    w1 = length(frame_title(rec, 1))
    w2 = length(frame_title(rec, num_frames(rec) ÷ 2))
    w3 = length(frame_title(rec, num_frames(rec)))
    @assert w1 == w2 == w3 "frame title length varies ($w1, $w2, $w3)"
    println("  ok: title length constant = $w1 chars")
end

println("== animation export (gif + mp4) ==")
let rec = check("SIR", "Square lattice")
    gif = tempname() * ".gif"
    animate_recording(rec; fps = export_fps(rec, 10.0), color_scheme = :sir,
                      filename = gif, figure_size = (300, 300))
    @assert isfile(gif) && filesize(gif) > 0 "gif not written"
    println("  ok: gif $(filesize(gif)) bytes")
end
let rec = check("SIR", "Regular tree"; dim = 3)
    mp4 = tempname() * ".mp4"
    animate_recording(rec; fps = export_fps(rec, 10.0), color_scheme = :sir,
                      filename = mp4, figure_size = (300, 300), dim = 3, turntable = true)
    @assert isfile(mp4) && filesize(mp4) > 0 "mp4 not written"
    println("  ok: mp4 $(filesize(mp4)) bytes")
end

println("\nALL SMOKE TESTS PASSED ✅")
