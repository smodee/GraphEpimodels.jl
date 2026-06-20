### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e9b00000-0000-0000-0000-000000000024
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    using GraphEpimodels, PlutoUI
    # Import only the CairoMakie names the notebook uses directly, so its `Slider`
    # (a Makie widget) doesn't collide with PlutoUI's `Slider`. Loading CairoMakie
    # this way still runs its __init__ (backend activation) and triggers the
    # GraphEpimodels CairoMakie extension.
    using CairoMakie: Figure, Axis, lines!, axislegend
    import Random, NetworkLayout, Base64, Printf
end

# ╔═╡ e9b00000-0000-0000-0000-000000000023
begin
    include(joinpath(@__DIR__, "engine.jl"))
    ENGINE_READY = true
end

# ╔═╡ e9b00000-0000-0000-0000-000000000001
md"""
# GraphEpimodels — Interactive Explorer

Choose a model, a graph, and parameters; press **▶ Run simulation**; then play,
pause, scrub, re-seed, and export the animation. No coding required.
"""

# ╔═╡ e9b00000-0000-0000-0000-000000000002
md"""
## 🦠 Setup

**Model** $(@bind model Select(["SIR", "ZIM", "Maki-Thompson", "Chase-Escape"]))
   **Graph** $(@bind graph_family Select(["Square lattice", "Triangular lattice", "Hexagonal lattice", "Cube lattice", "Complete graph", "Path", "Cycle", "Star", "Regular tree", "d-ary tree", "Erdos-Renyi"]))
"""

# ╔═╡ e9b00000-0000-0000-0000-000000000003
@bind gsize PlutoUI.combine() do Child
    f = graph_family
    if f == "Square lattice"
        md"""
        width $(Child("width", Slider(5:5:200, default = 60, show_value = true)))
        height $(Child("height", Slider(5:5:200, default = 60, show_value = true)))
        boundary $(Child("boundary", Select([:absorbing, :periodic])))
        """
    elseif f in ("Triangular lattice", "Hexagonal lattice")
        md"""
        width $(Child("width", Slider(5:5:200, default = 50, show_value = true)))
        height $(Child("height", Slider(5:5:200, default = 50, show_value = true)))
        """
    elseif f == "Cube lattice"
        md"""
        width $(Child("width", Slider(3:1:40, default = 12, show_value = true)))
        height $(Child("height", Slider(3:1:40, default = 12, show_value = true)))
        depth $(Child("depth", Slider(3:1:40, default = 12, show_value = true)))
        boundary $(Child("boundary", Select([:absorbing, :periodic])))
        """
    elseif f in ("Complete graph", "Path", "Cycle", "Star")
        md"""nodes n $(Child("n", Slider(3:1:300, default = 30, show_value = true)))"""
    elseif f == "Regular tree"
        md"""
        degree d $(Child("degree", Slider(2:1:6, default = 3, show_value = true)))
        height $(Child("height", Slider(1:1:8, default = 5, show_value = true)))
        """
    elseif f == "d-ary tree"
        md"""
        branching $(Child("branching", Slider(2:1:6, default = 2, show_value = true)))
        height $(Child("height", Slider(1:1:8, default = 6, show_value = true)))
        """
    else  # Erdos-Renyi
        md"""
        nodes n $(Child("n", Slider(10:5:500, default = 80, show_value = true)))
        edge prob p $(Child("p", Slider(0.0:0.005:0.3, default = 0.06, show_value = true)))
        """
    end
end

# ╔═╡ e9b00000-0000-0000-0000-000000000004
@bind mparams PlutoUI.combine() do Child
    if model == "SIR"
        md"""
        β (infect) $(Child("beta", Slider(0.1:0.1:6.0, default = 3.0, show_value = true)))
        γ (recover) $(Child("gamma", Slider(0.1:0.1:6.0, default = 1.0, show_value = true)))
        """
    elseif model == "ZIM"
        md"""
        λ (bite) $(Child("lambda", Slider(0.1:0.1:6.0, default = 3.0, show_value = true)))
        μ (fight) $(Child("mu", Slider(0.1:0.1:6.0, default = 1.0, show_value = true)))
        """
    elseif model == "Maki-Thompson"
        md"""
        α (spread) $(Child("alpha", Slider(0.1:0.1:6.0, default = 3.0, show_value = true)))
        β (stifle) $(Child("beta", Slider(0.1:0.1:6.0, default = 1.0, show_value = true)))
        stifler contact $(Child("stifler", CheckBox(default = true)))
        """
    else  # Chase-Escape
        md"""
        λ (prey) $(Child("lambda", Slider(0.1:0.1:6.0, default = 3.0, show_value = true)))
        μ (predator) $(Child("mu", Slider(0.1:0.1:6.0, default = 1.0, show_value = true)))
        ghost start $(Child("ghost", CheckBox(default = true)))
        """
    end
end

# ╔═╡ e9b00000-0000-0000-0000-000000000005
md"""
**Initial** $(@bind init_kind Select(["Center", "Center patch", "Random"]))
   patch radius $(@bind patch_r Slider(0:1:6, default = 2, show_value = true))
   **Seed** $(@bind seed_field NumberField(1:1_000_000_000, default = 1))
   $(@bind reroll CounterButton("🎲 New seed"))
   $(@bind go CounterButton("▶ Run simulation"))
"""

# ╔═╡ e9b00000-0000-0000-0000-000000000006
md"""
**Time** $(@bind time_model Select(["Continuous", "Discrete"]))
   **Play time** $(@bind target_time Slider(3:1:30, default = 10, show_value = true)) s
   stop on escape $(@bind stop_escape CheckBox(default = false))
"""

# ╔═╡ e9b00000-0000-0000-0000-000000000007
md"""**Colours** $(@bind scheme Select(string.(available_color_schemes()), default = string(default_color_scheme(model))))"""

# ╔═╡ e9b00000-0000-0000-0000-000000000008
md"""
**View** $(@bind dim_choice Select(["2D", "3D"]))
   boundary $(@bind show_boundary CheckBox(default = false))
   grid $(@bind show_grid CheckBox(default = false))
   turntable (3D) $(@bind turntable CheckBox(default = false))
   $(@bind rerender CounterButton("🎨 Re-render"))
"""

# ╔═╡ e9b00000-0000-0000-0000-000000000013
md"""
### 💾 Export

format $(@bind exp_fmt Select([".gif", ".mp4"]))
   size $(@bind fig_px Select([400 => "400 px", 600 => "600 px", 800 => "800 px", 1000 => "1000 px"], default = 600))
   transparent $(@bind transparent CheckBox(default = false))
   $(@bind do_export CounterButton("Render & export"))
"""

# ╔═╡ e9b00000-0000-0000-0000-000000000015
md"""
---
### Notes & limitations
- **Sampling is automatic:** pick a **time model** (continuous = equal sim-time
  spacing, discrete = equal event spacing) and a **play time**; the run is sampled
  so the clip lasts about that long. Extinct-early runs play briefly with a hint.
- The in-notebook **player runs in your browser**, so it plays smoothly. The preview
  always renders on a white background (so it's readable in any theme); **size** and
  **transparent** under Export affect the downloaded GIF/MP4 only.
- Model / graph / parameter / seed controls take effect on **▶ Run simulation** (or
  **🎲 New seed**). Appearance controls (colours, view, turntable) only take effect
  on **🎨 Re-render** — change several, then re-render once.
- **3D** (cube / tree / star / complete): tick **turntable** to rotate the camera
  across the clip, in both the preview and the export.
"""

# ╔═╡ e9b00000-0000-0000-0000-000000000016
md"""
---
## ⚙️ Implementation (engine)

The cells below stage the controls, build + record the run (auto-sampled), render
the preview frames, and assemble the player. They reuse the package's public API
and the `render_frame` drawing core. See `engine.jl` (also used by `smoke_test.jl`).
"""

# ╔═╡ e9b00000-0000-0000-0000-000000000020
cfg = (
    model        = model,
    graph_family = graph_family,
    gsize        = gsize,
    mparams      = mparams,
    init_kind    = init_kind,
    patch_r      = patch_r,
    time_model   = time_model,
    target_time  = Float64(target_time),
    stop_escape  = stop_escape,
    seed         = seed_field,
)

# ╔═╡ e9b00000-0000-0000-0000-000000000022
begin
    const STAGED = Ref{Any}(nothing)        # latest staged sim config (read on Run)
    const SEED_STATE = Ref(0)               # last seen "New seed" counter
    const SEED_USED = Ref(0)                # seed of the current recording
    const RENDER_CFG = Ref{Any}((scheme = :sir, dim = "2D", boundary = false,
                                 grid = false, turntable = false,
                                 target = 10.0))  # default appearance
    const LAST_RENDERED = Ref{Any}(nothing) # appearance of the current preview
    const EXPORT_STATE = Ref(0)             # last seen "Export" counter
    const EXPORT_OUT = Ref{Any}(nothing)
end

# ╔═╡ e9b00000-0000-0000-0000-000000000017
recording = let
    go; reroll                          # re-run only when a button is clicked
    ENGINE_READY                        # ensure engine.jl is included first
    cfg_now = STAGED[]
    if cfg_now === nothing || (go == 0 && reroll == 0)
        nothing                         # nothing simulated yet — press Run
    else
        seed = reroll > SEED_STATE[] ?
            (SEED_STATE[] = reroll; rand(1:1_000_000_000)) : cfg_now.seed
        SEED_USED[] = seed
        build_recording(cfg_now, seed)
    end
end

# ╔═╡ e9b00000-0000-0000-0000-000000000012
if recording === nothing
    md""
else
    let
        ts = recording.times
        S = [c[1] for c in recording.counts]
        I = [c[2] for c in recording.counts]
        R = [c[3] for c in recording.counts]
        fig = Figure(size = (600, 220))
        ax = Axis(fig[1, 1]; xlabel = "time", ylabel = "count")
        lines!(ax, ts, S; color = :steelblue, label = "S")
        lines!(ax, ts, I; color = :crimson,   label = "I")
        lines!(ax, ts, R; color = :gray,      label = "R")
        axislegend(ax; position = :rt)
        fig
    end
end

# ╔═╡ e9b00000-0000-0000-0000-000000000014
export_panel = let
    do_export
    if recording === nothing
        md"_Run a simulation first._"
    elseif do_export == EXPORT_STATE[]
        EXPORT_OUT[] === nothing ?
            md"_Press **Render & export** to produce a downloadable file (uses the current View settings)._" : EXPORT_OUT[]
    else
        EXPORT_STATE[] = do_export
        path = tempname() * exp_fmt
        g = recording.graph
        fps = export_fps(recording, Float64(target_time))
        if has_cells(g)
            animate_recording(recording; fps = fps, color_scheme = Symbol(scheme),
                              filename = path, figure_size = (fig_px, fig_px),
                              show_boundary = show_boundary, show_grid = show_grid,
                              transparent_background = transparent)
        else
            animate_recording(recording; fps = fps, color_scheme = Symbol(scheme),
                              filename = path, figure_size = (fig_px, fig_px),
                              dim = (dim_choice == "3D" ? 3 : 2), turntable = turntable)
        end
        EXPORT_OUT[] = DownloadButton(read(path), "simulation$(exp_fmt)")
        EXPORT_OUT[]
    end
end

# ╔═╡ e9b00000-0000-0000-0000-000000000018
begin
    # Stage the appearance settings (cheap) without triggering a re-render.
    # (transparent is export-only; the preview always renders opaque.)
    RENDER_CFG[] = (scheme = Symbol(scheme), dim = dim_choice,
                    boundary = show_boundary, grid = show_grid,
                    turntable = turntable, target = Float64(target_time))
    nothing
end

# ╔═╡ e9b00000-0000-0000-0000-000000000019
rendered = let
    recording; rerender; ENGINE_READY   # render on a new run or on Re-render
    rc = RENDER_CFG[]
    if recording === nothing || rc === nothing
        nothing
    else
        g = recording.graph
        viz = visualizer_for(g; color_scheme = rc.scheme, figure_size = (PREVIEW_PX, PREVIEW_PX))
        if viz isa LatticeVisualizer
            viz.show_boundary = rc.boundary
            viz.show_grid = rc.grid
        else
            viz.dim = rc.dim == "3D" ? 3 : 2
        end
        pos = has_cells(g) ? nothing : frame_positions(g, viz.dim)
        plan = preview_plan(recording, rc.target)
        frames = build_frame_cache(recording, viz, plan.indices;
                                   positions = pos, turntable = rc.turntable)
        LAST_RENDERED[] = rc
        (frames = frames, fps = plan.fps, trivial = plan.trivial, indices = plan.indices)
    end
end

# ╔═╡ e9b00000-0000-0000-0000-000000000009
let
    rendered    # re-run after each render so the hint clears
    cur = (scheme = Symbol(scheme), dim = dim_choice, boundary = show_boundary,
           grid = show_grid, turntable = turntable, target = Float64(target_time))
    if recording !== nothing && LAST_RENDERED[] !== nothing && cur != LAST_RENDERED[]
        md"⚠️ _Appearance changed — click **🎨 Re-render** to update the preview._"
    else
        md""
    end
end

# ╔═╡ e9b00000-0000-0000-0000-000000000010
if rendered === nothing
    md"### ▶ Set parameters above and press **Run simulation**"
else
    frame_player(rendered.frames, rendered.fps)   # client-side: smooth play / pause / scrub
end

# ╔═╡ e9b00000-0000-0000-0000-000000000011
if recording === nothing || rendered === nothing
    md""
else
    (nS, nI, nR) = recording.counts[end]
    hint = rendered.trivial ?
        "  ·  ⚠️ extinct early ($(ever_infected(recording)) ever infected) — try 🎲 New seed" : ""
    md"""
    **Final** t = $(round(recording.times[end], digits = 2))  ·  steps $(recording.steps[end])  ·  S=$nS, I=$nI, R=$nR  ·  captured $(num_frames(recording))  ·  preview $(length(rendered.indices)) @ $(round(rendered.fps, digits = 1)) fps  ·  seed $(SEED_USED[])$hint
    """
end

# ╔═╡ e9b00000-0000-0000-0000-000000000021
begin
    STAGED[] = cfg     # stage latest controls; does NOT trigger the recording cell
    nothing
end

# ╔═╡ Cell order:
# ╟─e9b00000-0000-0000-0000-000000000001
# ╟─e9b00000-0000-0000-0000-000000000002
# ╟─e9b00000-0000-0000-0000-000000000003
# ╟─e9b00000-0000-0000-0000-000000000004
# ╟─e9b00000-0000-0000-0000-000000000005
# ╟─e9b00000-0000-0000-0000-000000000006
# ╟─e9b00000-0000-0000-0000-000000000007
# ╟─e9b00000-0000-0000-0000-000000000008
# ╟─e9b00000-0000-0000-0000-000000000009
# ╟─e9b00000-0000-0000-0000-000000000010
# ╟─e9b00000-0000-0000-0000-000000000011
# ╟─e9b00000-0000-0000-0000-000000000012
# ╟─e9b00000-0000-0000-0000-000000000013
# ╟─e9b00000-0000-0000-0000-000000000014
# ╟─e9b00000-0000-0000-0000-000000000015
# ╟─e9b00000-0000-0000-0000-000000000016
# ╟─e9b00000-0000-0000-0000-000000000017
# ╟─e9b00000-0000-0000-0000-000000000018
# ╟─e9b00000-0000-0000-0000-000000000019
# ╟─e9b00000-0000-0000-0000-000000000020
# ╟─e9b00000-0000-0000-0000-000000000021
# ╟─e9b00000-0000-0000-0000-000000000022
# ╟─e9b00000-0000-0000-0000-000000000023
# ╟─e9b00000-0000-0000-0000-000000000024
