# Interactive Epidemic Explorer (Pluto pilot)

A self-contained [Pluto](https://plutojl.org) notebook GUI for `GraphEpimodels.jl`:
pick a model, a graph, and parameters from menus and sliders, press **▶ Run
simulation**, then play / pause / change speed / scrub / re-seed the animation and
export a GIF/MP4 — no coding required.

It reuses the package's public API and the `render_frame` drawing core unchanged
(CairoMakie backend); it does **not** modify the core package.

## Files

| File | Purpose |
|------|---------|
| `epidemics_explorer.jl` | The Pluto notebook (the app). |
| `engine.jl` | Non-UI logic (build graph/process, record, render helpers). `include`d by the notebook. |
| `smoke_test.jl` | Headless check of the engine across all models/graphs. |
| `Project.toml` / `Manifest.toml` | The notebook's environment (GraphEpimodels dev-linked + CairoMakie, PlutoUI, NetworkLayout). |

## First-time setup

From the repository root, instantiate the environment (this dev-links the parent
package and downloads CairoMakie/PlutoUI/NetworkLayout):

```julia
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

Optionally verify everything works without the UI:

```julia
julia --project=notebooks notebooks/smoke_test.jl
```

## Launch

Pluto is the *runtime* (it runs the notebook); it is not a dependency of the
notebook itself. Install it once in your global environment if you don't have it:

```julia
julia -e 'using Pkg; Pkg.add("Pluto")'
```

Then launch the explorer:

```julia
julia -e 'using Pluto; Pluto.run(notebook="notebooks/epidemics_explorer.jl")'
```

The notebook's first cell activates this folder's environment automatically
(`Pkg.activate(@__DIR__)`), so the local `GraphEpimodels` dev dependency resolves
without using Pluto's built-in package manager.

## Notes

- **Automatic sampling**: instead of fiddling with sampler/dt/fps, pick a **time
  model** (continuous = equal sim-time spacing, discrete = equal event spacing) and
  a **play time**; a quick measure-then-record pass samples the run so the clip
  lasts about that long. Extinct-early runs play briefly with a "try a new seed"
  hint instead of being stretched out.
- **Smooth in-notebook playback**: the preview frames are rendered once (CairoMakie)
  and animated by a small **client-side JS player** (play / pause / scrub), so it
  isn't capped by Pluto's per-tick round-trip. Previews render opaque at a fixed
  resolution and scale to fit; **size** and **transparent** (under Export) affect the
  downloaded GIF/MP4 only. Extinct-early runs play a short clip, not the full target.
- **Two-stage controls**: model / graph / parameter / seed changes take effect on
  **▶ Run simulation** (or **🎲 New seed**); appearance changes (colours, view,
  turntable) take effect on **🎨 Re-render** — change several, then re-render once.
  A "settings changed" hint appears when the preview is stale.
- **3D** graphs (cube / tree / star / complete): tick **turntable** to rotate the
  camera across the clip, in both the preview and the export.
- A natural next step is a WGLMakie/GLMakie app for fully interactive playback + 3D.
