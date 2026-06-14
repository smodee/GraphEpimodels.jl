"""
Animated visualization of epidemic processes on square lattices.

The performance-optimized Gillespie loop (`step!`) is left untouched. Instead, a
dedicated runner replays it while capturing lightweight `Vector{Int8}` state
snapshots according to a pluggable *frame sampler*, then renders the snapshots
into an animated GIF using the existing `LatticeVisualizer` heatmap.

Two sampling regimes (see `FrameSampler`):
- `TimeInterval(dt)` — equal simulation-time spacing; faithful temporal playback,
  the right choice for large lattices (Gillespie `dt` varies per step, so
  equal-time frames preserve the true speed of spread).
- `EveryStep()`      — one frame per transition; best for small lattices where
  every single event is interesting.

Output format follows the filename extension: `.gif` or `.mp4` both work via the
Makie/CairoMakie `record` backend. Works for any graph type — lattices render as
dual-tiling cells, general graphs as node-link diagrams — via `visualizer_for`.

# Example
```julia
using GraphEpimodels

# Small lattice — animate every transition
sir = create_sir_simulation(30, 30, 0.6, 1.0; initial_infected=:center, rng_seed=1)
animate_simulation(sir; sampler=EveryStep(), color_scheme=:sir, filename="sir_small.gif")

# Large lattice — equal-time sampling for faithful playback
big = create_sir_simulation(200, 200, 0.6, 1.0; initial_infected=:center, rng_seed=1)
animate_simulation(big; sampler=TimeInterval(0.5), max_time=40.0, filename="sir_large.mp4")
```
"""

using CairoMakie

# =============================================================================
# Frame Samplers — decide *when* a snapshot is captured
# =============================================================================

"""
Abstract base type for frame-sampling policies used during recording.

A sampler answers a single question via `should_capture`: given the current
simulation time/step and the time/step of the previously captured frame, should
a new frame be captured now?
"""
abstract type FrameSampler end

"""
Capture one frame per transition (every `step!`).

Best for small lattices, where seeing every individual event is useful. Frame
count grows with the number of steps, so avoid on large lattices.
"""
struct EveryStep <: FrameSampler end

"""
Capture a frame every `n` steps.

A middle ground between `EveryStep` and `TimeInterval` when you want to thin a
per-transition animation by a fixed stride.
"""
struct EveryNSteps <: FrameSampler
    n::Int

    function EveryNSteps(n::Int)
        n >= 1 || throw(ArgumentError("EveryNSteps requires n >= 1, got $n"))
        new(n)
    end
end

"""
Sample-and-hold on a fixed time grid `0, dt, 2dt, ...`.

Emits exactly one frame per grid point, each showing the state held at that grid
time, so frames are equally spaced in *simulation time* (not steps). Played back
at a constant fps this reflects the true speed of the process — and a slow period
(a single large Gillespie `dt` spanning several grid points) correctly shows as a
held/paused image rather than being skipped over.

This is the right choice for large lattices: the number of frames is bounded by
`total_time / dt` regardless of how many Gillespie steps occur, and state
snapshots are taken only at grid crossings, never on every step.
"""
struct TimeInterval <: FrameSampler
    dt::Float64

    function TimeInterval(dt::Real)
        dt > 0 || throw(ArgumentError("TimeInterval requires dt > 0, got $dt"))
        new(Float64(dt))
    end
end

# Per-step capture logic, dispatched on sampler type. `push_state!(time, step)`
# snapshots the current graph state into the recording. Returns the (possibly
# updated) time-grid cursor `next_grid`, which only `TimeInterval` advances.

# One frame per transition.
function _capture_after_step!(::EveryStep, push_state!, t::Float64, s::Int, next_grid::Float64)
    push_state!(t, s)
    return next_grid
end

# One frame every `n` steps.
function _capture_after_step!(samp::EveryNSteps, push_state!, t::Float64, s::Int, next_grid::Float64)
    if s % samp.n == 0
        push_state!(t, s)
    end
    return next_grid
end

# Sample-and-hold: emit one frame per grid point reached by this step, each
# stamped with its grid time. A single large `dt` that spans several grid points
# therefore produces several (held) frames, giving uniform spacing in simulation
# time. Snapshots are taken only here, at crossings — not on every step. (The held
# state is the post-step state, so a frame can be at most one event ahead of the
# exact held value: a single-node difference, visually negligible.)
function _capture_after_step!(samp::TimeInterval, push_state!, t::Float64, s::Int, next_grid::Float64)
    while next_grid <= t
        push_state!(next_grid, s)
        next_grid += samp.dt
    end
    return next_grid
end

# =============================================================================
# Simulation Recording — captured frames + metadata
# =============================================================================

"""
A recorded simulation: a sequence of lattice state snapshots with metadata.

Frames are raw `Int8` state vectors (1 byte per node), so memory stays modest;
time-based sampling keeps the frame count bounded for large lattices.

# Fields
- `graph::AbstractEpidemicGraph`: The graph (kept for dimensions / rendering)
- `frames::Vector{Vector{Int8}}`: One node-state snapshot per frame
- `times::Vector{Float64}`: Simulation time of each frame
- `steps::Vector{Int}`: Step count at each frame
- `counts::Vector{NTuple{3, Int}}`: `(S, I, R)` counts per frame (for titles)
- `process_name::String`: Short process name (e.g. "SIR") for titles
"""
struct SimulationRecording
    graph::AbstractEpidemicGraph
    frames::Vector{Vector{Int8}}
    times::Vector{Float64}
    steps::Vector{Int}
    counts::Vector{NTuple{3, Int}}
    process_name::String
end

"""Number of recorded frames."""
num_frames(rec::SimulationRecording)::Int = length(rec.frames)

# =============================================================================
# Internal Helpers
# =============================================================================

"""Count `(S, I, R)` directly from a raw Int8 snapshot."""
function _count_states_raw(states::Vector{Int8})::NTuple{3, Int}
    nS = 0; nI = 0; nR = 0
    @inbounds for v in states
        if v == STATE_SUSCEPTIBLE
            nS += 1
        elseif v == STATE_INFECTED
            nI += 1
        else
            nR += 1
        end
    end
    return (nS, nI, nR)
end

"""Short process name for frame titles, e.g. `SIRProcess` -> "SIR"."""
function _process_name(process::AbstractEpidemicProcess)::String
    name = string(nameof(typeof(process)))
    return endswith(name, "Process") ? name[1:end-7] : name
end

"""Build the per-frame title string from recorded metadata."""
function _frame_title(rec::SimulationRecording, idx::Int)::String
    t = rec.times[idx]
    s = rec.steps[idx]
    (_, nI, nR) = rec.counts[idx]
    return "$(rec.process_name)  (t=$(round(t, digits=2)), step=$s, I=$nI, R=$nR)"
end

# =============================================================================
# Recorder — replays the Gillespie loop and captures frames
# =============================================================================

"""
Run a process while recording state snapshots according to `sampler`.

Mirrors the loop and termination conditions of `run_simulation`, but instead of
the heavyweight `save_history` it captures lightweight `Int8` snapshots. The
initial state (t=0) and the final state are always captured, so the animation
starts and ends on the true endpoints.

Runs from the process's *current* state (like `run_simulation`); pass a freshly
created `create_*_simulation(...)` process for a clean run.

# Arguments
- `process::AbstractEpidemicProcess`: The process to run and record
- `sampler::FrameSampler`: When to capture frames (default: `TimeInterval(1.0)`)
- `max_time::Float64`: Stop at this simulation time (default: `Inf`)
- `max_steps::Int`: Stop after this many steps (default: `typemax(Int)`)
- `stop_on_escape::Bool`: Stop once infection reaches the boundary (default: false)

# Returns
- `SimulationRecording`
"""
function record_simulation(process::AbstractEpidemicProcess;
                           sampler::FrameSampler = TimeInterval(1.0),
                           max_time::Float64 = Inf,
                           max_steps::Int = typemax(Int),
                           stop_on_escape::Bool = false)::SimulationRecording
    graph = get_graph(process)

    frames = Vector{Vector{Int8}}()
    times  = Float64[]
    steps  = Int[]
    counts = NTuple{3, Int}[]

    # Snapshot the current graph state, stamped with the given (time, step).
    push_state! = function (t::Float64, s::Int)
        snapshot = copy(node_states_raw(graph))
        push!(frames, snapshot)
        push!(times, t)
        push!(steps, s)
        push!(counts, _count_states_raw(snapshot))
    end

    # Initial frame (t=0, step 0)
    push_state!(current_time(process), step_count(process))

    # Time-grid cursor (used only by TimeInterval): first grid point after t=0.
    next_grid = sampler isa TimeInterval ? sampler.dt : 0.0

    while (current_time(process) < max_time &&
           step_count(process) < max_steps &&
           is_active(process))

        dt = step!(process)

        next_grid = _capture_after_step!(sampler, push_state!,
                                         current_time(process), step_count(process), next_grid)

        # Stop when escaped if requested
        if stop_on_escape && has_escaped(process)
            break
        end

        # No more events possible
        if dt == Inf
            break
        end
    end

    # Always end on the true final state (avoid duplicating an already-captured frame)
    if isempty(steps) || steps[end] != step_count(process)
        push_state!(current_time(process), step_count(process))
    end

    return SimulationRecording(graph, frames, times, steps, counts, _process_name(process))
end

# =============================================================================
# Animation Builder
# =============================================================================

# Draw one recorded frame into an existing axis, dispatched on visualizer type.
# Reuses the same drawing cores as the static `render_frame`, so animation frames
# look identical to static snapshots.
_draw_frame!(ax, viz::LatticeVisualizer, graph, states; positions = nothing) =
    _draw_lattice!(ax, viz, graph, states)
_draw_frame!(ax, viz::NetworkVisualizer, graph, states; positions = nothing) =
    _draw_network!(ax, viz, graph, states; positions = positions)

"""Fixed drawing limits (with margin) so the view doesn't jitter between frames."""
function _frame_limits(positions::Matrix{Float64})
    xlo, xhi = extrema(@view positions[1, :])
    ylo, yhi = extrema(@view positions[2, :])
    mx = 0.05 * (xhi - xlo) + 1.0
    my = 0.05 * (yhi - ylo) + 1.0
    return (xlo - mx, xhi + mx, ylo - my, yhi + my)
end

"""
Render a recording to an animated GIF or MP4 (format from the extension).

Each frame is drawn with the same drawing core used by `visualize_state`, so
frames look identical to static snapshots. The visualizer is chosen by graph type
via `visualizer_for`, so this works for square / triangular / hexagonal lattices
and for general (adjacency) graphs alike.

# Arguments
- `rec::SimulationRecording`: The recording to animate
- `color_scheme::Symbol`: Color scheme (default: `:sir`)
- `fps::Int`: Frames per second of the output (default: 15)
- `filename::String`: Output path; `.gif` or `.mp4` (default: "simulation.gif")
- `figure_size::Tuple{Int, Int}`: Frame size in pixels (default: (600, 600))
- `show_boundary::Bool`: Outline the lattice boundary (lattices only; default: false)
- `show_grid::Bool`: Stroke cell outlines (cell lattices only; default: false)

# Returns
- `String`: The output filename.
"""
function animate_recording(rec::SimulationRecording;
                           color_scheme::Symbol = :sir,
                           fps::Int = 15,
                           filename::String = "simulation.gif",
                           figure_size::Tuple{Int, Int} = (600, 600),
                           show_boundary::Bool = false,
                           show_grid::Bool = false)
    graph = rec.graph
    viz = visualizer_for(graph; color_scheme = color_scheme, figure_size = figure_size)
    if viz isa LatticeVisualizer
        viz.show_boundary = show_boundary
        viz.show_grid = show_grid
    end

    # A general (node-link) graph gets a single fixed layout so nodes don't move
    # between frames; lattices use their intrinsic node positions.
    positions = viz isa NetworkVisualizer ? _resolve_positions(graph) :
                node_positions(graph)
    xlo, xhi, ylo, yhi = _frame_limits(positions)

    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    limits!(ax, xlo, xhi, ylo, yhi)

    layout_pos = viz isa NetworkVisualizer ? positions : nothing
    n = num_frames(rec)
    record(fig, filename, 1:n; framerate = fps) do idx
        empty!(ax)
        ax.title = _frame_title(rec, idx)
        _draw_frame!(ax, viz, graph, rec.frames[idx]; positions = layout_pos)
    end

    println("Animation saved to: $filename ($n frames, $fps fps)")
    return filename
end

# =============================================================================
# Convenience API — run, record, and save in one call
# =============================================================================

"""
Run a process and save an animated GIF of its evolution in one call.

Records the run with `record_simulation` and renders it with `animate_recording`.

Returns the `SimulationRecording` so the animation can be re-rendered at a
different fps / color scheme without re-simulating:
```julia
rec = animate_simulation(sir; filename="run.gif")
animate_recording(rec; fps=30, color_scheme=:medical, filename="run_fast.gif")
```

Runs from the process's *current* state (like `run_simulation`); pass a freshly
created `create_*_simulation(...)` process for a clean run.

# Arguments
- `process::AbstractEpidemicProcess`: The process to run and animate
- `sampler::FrameSampler`: When to capture frames (default: `TimeInterval(1.0)`)
- `max_time::Float64`: Stop at this simulation time (default: `Inf`)
- `max_steps::Int`: Stop after this many steps (default: `typemax(Int)`)
- `stop_on_escape::Bool`: Stop once infection reaches the boundary (default: false)
- `color_scheme::Symbol`: Color scheme (default: `:sir`)
- `fps::Int`: Frames per second of the output GIF (default: 15)
- `filename::String`: Output path (default: "simulation.gif")
- `figure_size::Tuple{Int, Int}`: Frame size in pixels (default: (600, 600))
- `show_boundary::Bool`: Highlight the lattice boundary (default: false)
- `show_grid::Bool`: Show grid lines between nodes (default: false)

# Returns
- `SimulationRecording`
"""
function animate_simulation(process::AbstractEpidemicProcess;
                            sampler::FrameSampler = TimeInterval(1.0),
                            max_time::Float64 = Inf,
                            max_steps::Int = typemax(Int),
                            stop_on_escape::Bool = false,
                            color_scheme::Symbol = :sir,
                            fps::Int = 15,
                            filename::String = "simulation.gif",
                            figure_size::Tuple{Int, Int} = (600, 600),
                            show_boundary::Bool = false,
                            show_grid::Bool = false)::SimulationRecording
    rec = record_simulation(process;
                            sampler = sampler,
                            max_time = max_time,
                            max_steps = max_steps,
                            stop_on_escape = stop_on_escape)

    animate_recording(rec;
                      color_scheme = color_scheme,
                      fps = fps,
                      filename = filename,
                      figure_size = figure_size,
                      show_boundary = show_boundary,
                      show_grid = show_grid)

    return rec
end
