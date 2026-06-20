"""
Recording of epidemic processes for animated visualization.

The performance-optimized Gillespie loop (`step!`) is left untouched. Instead, a
dedicated runner (`record_simulation`) replays it while capturing lightweight
`Vector{Int8}` state snapshots according to a pluggable *frame sampler*. The
captured `SimulationRecording` is then rendered into a GIF/MP4 by
`animate_recording` / `animate_simulation`.

This file holds the backend-independent recording machinery (frame samplers,
`SimulationRecording`, `record_simulation`). The Makie rendering
(`animate_recording`, `animate_simulation`) lives in
ext/GraphEpimodelsCairoMakieExt.jl and loads with `using CairoMakie`.

Two sampling regimes (see `FrameSampler`):
- `TimeInterval(dt)` — equal simulation-time spacing; faithful temporal playback,
  the right choice for large lattices (Gillespie `dt` varies per step, so
  equal-time frames preserve the true speed of spread).
- `EveryStep()`      — one frame per transition; best for small lattices where
  every single event is interesting.

# Example
```julia
using GraphEpimodels, CairoMakie   # CairoMakie enables animate_*

# Small lattice — animate every transition
sir = create_sir_process(30, 30, 0.6, 1.0; initial_infected=:center, rng_seed=1)
animate_simulation(sir; sampler=EveryStep(), color_scheme=:sir, filename="sir_small.gif")

# Large lattice — equal-time sampling for faithful playback
big = create_sir_process(200, 200, 0.6, 1.0; initial_infected=:center, rng_seed=1)
animate_simulation(big; sampler=TimeInterval(0.5), max_time=40.0, filename="sir_large.mp4")
```
"""

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
# Adaptive single-pass recorder — bounded frames without a measurement pass
# =============================================================================
#
# `record_simulation` needs the run length up front to choose a sampler `dt` /
# stride, which forces a throwaway measurement run before the recording run. The
# adaptive recorder removes that second pass: it captures as the Gillespie loop
# runs and keeps the frame count in `[max_frames ÷ 2, max_frames]` by halving the
# rate whenever the buffer fills. Memory is bounded regardless of run length, the
# total number of snapshot copies is O(max_frames · log(run length)) (NOT one per
# event), and a surviving run still ends with ≥ `max_frames ÷ 2` frames.

"""
Keep every other frame (positions 1, 3, 5, …) across all four parallel buffers,
halving the frame count while preserving uniform spacing and the first frame.

Called on buffer overflow: for `:discrete` this doubles the effective step stride;
for `:continuous` (on the established time grid) it doubles `dt`.
"""
function _decimate_keep_every_other!(frames::Vector{Vector{Int8}}, times::Vector{Float64},
                                     steps::Vector{Int}, counts::Vector{NTuple{3, Int}})
    keep = 1:2:length(frames)
    keepat!(frames, keep)
    keepat!(times,  keep)
    keepat!(steps,  keep)
    keepat!(counts, keep)
    return nothing
end

"""
Resample per-event snapshots onto a uniform time grid of `n` points spanning
`[0, t_end]` by sample-and-hold: grid point `i` shows the latest captured frame
whose time is ≤ that grid time. Assumes `times` is sorted ascending.

This is the single conversion from "every event (irregular times)" to "equal
simulation-time spacing", shared by the two continuous-mode paths that need it:
the bootstrap hand-off at the first overflow, and a run that stops before the
bootstrap fills. Returns fresh `(frames, times, steps, counts)` vectors; frame
snapshots are shared by reference (they are never mutated after capture), so held
grid points cost no extra memory.
"""
function _resample_to_time_grid(frames::Vector{Vector{Int8}}, times::Vector{Float64},
                                steps::Vector{Int}, counts::Vector{NTuple{3, Int}},
                                n::Int, t_end::Float64)
    n = max(2, n)
    dt = t_end > 0 ? t_end / (n - 1) : 0.0
    src = length(times)
    out_frames = Vector{Vector{Int8}}(undef, n)
    out_times  = Vector{Float64}(undef, n)
    out_steps  = Vector{Int}(undef, n)
    out_counts = Vector{NTuple{3, Int}}(undef, n)
    j = 1
    @inbounds for i in 1:n
        g = (i - 1) * dt
        while j < src && times[j + 1] <= g
            j += 1
        end
        out_frames[i] = frames[j]
        out_times[i]  = g
        out_steps[i]  = steps[j]
        out_counts[i] = counts[j]
    end
    return out_frames, out_times, out_steps, out_counts
end

# Replace the buffers' contents in place with their resampling onto a uniform
# time grid (see `_resample_to_time_grid`).
function _replace_with_time_grid!(frames, times, steps, counts, n::Int, t_end::Float64)
    nf, nt, ns, nc = _resample_to_time_grid(frames, times, steps, counts, n, t_end)
    empty!(frames); append!(frames, nf)
    empty!(times);  append!(times,  nt)
    empty!(steps);  append!(steps,  ns)
    empty!(counts); append!(counts, nc)
    return nothing
end

"""
Record a process in a **single pass** at an automatically-bounded frame rate, so
no separate measurement run is needed to choose a sampler.

Frames are captured as the Gillespie loop runs; whenever the buffer reaches
`max_frames` the rate is halved (keeping every other frame), so the frame count
stays in `[max_frames ÷ 2, max_frames]` and memory is bounded regardless of run
length. A run that "survives" finishes with at least `max_frames ÷ 2` frames; a
run that dies early keeps every event it produced.

`time_model`:
- `:discrete` — equal *event* spacing. Captures every step; on overflow keeps every
  other frame and doubles the step stride (1 → 2 → 4 → …). Uniform in step count.
- `:continuous` — equal *simulation-time* spacing. Captures every step until the
  first overflow, then uses that measured span to lay down a uniform time grid
  (`dt = t / (max_frames ÷ 2 - 1)`) and switches to sample-and-hold on the grid,
  doubling `dt` on each further overflow. The initial rate is thus *measured*, not
  guessed — there is no heuristic. A run that stops before the first overflow is
  resampled onto a time grid at the end (`_resample_to_time_grid`).

Like [`record_simulation`](@ref) it runs from the process's *current* state and
always captures the initial (t=0) and final states. For a clean run, pass a
freshly created process.

# Arguments
- `process::AbstractEpidemicProcess`: process to run and record
- `time_model::Symbol`: `:continuous` (default) or `:discrete`
- `max_frames::Int`: buffer cap, halved on overflow (default 512)
- `max_time::Float64`: stop at this simulation time (default `Inf`)
- `max_steps::Int`: stop after this many steps (default `typemax(Int)`)
- `stop_on_escape::Bool`: stop once infection reaches the boundary (default false)

# Returns
- `SimulationRecording`
"""
function record_simulation_adaptive(process::AbstractEpidemicProcess;
                                    time_model::Symbol = :continuous,
                                    max_frames::Int = 512,
                                    max_time::Float64 = Inf,
                                    max_steps::Int = typemax(Int),
                                    stop_on_escape::Bool = false)::SimulationRecording
    time_model in (:continuous, :discrete) ||
        throw(ArgumentError("time_model must be :continuous or :discrete, got :$time_model"))
    max_frames >= 4 ||
        throw(ArgumentError("record_simulation_adaptive requires max_frames >= 4, got $max_frames"))

    graph  = get_graph(process)
    frames = Vector{Vector{Int8}}()
    times  = Float64[]
    steps  = Int[]
    counts = NTuple{3, Int}[]

    push_state! = function (t::Float64, s::Int)
        snapshot = copy(node_states_raw(graph))
        push!(frames, snapshot)
        push!(times, t)
        push!(steps, s)
        push!(counts, _count_states_raw(snapshot))
    end

    # Initial frame (t=0, step 0)
    push_state!(current_time(process), step_count(process))

    half = max_frames ÷ 2

    if time_model == :discrete
        stride = 1
        while (current_time(process) < max_time &&
               step_count(process) < max_steps &&
               is_active(process))
            d = step!(process)
            s = step_count(process)
            if s % stride == 0
                push_state!(current_time(process), s)
                if length(frames) >= max_frames
                    _decimate_keep_every_other!(frames, times, steps, counts)
                    stride *= 2
                end
            end
            stop_on_escape && has_escaped(process) && break
            d == Inf && break
        end
    else  # :continuous
        grid_active = false
        dt = 0.0
        next_grid = 0.0
        while (current_time(process) < max_time &&
               step_count(process) < max_steps &&
               is_active(process))
            d = step!(process)
            t = current_time(process)
            s = step_count(process)

            if !grid_active
                # Bootstrap: capture every step until the first overflow.
                push_state!(t, s)
                if length(frames) >= max_frames
                    t_boot = times[end]
                    if t_boot > 0
                        # Hand off to a uniform time grid measured from the bootstrap.
                        _replace_with_time_grid!(frames, times, steps, counts, half, t_boot)
                        dt = times[2] - times[1]
                        next_grid = times[end] + dt
                        grid_active = true
                    else
                        # Degenerate (all events still at t≈0): can't grid yet, just thin.
                        _decimate_keep_every_other!(frames, times, steps, counts)
                    end
                end
            else
                # Sample-and-hold on the time grid, decimating *inside* the emit loop
                # so a single large dt can't overflow past max_frames before we coarsen.
                while next_grid <= t
                    push_state!(next_grid, s)
                    next_grid += dt
                    if length(frames) >= max_frames
                        _decimate_keep_every_other!(frames, times, steps, counts)
                        dt *= 2
                        next_grid = times[end] + dt
                    end
                end
            end

            stop_on_escape && has_escaped(process) && break
            d == Inf && break
        end

        # Stopped before the bootstrap filled: still per-event, so convert to a time
        # grid now over the full elapsed span (same operation as the hand-off).
        if !grid_active && length(times) >= 2
            _replace_with_time_grid!(frames, times, steps, counts, length(times), times[end])
        end
    end

    # Always end on the true final state (avoid duplicating an already-captured one).
    if isempty(steps) || steps[end] != step_count(process)
        push_state!(current_time(process), step_count(process))
    end

    return SimulationRecording(graph, frames, times, steps, counts, _process_name(process))
end
