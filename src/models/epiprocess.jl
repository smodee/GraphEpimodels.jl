"""
Abstract epidemic process interface and common functionality.

This module defines the interface that all epidemic process implementations
must follow, along with shared utilities and default implementations.
"""

using Random

# Import graph interface (assumes graphs/graphs.jl is loaded)
# Note: AbstractEpidemicGraph, NodeState, etc. come from graphs.jl

# =============================================================================
# Abstract Process Types
# =============================================================================

"""
Abstract base type for all epidemic processes.

All epidemic process implementations must inherit from this type and 
implement the required interface methods.
"""
abstract type AbstractEpidemicProcess end

"""
Abstract type for SIR-like processes where nodes transition S → I → R.

Examples:
- ZIM: Infection by neighbors, removal by fighting back
- SIR: Infection by neighbors, spontaneous recovery  
- SIRS: SIR with possible reinfection
"""
abstract type SIRLikeProcess <: AbstractEpidemicProcess end

# =============================================================================
# Required Interface Methods (must be implemented by all process types)
# =============================================================================

"""
Get the graph on which this process operates.

Default reads the `graph` field (the convention every process here follows); a
type that stores its graph elsewhere should override this method.
"""
get_graph(process::AbstractEpidemicProcess)::AbstractEpidemicGraph = process.graph

"""
Execute one simulation step.

# Arguments  
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Float64`: Time increment for this step (Inf if no events possible)
"""
function step!(process::AbstractEpidemicProcess)::Float64
    error("step! must be implemented by concrete process type $(typeof(process))")
end

"""
Reset process to initial conditions.

# Arguments
- `process::AbstractEpidemicProcess`: The process to reset
- `initial_infected::Vector{Int}`: Nodes to start as infected
"""
function reset!(process::AbstractEpidemicProcess,
                initial_infected::Vector{Int};
                rng_seed::Union{Int, Nothing} = nothing)
    error("reset! must be implemented by concrete process type $(typeof(process))")
end

"""
Get current simulation time.

Default reads the `time` field; override if your type stores it elsewhere.
"""
@inline current_time(process::AbstractEpidemicProcess)::Float64 = process.time

"""
Get number of simulation steps executed.

Default reads the `steps` field; override if your type stores it elsewhere.
"""
@inline step_count(process::AbstractEpidemicProcess)::Int = process.steps

"""
Check if the process should continue (has active events).

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Bool`: true if process should continue
"""
function is_active(process::AbstractEpidemicProcess)::Bool
    error("is_active must be implemented by concrete process type $(typeof(process))")
end

# =============================================================================
# Optional Interface Methods (have default implementations)
# =============================================================================

"""
Get total rate of all possible events (for Gillespie algorithm).

Default implementation returns 1.0. Override for performance-critical applications.

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Float64`: Total event rate
"""
function get_total_rate(process::AbstractEpidemicProcess)::Float64
    return 1.0  # Default fallback
end

# =============================================================================
# Derived Functions (implemented using the interface)
# =============================================================================

"""
Get current simulation statistics.

Uses the interface methods to compute common statistics for any process type.

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Dict{Symbol, Any}`: Statistics dictionary
"""
function get_statistics(process::AbstractEpidemicProcess)::Dict{Symbol, Any}
    graph = get_graph(process)
    state_counts = count_states(graph)
    
    return Dict{Symbol, Any}(
        :time => current_time(process),
        :step_count => step_count(process),
        :susceptible => state_counts[SUSCEPTIBLE],
        :infected => state_counts[INFECTED],
        :removed => state_counts[REMOVED],
        :total_ever_infected => state_counts[INFECTED] + state_counts[REMOVED],
        :is_active => is_active(process),
        :total_rate => get_total_rate(process)
    )
end

"""
Check if infection has reached graph boundary.

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Bool`: true if infection has reached boundary
"""
function has_escaped(process::AbstractEpidemicProcess)::Bool
    graph = get_graph(process)
    boundary_nodes = get_boundary_nodes(graph)
    
    if isempty(boundary_nodes)
        return false  # No boundary concept for this graph type
    end
    
    states = node_states_raw(graph)
    infected_state = state_to_int(INFECTED)
    
    for node in boundary_nodes
        if states[node] == infected_state
            return true
        end
    end
    
    return false
end

"""
Get total number of nodes ever infected during this simulation.

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Int`: Total nodes that have been infected or removed
"""
function get_cluster_size(process::AbstractEpidemicProcess)::Int
    stats = get_statistics(process)
    return stats[:total_ever_infected]
end

"""
Short, human-readable process name for plot titles and color-scheme lookup, e.g.
`SIRProcess` -> "SIR". Strips a trailing "Process" from the type name.
"""
function _process_name(process::AbstractEpidemicProcess)::String
    name = string(nameof(typeof(process)))
    return endswith(name, "Process") ? name[1:end-7] : name
end

# =============================================================================
# Simulation Runner (works with any process type)
# =============================================================================

"""
Run complete simulation until stopping condition.

# Arguments
- `process::AbstractEpidemicProcess`: The process to run
- `max_time::Float64`: Maximum simulation time (default: Inf)
- `max_steps::Int`: Maximum number of steps (default: 1_000_000)
- `save_history::Bool`: Whether to save state snapshots (default: false)
- `history_interval::Int`: Steps between history saves (default: 100)

# Returns
- `Dict{Symbol, Any}`: Final simulation statistics and optional history

# Example
```julia
julia> zim = create_zim_process(100, 100, 2.0)
julia> results = run_simulation(zim; max_time=50.0)
julia> println("Final infected: ", results[:infected])
```
"""
function run_simulation(process::AbstractEpidemicProcess;
                       max_time::Float64 = Inf,
                       max_steps::Int = typemax(Int),
                       stop_on_escape::Bool = false,
                       save_history::Bool = false,
                       history_interval::Int = 100)::Dict{Symbol, Any}
    
    history = save_history ? Dict{Symbol, Any}[] : nothing
    
    while (current_time(process) < max_time && 
           step_count(process) < max_steps && 
           is_active(process))
        
        # Save history snapshot if requested
        if save_history && step_count(process) % history_interval == 0
            snapshot = get_statistics(process)
            if save_history
                # Save node states for this snapshot
                snapshot[:node_states] = copy(node_states_raw(get_graph(process)))
            end
            push!(history, snapshot)
        end
        
        # Execute one step
        dt = step!(process)

        # Stop when escaped if the option is switched on
        if stop_on_escape
            if has_escaped(process)
                break
            end
        end
        
        # Break if no more events possible
        if dt == Inf
            break
        end
    end
    
    # Get final statistics
    final_stats = get_statistics(process)
    
    # Add termination reason
    if current_time(process) >= max_time
        final_stats[:termination_reason] = :max_time_reached
    elseif step_count(process) >= max_steps
        final_stats[:termination_reason] = :max_steps_reached
    elseif !is_active(process)
        final_stats[:termination_reason] = :no_active_events
    elseif has_escaped(process)
        final_stats[:termination_reason] = :escaped
    else
        final_stats[:termination_reason] = :unknown
    end
    
    if save_history
        final_stats[:history] = history
    end
    
    return final_stats
end

# =============================================================================
# Process Validation Utilities (General)
# =============================================================================

"""
Validate initial infected node list.

# Arguments
- `nodes::Vector{Int}`: List of node indices
- `graph::AbstractEpidemicGraph`: The graph

# Returns
- `Vector{Int}`: Validated node array

# Throws
- `ArgumentError`: If node indices are invalid
"""
function validate_initial_infected(nodes::Vector{Int}, 
                                  graph::AbstractEpidemicGraph)::Vector{Int}
    n_nodes = num_nodes(graph)
    
    if isempty(nodes)
        throw(ArgumentError("Initial infected list cannot be empty"))
    end
    
    for node in nodes
        if node < 1 || node > n_nodes
            throw(ArgumentError("Node index $node out of range [1, $n_nodes]"))
        end
    end
    
    if length(unique(nodes)) != length(nodes)
        throw(ArgumentError("Duplicate nodes in initial infected list"))
    end

    return nodes
end

# =============================================================================
# Shared SIR-like Substrate
# =============================================================================
#
# Every concrete SIR-like model (ZIM, SIR, Maki-Thompson, Chase-Escape) shares the
# same Gillespie skeleton, reset prologue, parameter/statistics shape, and event
# bookkeeping. The helpers below hold that shared machinery so each model file
# implements only what genuinely differs: its rate formula (`get_total_rate`), its
# activity test (`is_active`), its event selection (`_fire_event!`), and its
# tracker seeding (`_clear_trackers!` plus the per-model reset body).

"""
One Gillespie step shared by every `SIRLikeProcess`: draw the exponential waiting
time from the total event rate, fire one event via the model's `_fire_event!`
hook, then advance the clock and step counter. Returns the time increment, or
`Inf` when the process is inactive or has zero total rate.

Reads the `rng`/`time`/`steps` fields (the SIR-like field convention). The waiting
time is drawn *before* `_fire_event!` so the RNG draw order matches the original
per-model steppers exactly.
"""
function step!(process::SIRLikeProcess)::Float64
    is_active(process) || return Inf
    total_rate = get_total_rate(process)
    total_rate > 0 || return Inf
    dt = randexp(process.rng) / total_rate
    _fire_event!(process, total_rate)
    process.time  += dt
    process.steps += 1
    return dt
end

"""
Select and apply one event for `process`, given the current `total_rate` (used to
normalize per-event-class probabilities). Implemented by each concrete process and
called only from the shared [`step!`](@ref), after the waiting time is drawn.
"""
function _fire_event! end

"""
Common reset prologue: zero the clock and step counter, optionally reseed the RNG,
clear the model's trackers/sets (via the `_clear_trackers!` hook), and reset every
node to SUSCEPTIBLE. Returns the raw state vector so the caller can place its seeds
and build its tracker weights.
"""
function _reset_prologue!(process::SIRLikeProcess; rng_seed::Union{Int, Nothing})
    process.time  = 0.0
    process.steps = 0
    rng_seed !== nothing && Random.seed!(process.rng, rng_seed)
    _clear_trackers!(process)
    states = node_states_raw(process.graph)
    fill!(states, state_to_int(SUSCEPTIBLE))
    return states
end

"""Clear a process's active-node tracker(s) and any node-id sets. Per-model hook."""
function _clear_trackers! end

"""
Validate epidemic rate parameters: each rate must be strictly positive, and very
large rates (> 1000) draw a numerical-stability warning. Each argument is a
`"name" => value` pair; the name appears in the error/warning text.
"""
function _validate_rates(rates::Pair{String, Float64}...)
    for (name, value) in rates
        value > 0.0 ||
            throw(ArgumentError("$name must be positive, got $value"))
    end
    if any(p -> last(p) > 1000.0, rates)
        desc = join(("$name=$value" for (name, value) in rates), ", ")
        @warn "Very large rates ($desc) may cause numerical issues"
    end
end

"""
Build a model statistics dict: start from the common [`get_statistics`](@ref), add
`:escaped`, then merge the model-specific entries passed as keyword arguments.
"""
function _augment_statistics(process::AbstractEpidemicProcess; kwargs...)
    stats = get_statistics(process)
    stats[:escaped] = has_escaped(process)
    for (key, value) in kwargs
        stats[key] = value
    end
    return stats
end

"""
Build a `width × height` square lattice and forward it (with the remaining
positional and keyword arguments) to `create_fn`. Backs every model's
`create_*_process(width, height, ...)` convenience overload.
"""
function create_on_square_lattice(create_fn, width::Int, height::Int, args...;
                                  boundary::Symbol = :absorbing, kwargs...)
    lattice = create_square_lattice(width, height, boundary)
    return create_fn(lattice, args...; kwargs...)
end

"""
From `neighbors` (a node's neighbor list) collect those currently SUSCEPTIBLE into
`buf` (cleared first), then return one of them chosen uniformly at random, or `0`
if there are none. `buf` lets the caller control allocation (a reused scratch
buffer or a fresh `Int[]`); the candidate order matches `neighbors`, so the RNG
draw is identical regardless of which buffer is used.
"""
function _random_susceptible_neighbor(neighbors, states::Vector{Int8},
                                      buf::Vector{Int}, rng::AbstractRNG)::Int
    susceptible_state = state_to_int(SUSCEPTIBLE)
    empty!(buf)
    @inbounds for nb in neighbors
        states[nb] == susceptible_state && push!(buf, nb)
    end
    isempty(buf) && return 0
    return rand(rng, buf)
end

"""
Single-tracker S→I maintenance shared by ZIM and SIR. After `new_infected` (was S)
becomes I via `attacker`, update `tracker`: add `new_infected` with its
susceptible-neighbor count, decrement `attacker` by one (it lost a susceptible
neighbor), and decrement every *other* infected neighbor of `new_infected` by one.
`neighbors` is `new_infected`'s neighbor list (fetched by the caller, so it
controls buffer reuse vs. allocation).
"""
function _track_si_infection!(tracker::DictActiveTracker, graph, neighbors,
                              attacker::Int, new_infected::Int)
    add_active_node!(tracker, new_infected,
                     count_neighbors_by_state(graph, new_infected, SUSCEPTIBLE))
    current = get(tracker.active_nodes, attacker, 0)
    update_active_node!(tracker, attacker, current - 1)

    states = node_states_raw(graph)
    infected_state = state_to_int(INFECTED)
    for neighbor in neighbors
        if states[neighbor] == infected_state && neighbor != attacker
            c = get(tracker.active_nodes, neighbor, 0)
            c > 0 && update_active_node!(tracker, neighbor, c - 1)
        end
    end
    return nothing
end