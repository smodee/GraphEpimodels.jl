"""
SIR (Susceptible-Infected-Removed) model implementation.

High-performance implementation using the same active node tracking strategy as ZIM.

The SIR process:
1. Each infected node transmits to a susceptible neighbor at rate β × #susceptible_neighbors
2. Each infected node recovers spontaneously at rate γ
3. Process stops when no infected nodes remain
"""

using Random

# =============================================================================
# SIR Process Implementation
# =============================================================================

"""
High-performance SIR (Susceptible-Infected-Removed) epidemic process.

Uses a two-component rate structure:
- Infection events: total rate β × (total boundary size)
- Recovery events: total rate γ × (number of infected nodes)

Active node tracking (`DictActiveTracker`) is used for infection events only
(weighted by susceptible neighbor count). A separate `Set{Int}` tracks all
infected nodes for O(1)-per-step recovery sampling.

# Fields
- `graph::AbstractEpidemicGraph`: The underlying graph
- `β::Float64`: Transmission rate per infectious contact
- `γ::Float64`: Recovery rate
- `active_tracker::DictActiveTracker`: Infected nodes with susceptible neighbors
- `infected_nodes::Set{Int}`: All currently infected nodes
- `time::Float64`: Current simulation time
- `steps::Int`: Number of steps executed
- `rng::AbstractRNG`: Random number generator
"""
mutable struct SIRProcess{G<:AbstractEpidemicGraph, R<:AbstractRNG} <: SIRLikeProcess
    # Concrete `graph`/`rng` type parameters: abstract fields would force the
    # neighbor queries and rand() calls in step! through dynamic dispatch (and
    # rand() would box its result ~16 bytes/call) — overhead/GC pressure (#1/#3).
    graph::G
    β::Float64
    γ::Float64
    active_tracker::DictActiveTracker
    infected_nodes::Set{Int}
    time::Float64
    steps::Int
    rng::R

    function SIRProcess(graph::G, β::Float64, γ::Float64;
                        rng::R = Random.default_rng()) where {G<:AbstractEpidemicGraph, R<:AbstractRNG}
        _validate_rates("Transmission rate β" => β, "Recovery rate γ" => γ)
        new{G,R}(graph, β, γ, DictActiveTracker(), Set{Int}(), 0.0, 0, rng)
    end
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

function is_active(process::SIRProcess)::Bool
    return !isempty(process.infected_nodes)
end

function get_total_rate(process::SIRProcess)::Float64
    infection_rate = process.β * get_total_boundary(process.active_tracker)
    recovery_rate  = process.γ * length(process.infected_nodes)
    return infection_rate + recovery_rate
end

# Two event classes: infection (rate β·boundary, weighted by susceptible-neighbor
# count) vs. spontaneous recovery (rate γ·#infected, uniform over infected nodes).
# (The shared step! draws the waiting time and advances the clock.)
@inline function _fire_event!(process::SIRProcess, total_rate::Float64)
    infection_rate = process.β * get_total_boundary(process.active_tracker)
    if rand(process.rng) < infection_rate / total_rate
        _sir_infect!(process, _weighted_sample_active(process.active_tracker, process.rng))
    else
        _sir_recover!(process, _sample_infected_uniform(process))
    end
end

function _clear_trackers!(process::SIRProcess)
    clear_active_nodes!(process.active_tracker)
    empty!(process.infected_nodes)
end

function reset!(process::SIRProcess,
                initial_infected::Vector{Int};
                rng_seed::Union{Int, Nothing} = nothing)
    validate_initial_infected(initial_infected, process.graph)
    states = _reset_prologue!(process; rng_seed = rng_seed)

    # Mark every seed infected first, then compute susceptible-neighbor counts.
    # Counting in the same pass would let an earlier seed see a later seed as still
    # susceptible, leaving its boundary count stale-high.
    infected_state = state_to_int(INFECTED)
    for node_id in initial_infected
        states[node_id] = infected_state
        push!(process.infected_nodes, node_id)
    end
    for node_id in initial_infected
        susceptible_count = count_neighbors_by_state(process.graph, node_id, SUSCEPTIBLE)
        add_active_node!(process.active_tracker, node_id, susceptible_count)
    end
end

# =============================================================================
# SIR Event Handlers (Internal)
# =============================================================================

function _sir_infect!(process::SIRProcess, acting_node::Int)
    neighbors = get_neighbors(process.graph, acting_node)
    states    = node_states_raw(process.graph)
    target = _random_susceptible_neighbor(neighbors, states, Int[], process.rng)

    if target == 0
        @warn "Infected node $acting_node has no susceptible neighbors but is marked active"
        remove_active_node!(process.active_tracker, acting_node)
        return
    end

    states[target] = state_to_int(INFECTED)
    push!(process.infected_nodes, target)
    # Same S→I tracker maintenance as ZIM (shared helper).
    _track_si_infection!(process.active_tracker, process.graph,
                         get_neighbors(process.graph, target), acting_node, target)
end

function _sir_recover!(process::SIRProcess, recovering_node::Int)
    states = node_states_raw(process.graph)
    states[recovering_node] = state_to_int(REMOVED)

    delete!(process.infected_nodes, recovering_node)
    remove_active_node!(process.active_tracker, recovering_node)
    # Recovery I→R does not change any neighbor's susceptible-neighbor count,
    # so no active_tracker updates are needed for other nodes.
end

# =============================================================================
# Internal Sampling Helper
# =============================================================================

function _sample_infected_uniform(process::SIRProcess)::Int
    n = length(process.infected_nodes)
    target_idx = rand(process.rng, 1:n)
    i = 0
    for node in process.infected_nodes
        i += 1
        if i == target_idx
            return node
        end
    end
    return first(process.infected_nodes)
end

# =============================================================================
# SIR-Specific Statistics
# =============================================================================

function get_sir_statistics(process::SIRProcess)::Dict{Symbol, Any}
    return _augment_statistics(process;
        β = process.β,
        γ = process.γ,
        basic_reproduction_number = process.β / process.γ,
        active_infected = length(process.active_tracker.active_nodes),
        total_boundary_size = get_total_boundary(process.active_tracker))
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
Create a configured SIR process ready to run.

# Arguments
- `graph::AbstractEpidemicGraph`: The graph to simulate on
- `β::Float64`: Transmission rate per contact
- `γ::Float64`: Recovery rate (default: 1.0)
- `initial_infected::Union{Symbol, Vector{Int}}`: :center, :random, or node indices
- `rng_seed::Union{Int, Nothing}`: Random seed

# Returns
- `SIRProcess`: Configured SIR process ready to run

# Example
```julia
julia> lattice = create_square_lattice(100, 100, :absorbing)
julia> sir = create_sir_process(lattice, 0.5)
julia> results = run_simulation(sir; max_time=100.0)
```
"""
function create_sir_process(graph::AbstractEpidemicGraph, β::Float64, γ::Float64 = 1.0;
                            initial_infected::Union{Symbol, Vector{Int}} = :center,
                            rng_seed::Union{Int, Nothing} = nothing)
    rng = create_rng(rng_seed)
    process = SIRProcess(graph, β, γ; rng=rng)
    reset!(process, resolve_initial_nodes(graph, initial_infected, rng))
    return process
end

"""
Convenience overload for creating an SIR process on a square lattice. Keyword
arguments (`boundary`, `initial_infected`, `rng_seed`) flow through to the
graph-based method.
"""
create_sir_process(width::Int, height::Int, β::Float64, γ::Float64 = 1.0; kwargs...) =
    create_on_square_lattice(create_sir_process, width, height, β, γ; kwargs...)
