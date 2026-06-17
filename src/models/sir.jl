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
(weighted by susceptible neighbor count). Recovery is uniform over *all* infected
nodes, so a dense `Vector{Int}` of infected node ids (with a `node → index` map)
tracks them for O(1) recovery sampling and O(1) swap-removal.

# Fields
- `graph::AbstractEpidemicGraph`: The underlying graph
- `β::Float64`: Transmission rate per infectious contact
- `γ::Float64`: Recovery rate
- `active_tracker::DictActiveTracker`: Infected nodes with susceptible neighbors
- `infected_nodes::Vector{Int}`: All currently infected nodes (dense, unordered)
- `infected_index::Dict{Int,Int}`: node_id → its position in `infected_nodes`
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
    # Recovery picks a uniformly random infected node. A dense Vector gives O(1)
    # index sampling, and the parallel node→index map turns removal into an O(1)
    # swap-with-last (vs. the previous Set, whose only random access was an
    # O(#infected) iteration walk to the k-th element — linear per recovery step).
    infected_nodes::Vector{Int}
    infected_index::Dict{Int, Int}
    time::Float64
    steps::Int
    rng::R
    # Reusable scratch buffers so per-step event handling allocates nothing on the
    # hot path (mirrors ZIM; see issues #1/#3). neighbor_buf backs get_neighbors!
    # calls; susceptible_buf collects susceptible neighbors when choosing a target.
    # A process is used by one thread at a time, so reuse needs no locking.
    neighbor_buf::Vector{Int}
    susceptible_buf::Vector{Int}

    function SIRProcess(graph::G, β::Float64, γ::Float64;
                        rng::R = Random.default_rng()) where {G<:AbstractEpidemicGraph, R<:AbstractRNG}
        _validate_rates("Transmission rate β" => β, "Recovery rate γ" => γ)
        neighbor_buf = Int[]; susceptible_buf = Int[]
        sizehint!(neighbor_buf, 8); sizehint!(susceptible_buf, 8)
        new{G,R}(graph, β, γ, DictActiveTracker(), Int[], Dict{Int, Int}(), 0.0, 0, rng,
            neighbor_buf, susceptible_buf)
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
    empty!(process.infected_index)
end

# =============================================================================
# Infected-set maintenance (dense Vector + node→index map)
# =============================================================================

"""Append a newly infected node, recording its position for O(1) swap-removal."""
@inline function _add_infected!(process::SIRProcess, node_id::Int)
    push!(process.infected_nodes, node_id)
    process.infected_index[node_id] = length(process.infected_nodes)
end

"""
Remove a recovered node in O(1) by swapping the last infected node into its slot
(keeping `infected_nodes` and `infected_index` consistent).
"""
@inline function _remove_infected!(process::SIRProcess, node_id::Int)
    nodes = process.infected_nodes
    idx = process.infected_index[node_id]
    last_node = nodes[end]
    @inbounds nodes[idx] = last_node
    process.infected_index[last_node] = idx
    pop!(nodes)
    delete!(process.infected_index, node_id)
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
        _add_infected!(process, node_id)
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
    # Reused scratch buffers (no allocation). neighbor_buf may alias an internal
    # list for non-lattice graphs, so it is read only until we re-fetch below.
    neighbors = get_neighbors!(process.neighbor_buf, process.graph, acting_node)
    states    = node_states_raw(process.graph)
    target = _random_susceptible_neighbor(neighbors, states, process.susceptible_buf, process.rng)

    if target == 0
        @warn "Infected node $acting_node has no susceptible neighbors but is marked active"
        remove_active_node!(process.active_tracker, acting_node)
        return
    end

    states[target] = state_to_int(INFECTED)
    _add_infected!(process, target)
    # Same S→I tracker maintenance as ZIM (shared helper). Re-fetch into
    # neighbor_buf (acting_node's neighbors are no longer needed).
    _track_si_infection!(process.active_tracker, process.graph,
                         get_neighbors!(process.neighbor_buf, process.graph, target),
                         acting_node, target)
end

function _sir_recover!(process::SIRProcess, recovering_node::Int)
    states = node_states_raw(process.graph)
    states[recovering_node] = state_to_int(REMOVED)

    _remove_infected!(process, recovering_node)
    remove_active_node!(process.active_tracker, recovering_node)
    # Recovery I→R does not change any neighbor's susceptible-neighbor count,
    # so no active_tracker updates are needed for other nodes.
end

# =============================================================================
# Internal Sampling Helper
# =============================================================================

function _sample_infected_uniform(process::SIRProcess)::Int
    n = length(process.infected_nodes)
    return @inbounds process.infected_nodes[rand(process.rng, 1:n)]
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
