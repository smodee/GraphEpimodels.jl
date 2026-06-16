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
        _validate_sir_parameters(β, γ)
        new{G,R}(graph, β, γ, DictActiveTracker(), Set{Int}(), 0.0, 0, rng)
    end
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

@inline function get_graph(process::SIRProcess)::AbstractEpidemicGraph
    return process.graph
end

@inline function current_time(process::SIRProcess)::Float64
    return process.time
end

@inline function step_count(process::SIRProcess)::Int
    return process.steps
end

function is_active(process::SIRProcess)::Bool
    return !isempty(process.infected_nodes)
end

function get_total_rate(process::SIRProcess)::Float64
    infection_rate = process.β * get_total_boundary(process.active_tracker)
    recovery_rate  = process.γ * length(process.infected_nodes)
    return infection_rate + recovery_rate
end

function step!(process::SIRProcess)::Float64
    if !is_active(process)
        return Inf
    end

    total_rate = get_total_rate(process)
    if total_rate <= 0
        return Inf
    end

    dt = randexp(process.rng) / total_rate

    infection_rate = process.β * get_total_boundary(process.active_tracker)
    if rand(process.rng) < infection_rate / total_rate
        acting_node = _weighted_sample_active(process.active_tracker, process.rng)
        _sir_infect!(process, acting_node)
    else
        recovering_node = _sample_infected_uniform(process)
        _sir_recover!(process, recovering_node)
    end

    process.time  += dt
    process.steps += 1

    return dt
end

function reset!(process::SIRProcess,
                initial_infected::Vector{Int};
                rng_seed::Union{Int, Nothing} = nothing)
    validate_initial_infected(initial_infected, process.graph)

    process.time  = 0.0
    process.steps = 0

    if rng_seed !== nothing
        Random.seed!(process.rng, rng_seed)
    end

    clear_active_nodes!(process.active_tracker)
    empty!(process.infected_nodes)

    states = node_states_raw(process.graph)
    fill!(states, state_to_int(SUSCEPTIBLE))

    # Mark every seed infected first, then compute susceptible-neighbor counts.
    # Counting in the same pass would let an earlier seed see a later seed as
    # still susceptible, leaving its boundary count stale-high.
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
    susceptible_state = state_to_int(SUSCEPTIBLE)
    infected_state    = state_to_int(INFECTED)

    susceptible_neighbors = Int[]
    for neighbor in neighbors
        if states[neighbor] == susceptible_state
            push!(susceptible_neighbors, neighbor)
        end
    end

    if isempty(susceptible_neighbors)
        @warn "Infected node $acting_node has no susceptible neighbors but is marked active"
        remove_active_node!(process.active_tracker, acting_node)
        return
    end

    target = rand(process.rng, susceptible_neighbors)
    states[target] = infected_state
    push!(process.infected_nodes, target)

    _update_active_tracking_after_sir_infection!(process, acting_node, target)
end

function _sir_recover!(process::SIRProcess, recovering_node::Int)
    states = node_states_raw(process.graph)
    states[recovering_node] = state_to_int(REMOVED)

    delete!(process.infected_nodes, recovering_node)
    remove_active_node!(process.active_tracker, recovering_node)
    # Recovery I→R does not change any neighbor's susceptible-neighbor count,
    # so no active_tracker updates are needed for other nodes.
end

"""
Update active tracking after a new infection — same logic as ZIM.

When node `new_infected` transitions S→I:
- `new_infected` joins the active tracker with its susceptible neighbor count
- `attacker` loses one susceptible neighbor (the newly infected node)
- All other infected neighbors of `new_infected` also lose one susceptible neighbor
"""
function _update_active_tracking_after_sir_infection!(process::SIRProcess, attacker::Int, new_infected::Int)
    new_susceptible_count = count_neighbors_by_state(process.graph, new_infected, SUSCEPTIBLE)
    add_active_node!(process.active_tracker, new_infected, new_susceptible_count)

    current_count = get(process.active_tracker.active_nodes, attacker, 0)
    update_active_node!(process.active_tracker, attacker, current_count - 1)

    new_infected_neighbors = get_neighbors(process.graph, new_infected)
    states = node_states_raw(process.graph)
    infected_state = state_to_int(INFECTED)

    for neighbor in new_infected_neighbors
        if states[neighbor] == infected_state && neighbor != attacker
            neighbor_count = get(process.active_tracker.active_nodes, neighbor, 0)
            if neighbor_count > 0
                update_active_node!(process.active_tracker, neighbor, neighbor_count - 1)
            end
        end
    end
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
    base_stats = get_statistics(process)
    base_stats[:β]                  = process.β
    base_stats[:γ]                  = process.γ
    base_stats[:basic_reproduction_number] = process.β / process.γ
    base_stats[:escaped]            = has_escaped(process)
    base_stats[:active_infected]    = length(process.active_tracker.active_nodes)
    base_stats[:total_boundary_size] = get_total_boundary(process.active_tracker)
    return base_stats
end

# =============================================================================
# Parameter Validation
# =============================================================================

function _validate_sir_parameters(β::Float64, γ::Float64)
    if β <= 0.0
        throw(ArgumentError("Transmission rate β must be positive, got $β"))
    end
    if γ <= 0.0
        throw(ArgumentError("Recovery rate γ must be positive, got $γ"))
    end
    if β > 1000.0 || γ > 1000.0
        @warn "Very large rates (β=$β, γ=$γ) may cause numerical issues"
    end
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
Convenience overload for creating an SIR process on a square lattice.
"""
function create_sir_process(width::Int, height::Int, β::Float64, γ::Float64 = 1.0;
                            boundary::Symbol = :absorbing,
                            initial_infected::Union{Symbol, Vector{Int}} = :center,
                            rng_seed::Union{Int, Nothing} = nothing)
    lattice = create_square_lattice(width, height, boundary)
    return create_sir_process(lattice, β, γ; initial_infected=initial_infected, rng_seed=rng_seed)
end
