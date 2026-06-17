"""
Zombie Infection Model (ZIM) implementation.

High-performance implementation of the ZIM process using efficient active node tracking.
Based on the algorithm from your original implementation, optimized for large lattices.

The ZIM process:
1. Each infected node (zombie) attacks susceptible neighbors at rate (λ + μ) × #susceptible_neighbors  
2. Outcome probability: λ/(λ+μ) for infection, μ/(λ+μ) for kill
3. Process stops when no active zombies remain or escape occurs
"""

using Random

# Import required interfaces (assumes graphs.jl and epiprocess.jl are loaded)

# =============================================================================
# ZIM Process Implementation  
# =============================================================================

"""
High-performance Zombie Infection Model process.

Uses efficient active node tracking (only processes zombies that can cause events)
and weighted sampling based on susceptible neighbor counts. Optimized for 
large-scale simulations on lattices and general graphs.

# Fields
- `graph::AbstractEpidemicGraph`: The underlying graph
- `λ::Float64`: Infection rate (bite rate)
- `μ::Float64`: Kill rate (fight-back rate)  
- `infection_prob::Float64`: Pre-computed λ/(λ+μ)
- `active_tracker::DictActiveTracker`: Tracks active zombies and neighbor counts
- `time::Float64`: Current simulation time
- `steps::Int`: Number of steps executed
- `rng::AbstractRNG`: Random number generator
"""
mutable struct ZIMProcess{G<:AbstractEpidemicGraph, R<:AbstractRNG} <: SIRLikeProcess
    # `graph` and `rng` are concrete type parameters rather than the abstract
    # supertypes. Abstract fields would force every get_neighbors!/
    # count_neighbors_by_state/rand call in step! through dynamic dispatch
    # (preventing inlining of the small lattice routines), and rand() would
    # additionally box its result (~16 bytes/call) — per-step overhead and GC
    # pressure (cf. issues #1/#3).
    graph::G
    λ::Float64
    μ::Float64
    infection_prob::Float64
    active_tracker::DictActiveTracker
    time::Float64
    steps::Int
    rng::R
    # Reusable scratch buffers for the per-step event handlers, so that stepping
    # allocates nothing on the hot path (see issues #1 / #3). neighbor_buf backs
    # get_neighbors! calls; susceptible_buf collects susceptible neighbors when
    # choosing an infection target. Each process is used by one thread at a time,
    # so reuse is safe without locking.
    neighbor_buf::Vector{Int}
    susceptible_buf::Vector{Int}

    function ZIMProcess(graph::G, λ::Float64, μ::Float64 = 1.0;
                       rng::R = Random.default_rng()) where {G<:AbstractEpidemicGraph, R<:AbstractRNG}

        _validate_rates("Infection rate λ" => λ, "Kill rate μ" => μ)

        infection_prob = λ / (λ + μ)
        active_tracker = DictActiveTracker()

        neighbor_buf = Int[]
        susceptible_buf = Int[]
        sizehint!(neighbor_buf, 8)
        sizehint!(susceptible_buf, 8)

        new{G,R}(graph, λ, μ, infection_prob, active_tracker, 0.0, 0, rng,
            neighbor_buf, susceptible_buf)
    end
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

function is_active(process::ZIMProcess)::Bool
    return has_active_nodes(process.active_tracker)
end

function get_total_rate(process::ZIMProcess)::Float64
    boundary_size = get_total_boundary(process.active_tracker)
    return (process.λ + process.μ) * boundary_size
end

# ZIM fires a single event class per active zombie: sample one (weighted by its
# susceptible-neighbor count), then split λ/(λ+μ) into infect vs. kill. (The shared
# step! draws the waiting time and advances the clock.)
@inline function _fire_event!(process::ZIMProcess, total_rate::Float64)
    acting_zombie = _weighted_sample_active(process.active_tracker, process.rng)
    if rand(process.rng) < process.infection_prob
        _zombie_wins!(process, acting_zombie)   # infect a susceptible neighbor
    else
        _zombie_loses!(process, acting_zombie)  # zombie is killed
    end
end

_clear_trackers!(process::ZIMProcess) = clear_active_nodes!(process.active_tracker)

function reset!(process::ZIMProcess,
                initial_infected::Vector{Int};
                rng_seed::Union{Int, Nothing} = nothing)
    validate_initial_infected(initial_infected, process.graph)
    states = _reset_prologue!(process; rng_seed = rng_seed)

    # Mark every seed infected first, then build active tracking in a second pass.
    # Counting susceptible neighbors in the same loop would let an earlier seed see
    # a later seed as still susceptible, leaving its count stale-high.
    infected_state = state_to_int(INFECTED)
    for node_id in initial_infected
        states[node_id] = infected_state
    end
    for node_id in initial_infected
        susceptible_count = count_neighbors_by_state(process.graph, node_id, SUSCEPTIBLE)
        add_active_node!(process.active_tracker, node_id, susceptible_count)
    end
end

# =============================================================================
# ZIM Event Handlers (Internal)
# =============================================================================

"""
Handle zombie victory: infect a random susceptible neighbor.

This is the performance-critical function that implements your original algorithm
with efficient active node tracking updates.
"""
function _zombie_wins!(process::ZIMProcess, zombie_node::Int)
    # Susceptible neighbors of the attacking zombie, via the reused scratch buffers
    # (no allocation). neighbor_buf may alias an internal list for non-lattice
    # graphs, so it is read only until we are finished with it.
    neighbors = get_neighbors!(process.neighbor_buf, process.graph, zombie_node)
    states = node_states_raw(process.graph)
    target = _random_susceptible_neighbor(neighbors, states,
                                          process.susceptible_buf, process.rng)

    if target == 0
        # This shouldn't happen if active tracking is correct.
        @warn "Zombie $zombie_node has no susceptible neighbors but is marked active"
        remove_active_node!(process.active_tracker, zombie_node)
        return
    end

    states[target] = state_to_int(INFECTED)
    # Re-fetch into neighbor_buf (the zombie's own neighbors are no longer needed).
    _track_si_infection!(process.active_tracker, process.graph,
                         get_neighbors!(process.neighbor_buf, process.graph, target),
                         zombie_node, target)
end

"""
Handle zombie defeat: zombie is killed and removed.
"""
function _zombie_loses!(process::ZIMProcess, zombie_node::Int)
    # Remove zombie from graph
    states = node_states_raw(process.graph)
    states[zombie_node] = state_to_int(REMOVED)
    
    # Remove from active tracking
    remove_active_node!(process.active_tracker, zombie_node)
    
    # Update neighbor active counts (they gained a susceptible "neighbor" - the removed zombie)
    _update_neighbors_after_zombie_death!(process, zombie_node)
end

"""
Update active tracking after a zombie death.
"""
function _update_neighbors_after_zombie_death!(process::ZIMProcess, dead_zombie::Int)
    # Find all zombie neighbors of the dead zombie - they gain a susceptible "slot"
    zombie_neighbors = get_neighbors!(process.neighbor_buf, process.graph, dead_zombie)
    states = node_states_raw(process.graph)
    infected_state = state_to_int(INFECTED)
    
    for neighbor in zombie_neighbors
        if states[neighbor] == infected_state
            # This zombie neighbor gained a susceptible neighbor (where the dead zombie was)
            current_count = get(process.active_tracker.active_nodes, neighbor, 0)
            new_count = count_neighbors_by_state(process.graph, neighbor, SUSCEPTIBLE)
            update_active_node!(process.active_tracker, neighbor, new_count)
        end
    end
end

# =============================================================================
# ZIM-Specific Utility Functions
# =============================================================================

"""
Get ZIM-specific statistics.
"""
function get_zim_statistics(process::ZIMProcess)::Dict{Symbol, Any}
    return _augment_statistics(process;
        λ = process.λ,
        μ = process.μ,
        infection_probability = process.infection_prob,
        active_zombies = length(process.active_tracker.active_nodes),
        total_boundary_size = get_total_boundary(process.active_tracker))
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
Create a complete ZIM simulation setup.

# Arguments  
- `graph::AbstractEpidemicGraph`: The graph to simulate on
- `λ::Float64`: Infection rate
- `μ::Float64`: Kill rate (default: 1.0)
- `initial_infected::Union{Symbol, Vector{Int}}`: :center, :random, or node indices
- `rng_seed::Union{Int, Nothing}`: Random seed (default: nothing)

# Returns
- `ZIMProcess`: Configured ZIM process ready to run

# Example
```julia  
julia> lattice = create_square_lattice(100, 100, :absorbing)
julia> zim = create_zim_process(lattice, 2.0)
julia> results = run_simulation(zim; max_time=100.0, stop_on_escape=true)
```
"""
function create_zim_process(graph::AbstractEpidemicGraph, λ::Float64, μ::Float64 = 1.0;
                           initial_infected::Union{Symbol, Vector{Int}} = :center,
                           rng_seed::Union{Int, Nothing} = nothing)
    rng = create_rng(rng_seed)
    process = ZIMProcess(graph, λ, μ; rng=rng)
    reset!(process, resolve_initial_nodes(graph, initial_infected, rng))
    return process
end

"""
Convenience overload for creating a ZIM process on a square lattice. Keyword
arguments (`boundary`, `initial_infected`, `rng_seed`) flow through to the
graph-based method.
"""
create_zim_process(width::Int, height::Int, λ::Float64, μ::Float64 = 1.0; kwargs...) =
    create_on_square_lattice(create_zim_process, width, height, λ, μ; kwargs...)