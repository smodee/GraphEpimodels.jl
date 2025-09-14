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
mutable struct ZIMProcess <: SIRLikeProcess
    graph::AbstractEpidemicGraph
    λ::Float64
    μ::Float64
    infection_prob::Float64
    active_tracker::DictActiveTracker
    time::Float64
    steps::Int
    rng::AbstractRNG
    
    function ZIMProcess(graph::AbstractEpidemicGraph, λ::Float64, μ::Float64 = 1.0;
                       rng::AbstractRNG = Random.default_rng())
        
        _validate_zim_parameters(λ, μ)
        
        infection_prob = λ / (λ + μ)
        active_tracker = DictActiveTracker()
        
        new(graph, λ, μ, infection_prob, active_tracker, 0.0, 0, rng)
    end
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

@inline function get_graph(process::ZIMProcess)::AbstractEpidemicGraph
    return process.graph
end

@inline function current_time(process::ZIMProcess)::Float64
    return process.time
end

@inline function step_count(process::ZIMProcess)::Int
    return process.steps
end

function is_active(process::ZIMProcess)::Bool
    return has_active_nodes(process.active_tracker)
end

function get_total_rate(process::ZIMProcess)::Float64
    boundary_size = get_total_boundary(process.active_tracker)
    return (process.λ + process.μ) * boundary_size
end

function sample_active_node(process::ZIMProcess, rng::AbstractRNG)::Int
    n_active = length(process.active_tracker.active_nodes)

    # Use performance-optimized function with pre-allocation if active set is large
    if n_active < 1024
        return _weighted_sample_active(process.active_tracker, rng)
    else
        return _weighted_sample_active_fast(process.active_tracker, n_active, rng)
    end
end

function step!(process::ZIMProcess)::Float64
    if !is_active(process)
        return Inf  # No active zombies
    end
    
    # Calculate time increment (Gillespie algorithm)
    total_rate = get_total_rate(process)
    if total_rate <= 0
        return Inf
    end
    
    dt = randexp(process.rng) / total_rate
    
    # Sample which zombie acts (weighted by susceptible neighbor count)
    acting_zombie = sample_active_node(process, process.rng)
    
    # Determine outcome: infection vs kill
    if rand(process.rng) < process.infection_prob
        # Zombie wins - infect a susceptible neighbor
        _zombie_wins!(process, acting_zombie)
    else
        # Zombie loses - gets killed
        _zombie_loses!(process, acting_zombie)
    end
    
    # Update time and step count
    process.time += dt
    process.steps += 1
    
    return dt
end

function reset!(process::ZIMProcess, initial_infected::Vector{Int})
    # Validate input
    validate_initial_infected(initial_infected, process.graph)
    
    # Reset time and counters
    process.time = 0.0
    process.steps = 0
    
    # Clear active tracking
    clear_active_nodes!(process.active_tracker)
    
    # Reset all nodes to susceptible
    states = node_states_raw(process.graph)
    fill!(states, state_to_int(SUSCEPTIBLE))
    
    # Set initial infected nodes and build active tracking
    infected_state = state_to_int(INFECTED)
    for node_id in initial_infected
        states[node_id] = infected_state
        
        # Add to active tracking if it has susceptible neighbors
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
    # Get susceptible neighbors of the attacking zombie
    neighbors = get_neighbors(process.graph, zombie_node)
    states = node_states_raw(process.graph)
    susceptible_state = state_to_int(SUSCEPTIBLE)
    infected_state = state_to_int(INFECTED)
    
    # Find susceptible neighbors
    susceptible_neighbors = Int[]
    for neighbor in neighbors
        if states[neighbor] == susceptible_state
            push!(susceptible_neighbors, neighbor)
        end
    end
    
    if isempty(susceptible_neighbors)
        # This shouldn't happen if active tracking is correct
        @warn "Zombie $zombie_node has no susceptible neighbors but is marked active"
        remove_active_node!(process.active_tracker, zombie_node)
        return
    end
    
    # Randomly select a susceptible neighbor to infect
    target = rand(process.rng, susceptible_neighbors)
    states[target] = infected_state
    
    # Update active tracking (this is the key performance optimization)
    _update_active_tracking_after_infection!(process, zombie_node, target)
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
Update active tracking after a successful infection.

This implements the efficient neighbor count updates from your original algorithm.
"""
function _update_active_tracking_after_infection!(process::ZIMProcess, attacking_zombie::Int, new_zombie::Int)
    # 1. Check if the new zombie becomes active
    new_susceptible_count = count_neighbors_by_state(process.graph, new_zombie, SUSCEPTIBLE)
    add_active_node!(process.active_tracker, new_zombie, new_susceptible_count)
    
    # 2. Update the attacking zombie's count (lost one susceptible neighbor)
    current_count = get(process.active_tracker.active_nodes, attacking_zombie, 0)
    update_active_node!(process.active_tracker, attacking_zombie, current_count - 1)
    
    # 3. Update all zombie neighbors of the new zombie (they each lost a susceptible neighbor)
    new_zombie_neighbors = get_neighbors(process.graph, new_zombie)
    states = node_states_raw(process.graph)
    infected_state = state_to_int(INFECTED)
    
    for neighbor in new_zombie_neighbors
        if states[neighbor] == infected_state && neighbor != attacking_zombie
            # This zombie neighbor lost a susceptible neighbor
            neighbor_count = get(process.active_tracker.active_nodes, neighbor, 0)
            if neighbor_count > 0
                update_active_node!(process.active_tracker, neighbor, neighbor_count - 1)
            end
        end
    end
end

"""
Update active tracking after a zombie death.
"""
function _update_neighbors_after_zombie_death!(process::ZIMProcess, dead_zombie::Int)
    # Find all zombie neighbors of the dead zombie - they gain a susceptible "slot"
    zombie_neighbors = get_neighbors(process.graph, dead_zombie)
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
    base_stats = get_statistics(process)
    
    # Add ZIM-specific information
    base_stats[:λ] = process.λ
    base_stats[:μ] = process.μ
    base_stats[:infection_probability] = process.infection_prob
    base_stats[:escaped] = has_escaped(process)
    base_stats[:active_zombies] = length(process.active_tracker.active_nodes)
    base_stats[:total_boundary_size] = get_total_boundary(process.active_tracker)
    
    return base_stats
end

# =============================================================================
# Parameter Validation (ZIM-specific)
# =============================================================================

"""
Validate ZIM-specific parameters.
"""
function _validate_zim_parameters(λ::Float64, μ::Float64)
    if λ <= 0.0
        throw(ArgumentError("Infection rate λ must be positive, got $λ"))
    end
    if μ <= 0.0
        throw(ArgumentError("Kill rate μ must be positive, got $μ"))
    end
    if λ > 1000.0 || μ > 1000.0
        @warn "Very large rates (λ=$λ, μ=$μ) may cause numerical issues"
    end
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
julia> zim = create_zim_simulation(lattice, 2.0)
julia> results = run_simulation(zim; max_time=100.0, stop_on_escape=true)
```
"""
function create_zim_simulation(graph::AbstractEpidemicGraph, λ::Float64, μ::Float64 = 1.0;
                              initial_infected::Union{Symbol, Vector{Int}} = :center,
                              rng_seed::Union{Int, Nothing} = nothing)::ZIMProcess
    
    # Create RNG using utils.jl function
    rng = create_rng(rng_seed)
    
    # Create process
    process = ZIMProcess(graph, λ, μ; rng=rng)
    
    # Determine initial infected nodes
    infected_nodes = if initial_infected == :center
        if isdefined(graph, :get_center_node) || hasmethod(get_center_node, (typeof(graph),))
            [get_center_node(graph)]
        else
            [num_nodes(graph) ÷ 2]  # Fallback for general graphs
        end
    elseif initial_infected == :random
        [rand(rng, 1:num_nodes(graph))]
    else
        initial_infected
    end
    
    # Initialize the process
    reset!(process, infected_nodes)
    
    return process
end

"""
Convenience function for creating ZIM on square lattices (backward compatibility).
"""
function create_zim_simulation(width::Int, height::Int, λ::Float64, μ::Float64 = 1.0;
                              boundary::Symbol = :absorbing,
                              initial_infected::Union{Symbol, Vector{Int}} = :center,
                              rng_seed::Union{Int, Nothing} = nothing)::ZIMProcess
    
    lattice = create_square_lattice(width, height, boundary)
    return create_zim_simulation(lattice, λ, μ; initial_infected=initial_infected, rng_seed=rng_seed)
end