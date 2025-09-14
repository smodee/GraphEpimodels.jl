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

"""
Abstract type for contact-like processes where nodes transition S ⇄ I.

Examples:
- Contact Process: Birth-death process on graphs
- SIS: SIR with immediate reinfection possibility
"""
abstract type ContactLikeProcess <: AbstractEpidemicProcess end

"""
Abstract type for voter-like processes with competing states/opinions.

Examples:
- Biased Voter: Asymmetric influence between states
- Voter Model: Symmetric opinion dynamics
"""
abstract type VoterLikeProcess <: AbstractEpidemicProcess end

# =============================================================================
# Required Interface Methods (must be implemented by all process types)
# =============================================================================

"""
Get the graph on which this process operates.

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `AbstractEpidemicGraph`: The underlying graph
"""
function get_graph(process::AbstractEpidemicProcess)::AbstractEpidemicGraph
    error("get_graph must be implemented by concrete process type $(typeof(process))")
end

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
function reset!(process::AbstractEpidemicProcess, initial_infected::Vector{Int})
    error("reset! must be implemented by concrete process type $(typeof(process))")
end

"""
Get current simulation time.

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Float64`: Current simulation time
"""
function current_time(process::AbstractEpidemicProcess)::Float64
    error("current_time must be implemented by concrete process type $(typeof(process))")
end

"""
Get number of simulation steps executed.

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Int`: Number of steps executed
"""
function step_count(process::AbstractEpidemicProcess)::Int
    error("step_count must be implemented by concrete process type $(typeof(process))")
end

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
julia> zim = create_zim_simulation(100, 100, 2.0)
julia> results = run_simulation(zim; max_time=50.0)
julia> println("Final infected: ", results[:infected])
```
"""
function run_simulation(process::AbstractEpidemicProcess;
                       max_time::Float64 = Inf,
                       max_steps::Int = 1_000_000,
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
    else
        final_stats[:termination_reason] = :unknown
    end
    
    if save_history
        final_stats[:history] = history
    end
    
    return final_stats
end

# =============================================================================
# Active Node Tracking (Critical for Performance)
# =============================================================================

"""
Abstract interface for tracking active nodes efficiently.

This is critical for performance - epidemic processes should only iterate 
over nodes that can actually cause events, not all infected nodes.
"""
abstract type ActiveNodeTracker end

"""
Simple active node tracker using a dictionary.

Maps node_id → number of susceptible neighbors.
This is the pattern from your efficient old implementation.
"""
mutable struct DictActiveTracker <: ActiveNodeTracker
    active_nodes::Dict{Int, Int}  # node_id → susceptible_neighbor_count
    
    DictActiveTracker() = new(Dict{Int, Int}())
end

"""
Add a node to active tracking.

# Arguments
- `tracker::DictActiveTracker`: The tracker
- `node_id::Int`: Node to add
- `neighbor_count::Int`: Number of susceptible neighbors
"""
function add_active_node!(tracker::DictActiveTracker, node_id::Int, neighbor_count::Int)
    if neighbor_count > 0
        tracker.active_nodes[node_id] = neighbor_count
    end
end

"""
Remove a node from active tracking.
"""
function remove_active_node!(tracker::DictActiveTracker, node_id::Int)
    delete!(tracker.active_nodes, node_id)
end

"""
Update neighbor count for an active node.
"""
function update_active_node!(tracker::DictActiveTracker, node_id::Int, new_count::Int)
    if new_count > 0
        tracker.active_nodes[node_id] = new_count
    else
        delete!(tracker.active_nodes, node_id)
    end
end

"""
Get all active nodes.
"""
function get_active_nodes(tracker::DictActiveTracker)::Vector{Int}
    return collect(keys(tracker.active_nodes))
end

"""
Get total boundary (sum of all neighbor counts).
"""
function get_total_boundary(tracker::DictActiveTracker)::Int
    return sum(values(tracker.active_nodes))
end

"""
Check if any nodes are active.
"""
function has_active_nodes(tracker::DictActiveTracker)::Bool
    return !isempty(tracker.active_nodes)
end

"""
Sample an active node randomly.
"""
function sample_active_node(tracker::DictActiveTracker, rng::AbstractRNG)::Int
    error("sample_active_node must be implemented by concrete process type")
end

"""
Clear all active nodes.
"""
function clear_active_nodes!(tracker::DictActiveTracker)
    empty!(tracker.active_nodes)
end

# =============================================================================
# Internal Sampling Helpers (Not Exported)
# =============================================================================
# Add these functions to epiprocess.jl, after the DictActiveTracker methods

"""
Uniform sampling from active nodes.

Selects an active node uniformly at random, regardless of neighbor counts.
Used by processes where all active nodes have equal event rates.

# Arguments
- `tracker::DictActiveTracker`: Active node tracker
- `rng::AbstractRNG`: Random number generator

# Returns
- `Int`: Randomly selected active node ID

# Throws
- `ArgumentError`: If no active nodes available
"""
function _uniform_sample_active(tracker::DictActiveTracker, rng::AbstractRNG)::Int
    if isempty(tracker.active_nodes)
        throw(ArgumentError("No active nodes to sample from"))
    end
    
    # Simple uniform sampling from dictionary keys
    active_nodes = collect(keys(tracker.active_nodes))
    return active_nodes[rand(rng, 1:length(active_nodes))]
end

"""
Weighted sampling from active nodes based on susceptible neighbor counts.

Implements efficient weighted sampling where the probability of selecting a node
is proportional to its number of susceptible neighbors. This is critical for
processes like ZIM where event rates depend on neighbor counts.

Uses the standard weighted sampling algorithm:
1. Calculate cumulative weights
2. Sample uniform random value in [0, total_weight]
3. Binary search to find selected node

# Arguments
- `tracker::DictActiveTracker`: Active node tracker with neighbor counts
- `rng::AbstractRNG`: Random number generator

# Returns
- `Int`: Selected node ID (weighted by neighbor count)

# Throws
- `ArgumentError`: If no active nodes available
"""
function _weighted_sample_active(tracker::DictActiveTracker, rng::AbstractRNG)::Int
    if isempty(tracker.active_nodes)
        throw(ArgumentError("No active nodes to sample from"))
    end
    
    # Special case: single active node
    if length(tracker.active_nodes) == 1
        return first(keys(tracker.active_nodes))
    end
    
    # Get total weight (sum of all susceptible neighbor counts)
    total_weight = get_total_boundary(tracker)
    
    if total_weight <= 0
        throw(ArgumentError("Total weight is zero - no valid nodes to sample"))
    end
    
    # Sample a random value in [0, total_weight)
    random_value = rand(rng) * total_weight
    
    # Find which node this corresponds to using cumulative weights
    cumulative_weight = 0.0
    for (node_id, weight) in tracker.active_nodes
        cumulative_weight += weight
        if random_value < cumulative_weight
            return node_id
        end
    end
    
    # Shouldn't reach here, but return last node as fallback
    # (can happen due to floating point rounding)
    return last(keys(tracker.active_nodes))
end

"""
Alternative weighted sampling implementation using pre-allocated arrays.

More efficient for very large numbers of active nodes, but requires allocation.
Use this version if you have thousands of active nodes.

# Arguments
- `tracker::DictActiveTracker`: Active node tracker with neighbor counts
- `n_active::Int`: Number of active nodes to use for pre-allocation 
- `rng::AbstractRNG`: Random number generator

# Returns
- `Int`: Selected node ID (weighted by neighbor count)
"""
function _weighted_sample_active_fast(tracker::DictActiveTracker, n_active::Int, rng::AbstractRNG)::Int
    if n_active == 0
        throw(ArgumentError("No active nodes to sample from"))
    end
    
    if n_active == 1
        return first(keys(tracker.active_nodes))
    end
    
    # Pre-allocate arrays for better performance with many nodes
    nodes = Vector{Int}(undef, n_active)
    weights = Vector{Int}(undef, n_active)
    
    # Fill arrays
    i = 1
    for (node_id, weight) in tracker.active_nodes
        nodes[i] = node_id
        weights[i] = weight
        i += 1
    end
    
    # Calculate cumulative weights
    cumsum_weights = cumsum(weights)
    total_weight = cumsum_weights[end]
    
    if total_weight <= 0
        throw(ArgumentError("Total weight is zero - no valid nodes to sample"))
    end
    
    # Binary search for efficiency
    random_value = rand(rng) * total_weight
    idx = searchsortedfirst(cumsum_weights, random_value)
    
    # Handle edge case where random_value == total_weight
    if idx > n_active
        idx = n_active
    end
    
    return nodes[idx]
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