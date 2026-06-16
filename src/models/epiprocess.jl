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
function reset!(process::AbstractEpidemicProcess,
                initial_infected::Vector{Int};
                rng_seed::Union{Int, Nothing} = nothing)
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

    # Scratch buffers reused by _weighted_sample_active_fast so that weighted
    # sampling allocates nothing on the hot path. A tracker is owned by a single
    # process (and thus a single thread during threaded runs), so reusing these
    # across calls is safe without any locking. See issues #1 / #3.
    nodes_buf::Vector{Int}        # idx → node_id
    cumw_buf::Vector{Int}         # idx → cumulative weight up to and including idx

    DictActiveTracker() = new(Dict{Int, Int}(), Int[], Int[])
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
Clear all active nodes.
"""
function clear_active_nodes!(tracker::DictActiveTracker)
    empty!(tracker.active_nodes)
end

# =============================================================================
# Internal Sampling Helpers (Not Exported)
# =============================================================================

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
    n_active = length(tracker.active_nodes)
    if n_active == 0
        throw(ArgumentError("No active nodes to sample from"))
    end

    # Delegate to the single canonical (deterministic) sampler so that every
    # active-set size goes through the same order-independent selection. The
    # previous implementation walked the Dict directly, which made the chosen
    # node depend on Dict iteration order — see _weighted_sample_active_fast.
    return _weighted_sample_active_fast(tracker, n_active, rng)
end

"""
Canonical weighted sampler over the active set, using the tracker's buffers.

Builds a cumulative-weight array and binary-searches it (`O(n)` to fill plus
`O(log n)` to sample). Two properties matter:

1. **Allocation-free.** The node/cumulative-weight scratch arrays live on the
   tracker and are resized in place, so this allocates nothing on the hot path.
   It runs on every Gillespie step of every simulation; per-step heap allocation
   here previously caused severe multithreaded GC pressure that serialized worker
   threads (issues #1 and #3).

2. **Deterministic.** Nodes are sampled in a canonical order (ascending node id),
   NOT in `Dict` iteration order. A `Dict`'s iteration order depends on its
   insertion/deletion/rehash history, so two trackers with identical contents but
   different histories — a freshly constructed process vs. one reused across many
   simulations, or per-thread processes that each ran a different block of work —
   would otherwise map the same random draw onto different nodes. That made
   results depend on execution mode (serial vs. threaded) for a fixed seed.
   Sorting by node id removes the dependence; QuickSort is in-place so this stays
   allocation-free.

The tracker is owned by a single process, so buffer reuse needs no locking.

# Arguments
- `tracker::DictActiveTracker`: Active node tracker with neighbor counts
- `n_active::Int`: Number of active nodes (length of `tracker.active_nodes`)
- `rng::AbstractRNG`: Random number generator

# Returns
- `Int`: Selected node ID (weighted by neighbor count)
"""
function _weighted_sample_active_fast(tracker::DictActiveTracker, n_active::Int, rng::AbstractRNG)::Int
    if n_active == 0
        throw(ArgumentError("No active nodes to sample from"))
    end

    if n_active == 1
        # first(d::Dict) returns a Pair{K,V}; .first extracts the key without
        # allocating a KeySet wrapper (unlike first(keys(d))).
        return first(tracker.active_nodes).first
    end

    # Reuse the tracker's scratch buffers instead of allocating fresh vectors.
    nodes = tracker.nodes_buf
    cumw = tracker.cumw_buf
    resize!(nodes, n_active)
    resize!(cumw, n_active)

    # Collect node ids, then sort into a canonical order so sampling does not
    # depend on Dict iteration order (see the docstring). QuickSort is in-place.
    # Iterate the Dict directly (not via keys()) to avoid allocating a KeySet
    # wrapper object on every call — the actual 16 B/step found in issue #11.
    i = 1
    for (node_id, _) in tracker.active_nodes
        nodes[i] = node_id
        i += 1
    end
    sort!(nodes; alg = QuickSort)

    # Build cumulative weights in canonical order.
    running = 0
    @inbounds for j in 1:n_active
        running += tracker.active_nodes[nodes[j]]
        cumw[j] = running
    end

    total_weight = running
    if total_weight <= 0
        throw(ArgumentError("Total weight is zero - no valid nodes to sample"))
    end

    # Binary search for efficiency (one rand() call, searchsortedfirst over the
    # cumulative weights).
    random_value = rand(rng) * total_weight
    idx = searchsortedfirst(cumw, random_value)

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