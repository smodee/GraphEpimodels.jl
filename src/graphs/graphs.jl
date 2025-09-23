"""
Graph interface and general implementations for epidemic modeling.

This module defines the abstract graph interface that all graph types must implement,
along with efficient primitive state management and general graph implementations.
"""

using Random

# =============================================================================
# State Constants and Conversion (High Performance)
# =============================================================================

# Primitive state constants (Int8 for maximum performance)
const STATE_SUSCEPTIBLE = Int8(0)
const STATE_INFECTED = Int8(1)
const STATE_REMOVED = Int8(2)

# Public enum for type safety in external API
@enum NodeState::Int8 SUSCEPTIBLE=0 INFECTED=1 REMOVED=2

# Convenient aliases
const S = SUSCEPTIBLE
const I = INFECTED  
const R = REMOVED

# Zero-cost conversion functions (inlined by compiler)
@inline state_to_int(state::NodeState)::Int8 = Int8(state)
@inline int_to_state(val::Int8)::NodeState = NodeState(val)

# =============================================================================
# Abstract Graph Interface
# =============================================================================

"""
Abstract base type for all epidemic graph implementations.

All concrete graph types must implement the core interface methods.
This design allows epidemic processes to work with any graph structure
(lattices, networks, trees, etc.) through a common interface.
"""
abstract type AbstractEpidemicGraph end

# =============================================================================
# Core Interface Methods (REQUIRED - must be implemented by all graph types)
# =============================================================================

"""
Get the total number of nodes in the graph.

# Returns
- `Int`: Number of nodes
"""
function num_nodes(graph::AbstractEpidemicGraph)::Int
    error("num_nodes must be implemented by concrete graph type $(typeof(graph))")
end

"""
Get the neighbors of a specific node.

# Arguments
- `graph::AbstractEpidemicGraph`: The graph
- `node_id::Int`: Node identifier (1-indexed)

# Returns
- `Vector{Int}`: Vector of neighbor node IDs
"""
function get_neighbors(graph::AbstractEpidemicGraph, node_id::Int)::Vector{Int}
    error("get_neighbors must be implemented by concrete graph type $(typeof(graph))")
end

"""
Get the raw node states as primitive Int8 array (internal, performance-critical).

This is the internal representation used for all performance-critical operations.
Use node_states() for the external API with NodeState enums.

# Returns
- `Vector{Int8}`: Raw state array (STATE_SUSCEPTIBLE=0, STATE_INFECTED=1, STATE_REMOVED=2)
"""
function node_states_raw(graph::AbstractEpidemicGraph)::Vector{Int8}
    error("node_states_raw must be implemented by concrete graph type $(typeof(graph))")
end

"""
Set all node states using primitive Int8 array (internal, performance-critical).

# Arguments
- `states::Vector{Int8}`: Raw state array
"""
function set_node_states_raw!(graph::AbstractEpidemicGraph, states::Vector{Int8})
    error("set_node_states_raw! must be implemented by concrete graph type $(typeof(graph))")
end

# =============================================================================
# Optional Interface Methods (have default implementations)
# =============================================================================

"""
Get the degree (number of neighbors) of a node.

Default implementation uses get_neighbors(), but can be overridden for efficiency.

# Arguments
- `graph::AbstractEpidemicGraph`: The graph  
- `node_id::Int`: Node identifier

# Returns
- `Int`: Number of neighbors
"""
function get_node_degree(graph::AbstractEpidemicGraph, node_id::Int)::Int
    return length(get_neighbors(graph, node_id))
end

"""
Get boundary nodes for graphs with boundary concepts (e.g., lattices).

Default implementation returns empty vector (most graphs have no boundary).
Lattices and similar structures should override this method.

# Returns
- `Vector{Int}`: Vector of boundary node IDs
"""
function get_boundary_nodes(graph::AbstractEpidemicGraph)::Vector{Int}
    return Int[]
end

"""
Check if the graph has a boundary concept.

# Returns
- `Bool`: true if graph has boundary nodes
"""
function has_boundary(graph::AbstractEpidemicGraph)::Bool
    return !isempty(get_boundary_nodes(graph))
end

# =============================================================================
# Public API (External Interface with Type Safety)  
# =============================================================================

"""
Get node states as NodeState enums (external API).

This provides type safety for external code while using efficient 
primitive arrays internally.

# Returns
- `Vector{NodeState}`: Node states as enums
"""
function node_states(graph::AbstractEpidemicGraph)::Vector{NodeState}
    return NodeState.(node_states_raw(graph))
end

"""
Set node states using NodeState enums (external API).

# Arguments
- `states::Vector{NodeState}`: Node states as enums
"""
function set_node_states!(graph::AbstractEpidemicGraph, states::Vector{NodeState})
    if length(states) != num_nodes(graph)
        throw(ArgumentError("Expected $(num_nodes(graph)) states, got $(length(states))"))
    end
    set_node_states_raw!(graph, Int8.(states))
end

"""
Get the state of a specific node (external API).

# Arguments
- `node_id::Int`: Node identifier

# Returns
- `NodeState`: Node state as enum
"""
function get_node_state(graph::AbstractEpidemicGraph, node_id::Int)::NodeState
    states = node_states_raw(graph)
    if node_id < 1 || node_id > length(states)
        throw(BoundsError("Node ID $node_id out of range [1, $(length(states))]"))
    end
    return int_to_state(states[node_id])
end

"""
Set the state of a specific node (external API).

# Arguments  
- `node_id::Int`: Node identifier
- `state::NodeState`: New state
"""
function set_node_state!(graph::AbstractEpidemicGraph, node_id::Int, state::NodeState)
    states = node_states_raw(graph)
    if node_id < 1 || node_id > length(states)
        throw(BoundsError("Node ID $node_id out of range [1, $(length(states))]"))
    end
    states[node_id] = state_to_int(state)
end

# =============================================================================
# Derived Functions (High-Performance Implementations)
# =============================================================================

"""
Count nodes in each state using primitive arrays for maximum performance.

# Returns
- `Dict{NodeState, Int}`: Mapping from states to counts
"""
function count_states(graph::AbstractEpidemicGraph)::Dict{NodeState, Int}
    states = node_states_raw(graph)
    
    # Vectorized counting on primitive array (very fast)
    susceptible_count = count(==(STATE_SUSCEPTIBLE), states)
    infected_count = count(==(STATE_INFECTED), states)
    removed_count = count(==(STATE_REMOVED), states)
    
    return Dict{NodeState, Int}(
        SUSCEPTIBLE => susceptible_count,
        INFECTED => infected_count,
        REMOVED => removed_count
    )
end

"""
Get all nodes currently in a specific state using vectorized search.

# Arguments
- `state::NodeState`: Target state

# Returns
- `Vector{Int}`: Vector of node IDs in the specified state
"""
function get_nodes_in_state(graph::AbstractEpidemicGraph, state::NodeState)::Vector{Int}
    states = node_states_raw(graph)
    target_int = state_to_int(state)
    return findall(==(target_int), states)  # Vectorized search on Int8
end

"""
Count neighbors of a node in a specific state (performance-optimized).

# Arguments
- `node_id::Int`: Node to query
- `target_state::NodeState`: State to count

# Returns
- `Int`: Number of neighbors in target state
"""
function count_neighbors_by_state(graph::AbstractEpidemicGraph, node_id::Int, 
                                 target_state::NodeState)::Int
    neighbors = get_neighbors(graph, node_id)
    if isempty(neighbors)
        return 0
    end
    
    states = node_states_raw(graph)
    target_int = state_to_int(target_state)
    
    count = 0
    @inbounds for neighbor in neighbors
        if states[neighbor] == target_int
            count += 1
        end
    end
    return count
end

"""
Get active edges (connections between nodes that can interact).

For epidemic processes, this typically means infected-susceptible pairs.

# Arguments
- `from_state::NodeState`: Source state (default: INFECTED)
- `to_state::NodeState`: Target state (default: SUSCEPTIBLE)

# Returns
- `Vector{Tuple{Int, Int}}`: Vector of (from_node, to_node) pairs
"""
function get_active_edges(graph::AbstractEpidemicGraph, 
                         from_state::NodeState = INFECTED,
                         to_state::NodeState = SUSCEPTIBLE)::Vector{Tuple{Int, Int}}
    active_edges = Tuple{Int, Int}[]
    states = node_states_raw(graph)
    from_int = state_to_int(from_state)
    to_int = state_to_int(to_state)
    
    # Only check nodes in the from_state
    for from_node in 1:length(states)
        if states[from_node] == from_int
            neighbors = get_neighbors(graph, from_node)
            for neighbor in neighbors
                if states[neighbor] == to_int
                    push!(active_edges, (from_node, neighbor))
                end
            end
        end
    end
    
    return active_edges
end
