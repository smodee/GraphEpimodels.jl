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

"""
Abstract base type for regular lattice graphs (square, triangular, hexagonal).

Lattice graphs share a key property: their connectivity is *implicit* in a
regular geometric arrangement, so neighbors are computed by O(1) coordinate
arithmetic rather than stored as explicit adjacency lists. They also carry an
intrinsic spatial layout (see the geometry interface below), which the
visualization layer uses to draw them as dual-tiling cells.

This is a sibling of `AdjacencyGraph` (which stores explicit adjacency lists and
has no intrinsic geometry), not a supertype of it: lattices deliberately avoid
materializing adjacency lists.
"""
abstract type AbstractLatticeGraph <: AbstractEpidemicGraph end

"""
Pre-compute the perimeter node indices of a `width × height` rectangular array.

A node is on the perimeter if it sits in the first/last row or column. Shared by
lattice constructors for absorbing-boundary node lists; `coord_to_index(row,
col)` maps a 1-indexed coordinate to the lattice's linear index.
"""
# Half of √3, the vertical pitch shared by the triangular/hexagonal lattices.
const _SQRT3_2 = sqrt(3) / 2

function _compute_perimeter_nodes(width::Int, height::Int, coord_to_index)::Vector{Int}
    nodes = Int[]
    for row in 1:height, col in 1:width
        if row == 1 || row == height || col == 1 || col == width
            push!(nodes, coord_to_index(row, col))
        end
    end
    return nodes
end

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
Non-allocating variant of [`get_neighbors`](@ref).

Returns the neighbors of `node_id` as a vector that callers must treat as
**read-only**. Concrete graph types may either fill and return the supplied
`buffer` (so repeated calls reuse one allocation) or return an internally stored
neighbor vector directly. Either way the returned vector must not be mutated by
the caller.

This exists so performance-critical event loops (e.g. the ZIM step) can walk a
node's neighbors without allocating a fresh `Vector{Int}` on every step — a major
source of multithreaded GC pressure (issues #1 / #3).

The default implementation falls back to the allocating [`get_neighbors`](@ref),
so it is always correct; graph types override it where a cheaper path exists.

# Arguments
- `buffer::Vector{Int}`: Caller-owned scratch buffer that may be reused
- `graph::AbstractEpidemicGraph`: The graph
- `node_id::Int`: Node identifier (1-indexed)

# Returns
- `Vector{Int}`: Read-only neighbor list (possibly `buffer`, possibly internal)
"""
function get_neighbors!(buffer::Vector{Int}, graph::AbstractEpidemicGraph, node_id::Int)::Vector{Int}
    return get_neighbors(graph, node_id)
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
# Geometry Interface (Optional - consumed by the visualization layer)
# =============================================================================
#
# Topology (who connects to whom) is what the simulation engine needs and is
# covered by the interface above. Geometry (where each node sits in space) is
# needed only for plotting, so it lives in this separate, optional interface
# with safe defaults: a graph with no intrinsic layout (e.g. a bare
# AdjacencyGraph) needs to implement nothing, and the visualizer falls back to a
# computed layout.

"""
Whether the graph carries an intrinsic (or attached) spatial layout.

Default: `false`. Lattices return `true`; an `AdjacencyGraph` returns `true`
only when node coordinates have been attached.
"""
function has_layout(graph::AbstractEpidemicGraph)::Bool
    return false
end

"""
Dimensionality of the node layout (2 or 3); `0` when there is no layout.

Default: `0`. 2D graphs return `2`; this becomes `3` for future 3D graphs with
no change to the rest of the interface.
"""
function layout_dim(graph::AbstractEpidemicGraph)::Int
    return 0
end

"""
Node coordinates as a `dim × N` matrix (column `i` is the position of node `i`).

Using `dim × N` (rather than `N × dim`) keeps each node's coordinates
contiguous in Julia's column-major storage and makes the 3D extension a pure
shape change (`2 × N` → `3 × N`).

No default: graphs that report `has_layout() == true` must implement this.
"""
function node_positions(graph::AbstractEpidemicGraph)::Matrix{Float64}
    error("node_positions must be implemented by graph type $(typeof(graph)) " *
          "that reports has_layout() == true")
end

"""
Whether the graph fills space with one polygonal cell per node (a tiling).

Default: `false`. Regular lattices return `true` and supply `cell_polygons`;
general networks return `false` and are drawn as node-link diagrams instead.
"""
function has_cells(graph::AbstractEpidemicGraph)::Bool
    return false
end

"""
Per-node cell polygons for tiling visualization.

Returns one entry per node; each entry is a `2 × V` matrix whose columns are the
polygon vertices (in order) of that node's cell. The cell is the *dual* of the
lattice, so the number of vertices equals the node's degree and each cell edge
corresponds to one outgoing graph edge (square→square, triangular→hexagon,
hexagonal→triangle).

No default: graphs that report `has_cells() == true` must implement this.
"""
function cell_polygons(graph::AbstractEpidemicGraph)::Vector{Matrix{Float64}}
    error("cell_polygons must be implemented by graph type $(typeof(graph)) " *
          "that reports has_cells() == true")
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
