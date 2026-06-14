"""
Path graph implementation for epidemic modeling.

A *path graph* `P_n` connects nodes in a line: `1 - 2 - 3 - … - n`. Its
connectivity is fully determined by `n`, so it is an [`AbstractImplicitGraph`](@ref):
a node's neighbors (`i-1` and `i+1`, where they exist) are computed by arithmetic
on demand rather than stored. The two endpoints have degree 1; interior nodes have
degree 2.
"""

using Random

# =============================================================================
# Path Graph
# =============================================================================

"""
Path graph `P_n`: nodes connected in a line `1 - 2 - … - n`.

Stores only the node count and the primitive `Int8` state vector; neighbors are
computed on demand.

# Fields
- `n_nodes::Int`: Number of nodes
- `states::Vector{Int8}`: Node states (primitive array)
"""
mutable struct PathGraph <: AbstractImplicitGraph
    n_nodes::Int
    states::Vector{Int8}

    function PathGraph(n::Int)
        if n < 1
            throw(ArgumentError("Number of nodes must be positive"))
        end
        new(n, zeros(Int8, n))  # All start SUSCEPTIBLE
    end
end

# =============================================================================
# Core Interface Implementation (Required Methods)
# =============================================================================

@inline function num_nodes(graph::PathGraph)::Int
    return graph.n_nodes
end

function node_states_raw(graph::PathGraph)::Vector{Int8}
    return graph.states
end

function set_node_states_raw!(graph::PathGraph, states::Vector{Int8})
    if length(states) != num_nodes(graph)
        throw(ArgumentError("Expected $(num_nodes(graph)) states, got $(length(states))"))
    end
    graph.states = states
end

function get_neighbors(graph::PathGraph, node_id::Int)::Vector{Int}
    return get_neighbors!(Int[], graph, node_id)
end

function get_neighbors!(neighbors::Vector{Int}, graph::PathGraph, node_id::Int)::Vector{Int}
    n = graph.n_nodes
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
    empty!(neighbors)
    node_id > 1 && push!(neighbors, node_id - 1)
    node_id < n && push!(neighbors, node_id + 1)
    return neighbors
end

# Degree: 1 at the two endpoints, 2 in the interior.
@inline function get_node_degree(graph::PathGraph, node_id::Int)::Int
    n = graph.n_nodes
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
    return (node_id > 1) + (node_id < n)
end

# =============================================================================
# Performance-Optimized Neighbor Counting
# =============================================================================

"""
Allocation-free neighbor counting for a path: check the at-most-two adjacent
indices directly instead of materializing a neighbor list.
"""
function count_neighbors_by_state(graph::PathGraph, node_id::Int,
                                  target_state::NodeState)::Int
    n = graph.n_nodes
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
    states = graph.states
    target_int = state_to_int(target_state)
    count = 0
    @inbounds begin
        if node_id > 1 && states[node_id - 1] == target_int
            count += 1
        end
        if node_id < n && states[node_id + 1] == target_int
            count += 1
        end
    end
    return count
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# A path is drawn as collinear nodes: node i sits at (i, 0). It is not a
# space-filling tiling, so it supplies no cells.

has_layout(::PathGraph)::Bool = true
layout_dim(::PathGraph)::Int = 2

function node_positions(graph::PathGraph)::Matrix{Float64}
    n = graph.n_nodes
    pos = Matrix{Float64}(undef, 2, n)
    @inbounds for idx in 1:n
        pos[1, idx] = Float64(idx)
        pos[2, idx] = 0.0
    end
    return pos
end

# =============================================================================
# Factory Function
# =============================================================================

"""
Create path graph (nodes connected in a line: 1-2-3-...-n).

# Arguments
- `n::Int`: Number of nodes

# Returns
- `PathGraph`: Path graph `P_n`

# Example
```julia
julia> path = create_path_graph(100)  # Linear chain
```
"""
function create_path_graph(n::Int)::PathGraph
    return PathGraph(n)
end
