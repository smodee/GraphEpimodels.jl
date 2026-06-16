"""
Cycle graph implementation for epidemic modeling.

A *cycle graph* `C_n` is a path with its ends joined: `1 - 2 - … - n - 1`. Its
connectivity is fully determined by `n`, so it is an [`AbstractImplicitGraph`](@ref):
a node's two neighbors (the previous and next index, with wraparound) are computed
by arithmetic on demand rather than stored. Every node has degree 2.
"""

using Random

# =============================================================================
# Cycle Graph
# =============================================================================

"""
Cycle graph `C_n`: a ring `1 - 2 - … - n - 1` (every node has degree 2).

Stores only the node count and the primitive `Int8` state vector; neighbors are
computed on demand.

# Fields
- `n_nodes::Int`: Number of nodes
- `states::Vector{Int8}`: Node states (primitive array)
"""
mutable struct CycleGraph <: AbstractImplicitGraph
    n_nodes::Int
    states::Vector{Int8}

    function CycleGraph(n::Int)
        if n < 3
            throw(ArgumentError("Cycle graph needs at least 3 nodes"))
        end
        new(n, zeros(Int8, n))  # All start SUSCEPTIBLE
    end
end

# Previous / next index on the ring (1-indexed, with wraparound).
@inline _cycle_prev(node_id::Int, n::Int)::Int = node_id == 1 ? n : node_id - 1
@inline _cycle_next(node_id::Int, n::Int)::Int = node_id == n ? 1 : node_id + 1

# =============================================================================
# Core Interface Implementation (Required Methods)
# =============================================================================

function get_neighbors!(neighbors::Vector{Int}, graph::CycleGraph, node_id::Int)::Vector{Int}
    _check_node(graph, node_id)
    n = graph.n_nodes
    empty!(neighbors)
    push!(neighbors, _cycle_prev(node_id, n))
    push!(neighbors, _cycle_next(node_id, n))
    return neighbors
end

# Degree is constant: every node on the ring has exactly 2 neighbors.
@inline function get_node_degree(graph::CycleGraph, node_id::Int)::Int
    _check_node(graph, node_id)
    return 2
end

# =============================================================================
# Performance-Optimized Neighbor Counting
# =============================================================================

"""
Allocation-free neighbor counting for a cycle: check the two adjacent indices
(with wraparound) directly instead of materializing a neighbor list. `n >= 3`
guarantees the two neighbors are distinct.
"""
function count_neighbors_by_state(graph::CycleGraph, node_id::Int,
                                  target_state::NodeState)::Int
    _check_node(graph, node_id)
    n = graph.n_nodes
    states = graph.states
    target_int = state_to_int(target_state)
    count = 0
    @inbounds begin
        states[_cycle_prev(node_id, n)] == target_int && (count += 1)
        states[_cycle_next(node_id, n)] == target_int && (count += 1)
    end
    return count
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# A cycle is drawn with its nodes evenly spaced on the unit circle, so adjacent
# indices are adjacent on the ring. It is not a space-filling tiling (no cells).

supported_layout_dims(::CycleGraph)::Tuple{Vararg{Int}} = (2,)

function node_positions(graph::CycleGraph; dim::Int = 2)::Matrix{Float64}
    _check_layout_dim(graph, dim)
    return _circle_layout(graph.n_nodes)
end

# =============================================================================
# Factory Function
# =============================================================================

"""
Create cycle graph (path with ends connected: 1-2-3-...-n-1).

# Arguments
- `n::Int`: Number of nodes

# Returns
- `CycleGraph`: Cycle graph `C_n`

# Example
```julia
julia> cycle = create_cycle_graph(10)  # 10-node ring
```
"""
function create_cycle_graph(n::Int)::CycleGraph
    return CycleGraph(n)
end
