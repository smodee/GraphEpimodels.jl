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

@inline function num_nodes(graph::CycleGraph)::Int
    return graph.n_nodes
end

function node_states_raw(graph::CycleGraph)::Vector{Int8}
    return graph.states
end

function set_node_states_raw!(graph::CycleGraph, states::Vector{Int8})
    if length(states) != num_nodes(graph)
        throw(ArgumentError("Expected $(num_nodes(graph)) states, got $(length(states))"))
    end
    graph.states = states
end

function get_neighbors(graph::CycleGraph, node_id::Int)::Vector{Int}
    return get_neighbors!(Int[], graph, node_id)
end

function get_neighbors!(neighbors::Vector{Int}, graph::CycleGraph, node_id::Int)::Vector{Int}
    n = graph.n_nodes
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
    empty!(neighbors)
    push!(neighbors, _cycle_prev(node_id, n))
    push!(neighbors, _cycle_next(node_id, n))
    return neighbors
end

# Degree is constant: every node on the ring has exactly 2 neighbors.
@inline function get_node_degree(graph::CycleGraph, node_id::Int)::Int
    if node_id < 1 || node_id > graph.n_nodes
        throw(BoundsError("Node ID $node_id out of range [1, $(graph.n_nodes)]"))
    end
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
    n = graph.n_nodes
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
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

has_layout(::CycleGraph)::Bool = true
layout_dim(::CycleGraph)::Int = 2

function node_positions(graph::CycleGraph)::Matrix{Float64}
    n = graph.n_nodes
    pos = Matrix{Float64}(undef, 2, n)
    @inbounds for idx in 1:n
        θ = 2π * (idx - 1) / n
        pos[1, idx] = cos(θ)
        pos[2, idx] = sin(θ)
    end
    return pos
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
