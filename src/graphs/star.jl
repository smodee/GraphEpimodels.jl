"""
Star graph implementation for epidemic modeling.

A *star graph* `S_n` has one central node (node 1) connected to every other node,
and the `n-1` leaves connected only to the center. Its connectivity is fully
determined by `n`, so it is an [`AbstractImplicitGraph`](@ref): neighbors are
computed on demand rather than stored. The center has degree `n-1`; every leaf has
degree 1.

Like the complete graph, the center neighbors every other node, so the center's
neighbor-by-state count is the global state count minus the center itself — an
allocation-free scan with no adjacency list (see `count_neighbors_by_state`).
"""

using Random

# =============================================================================
# Star Graph
# =============================================================================

"""
Star graph `S_n`: central node 1 connected to all others; leaves connect only to
the center.

Stores only the node count and the primitive `Int8` state vector; neighbors are
computed on demand.

# Fields
- `n_nodes::Int`: Number of nodes (1 center + `n-1` leaves)
- `states::Vector{Int8}`: Node states (primitive array)
"""
mutable struct StarGraph <: AbstractImplicitGraph
    n_nodes::Int
    states::Vector{Int8}

    function StarGraph(n::Int)
        if n < 2
            throw(ArgumentError("Star graph needs at least 2 nodes"))
        end
        new(n, zeros(Int8, n))  # All start SUSCEPTIBLE
    end
end

# =============================================================================
# Core Interface Implementation (Required Methods)
# =============================================================================

function get_neighbors!(neighbors::Vector{Int}, graph::StarGraph, node_id::Int)::Vector{Int}
    _check_node(graph, node_id)
    n = graph.n_nodes
    empty!(neighbors)
    if node_id == 1
        # Center: every leaf (nodes 2..n).
        sizehint!(neighbors, n - 1)
        @inbounds for j in 2:n
            push!(neighbors, j)
        end
    else
        # Leaf: only the center.
        push!(neighbors, 1)
    end
    return neighbors
end

# Degree: n-1 at the center (node 1), 1 at every leaf.
@inline function get_node_degree(graph::StarGraph, node_id::Int)::Int
    _check_node(graph, node_id)
    n = graph.n_nodes
    return node_id == 1 ? n - 1 : 1
end

# =============================================================================
# Performance-Optimized Neighbor Counting
# =============================================================================

"""
Allocation-free neighbor counting for a star.

- A leaf has the single neighbor `1` (the center), so the count is whether the
  center is in `target_state`.
- The center neighbors every leaf, so the count is the global number of nodes in
  `target_state`, minus the center itself if it is in that state.
"""
function count_neighbors_by_state(graph::StarGraph, node_id::Int,
                                  target_state::NodeState)::Int
    _check_node(graph, node_id)
    n = graph.n_nodes
    states = graph.states
    target_int = state_to_int(target_state)
    if node_id != 1
        @inbounds return states[1] == target_int ? 1 : 0
    end
    # Center: all leaves = every node except the center.
    total = count(==(target_int), states)
    @inbounds return total - (states[1] == target_int ? 1 : 0)
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# The center sits at the origin and the leaves are spread evenly around it: on
# the unit circle in 2D, on the unit sphere (Fibonacci spiral) in 3D. It is not a
# space-filling tiling, so it supplies no cells.

supported_layout_dims(::StarGraph)::Tuple{Vararg{Int}} = (2, 3)

function node_positions(graph::StarGraph; dim::Int = 2)::Matrix{Float64}
    _check_layout_dim(graph, dim)
    n = graph.n_nodes
    if dim == 3
        # Center at the origin; leaves on the unit sphere.
        pos = Matrix{Float64}(undef, 3, n)
        @inbounds pos[:, 1] .= 0.0
        @inbounds pos[:, 2:n] .= _fibonacci_sphere(n - 1)
        return pos
    end
    # Center at the origin; leaves evenly spaced on the unit circle.
    pos = Matrix{Float64}(undef, 2, n)
    @inbounds begin
        pos[1, 1] = 0.0
        pos[2, 1] = 0.0
        pos[:, 2:n] .= _circle_layout(n - 1)
    end
    return pos
end

# =============================================================================
# Factory Function
# =============================================================================

"""
Create star graph (one central node connected to all others).

# Arguments
- `n::Int`: Total number of nodes (node 1 is the center)

# Returns
- `StarGraph`: Star graph `S_n`

# Example
```julia
julia> star = create_star_graph(6)  # 1 center + 5 leaves
```
"""
function create_star_graph(n::Int)::StarGraph
    return StarGraph(n)
end
