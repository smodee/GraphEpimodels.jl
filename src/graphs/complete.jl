"""
Complete graph implementation for epidemic modeling.

A *complete graph* `K_n` connects every pair of distinct nodes. Its connectivity
is fully determined by `n`, so it is an [`AbstractImplicitGraph`](@ref): neighbors
are computed by arithmetic on demand and never stored. This makes the structure
O(n) memory instead of the O(n²) a materialized adjacency list would cost —
`K_n` has `n(n-1)/2` edges, so the explicit form is infeasible past a few
thousand nodes (a 50,000-node `K_n` would store 2.5 billion neighbor entries).

Neighbor counting by state is the simulation hot path. For a complete graph every
node sees all others, so the count of a node's neighbors in some state is just the
global count of that state minus the node itself — an allocation-free linear scan
of the contiguous `Int8` state array, with no adjacency list to gather over.
"""

using Random

# Import the graph interface (assumes graphs.jl is loaded first).

# =============================================================================
# Complete Graph
# =============================================================================

"""
Complete graph `K_n`: every node is adjacent to every other node.

Stores only the node count and the primitive `Int8` state vector; neighbors are
computed on demand (see the file header). All state operations use primitive
arrays for maximum performance, matching the rest of the package.

# Fields
- `n_nodes::Int`: Number of nodes
- `states::Vector{Int8}`: Node states (primitive array)
"""
mutable struct CompleteGraph <: AbstractImplicitGraph
    n_nodes::Int
    states::Vector{Int8}

    function CompleteGraph(n::Int)
        if n < 1
            throw(ArgumentError("Number of nodes must be positive"))
        end
        new(n, zeros(Int8, n))  # All start SUSCEPTIBLE
    end
end

# =============================================================================
# Core Interface Implementation (Required Methods)
# =============================================================================

function get_neighbors!(neighbors::Vector{Int}, graph::CompleteGraph, node_id::Int)::Vector{Int}
    _check_node(graph, node_id)
    n = graph.n_nodes
    empty!(neighbors)
    sizehint!(neighbors, n - 1)
    @inbounds for j in 1:n
        if j != node_id
            push!(neighbors, j)
        end
    end
    return neighbors
end

# Degree is constant: every node connects to all n-1 others.
@inline function get_node_degree(graph::CompleteGraph, node_id::Int)::Int
    _check_node(graph, node_id)
    return graph.n_nodes - 1
end

# =============================================================================
# Performance-Optimized Neighbor Counting
# =============================================================================

"""
Optimized neighbor counting for the complete graph.

Every node neighbors every other node, so the number of a node's neighbors in
`target_state` is the global count of that state, minus one if the node itself is
in that state. This is an allocation-free linear scan of the contiguous state
array — no adjacency list is materialized.
"""
function count_neighbors_by_state(graph::CompleteGraph, node_id::Int,
                                  target_state::NodeState)::Int
    _check_node(graph, node_id)
    target_int = state_to_int(target_state)
    total = count(==(target_int), graph.states)
    # Exclude the node itself (a node is not its own neighbor).
    @inbounds return total - (graph.states[node_id] == target_int ? 1 : 0)
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# A complete graph has no intrinsic embedding, but the conventional drawing places
# the nodes evenly on a circle (2D) or sphere (3D) so connections spread out. This
# gives a deterministic layout instead of falling back to a computed (force-directed)
# one. It is not a space-filling tiling, so it supplies no cells.

supported_layout_dims(::CompleteGraph)::Tuple{Vararg{Int}} = (2, 3)

function node_positions(graph::CompleteGraph; dim::Int = 2)::Matrix{Float64}
    _check_layout_dim(graph, dim)
    n = graph.n_nodes
    # Evenly on the unit circle (2D) or unit sphere (3D).
    return dim == 3 ? _fibonacci_sphere(n) : _circle_layout(n)
end

# =============================================================================
# Factory Function
# =============================================================================

"""
Create complete graph (all nodes connected to all others).

# Arguments
- `n::Int`: Number of nodes

# Returns
- `CompleteGraph`: Complete graph `K_n`

# Example
```julia
julia> complete = create_complete_graph(5)  # K_5
```
"""
function create_complete_graph(n::Int)::CompleteGraph
    return CompleteGraph(n)
end
