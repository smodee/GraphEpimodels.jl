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

@inline function num_nodes(graph::CompleteGraph)::Int
    return graph.n_nodes
end

function node_states_raw(graph::CompleteGraph)::Vector{Int8}
    return graph.states
end

function set_node_states_raw!(graph::CompleteGraph, states::Vector{Int8})
    if length(states) != num_nodes(graph)
        throw(ArgumentError("Expected $(num_nodes(graph)) states, got $(length(states))"))
    end
    graph.states = states
end

function get_neighbors(graph::CompleteGraph, node_id::Int)::Vector{Int}
    return get_neighbors!(Int[], graph, node_id)
end

function get_neighbors!(neighbors::Vector{Int}, graph::CompleteGraph, node_id::Int)::Vector{Int}
    n = graph.n_nodes
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
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
    if node_id < 1 || node_id > graph.n_nodes
        throw(BoundsError("Node ID $node_id out of range [1, $(graph.n_nodes)]"))
    end
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
    if node_id < 1 || node_id > graph.n_nodes
        throw(BoundsError("Node ID $node_id out of range [1, $(graph.n_nodes)]"))
    end
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
# the nodes evenly on a circle so every edge is visible. Providing this gives a
# deterministic layout instead of falling back to a computed (force-directed) one.
# It is not a space-filling tiling, so it supplies no cells.

has_layout(::CompleteGraph)::Bool = true
layout_dim(::CompleteGraph)::Int = 2

function node_positions(graph::CompleteGraph)::Matrix{Float64}
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
