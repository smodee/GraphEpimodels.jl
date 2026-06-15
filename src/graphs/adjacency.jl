"""
General graph implementation using adjacency lists.

Provides flexible graph representation for networks, trees, and arbitrary
graph structures. Optimized for epidemic modeling with primitive Int8 states.
"""

using Random

# Import graph interface (assumes graphs.jl is loaded)

# =============================================================================
# General Adjacency List Graph
# =============================================================================

"""
General graph implementation using adjacency list representation.

Supports any graph topology: networks, trees, complete graphs, etc.
Uses primitive Int8 states for maximum performance and minimal memory usage.

# Fields
- `adjacency_list::Vector{Vector{Int}}`: Neighbor lists for each node
- `n_nodes::Int`: Total number of nodes (pre-computed)
- `states::Vector{Int8}`: Node states (primitive array)
- `node_degrees::Vector{Int}`: Pre-computed node degrees (optional optimization)
- `coords::Union{Matrix{Float64}, Nothing}`: Optional `dim × n` node coordinates
  for plotting (column `i` = position of node `i`). `nothing` means no layout,
  and the visualizer computes one (e.g. force-directed).
"""
mutable struct AdjacencyGraph <: AbstractEpidemicGraph
    adjacency_list::Vector{Vector{Int}}
    n_nodes::Int
    states::Vector{Int8}
    node_degrees::Vector{Int}  # Pre-computed for performance
    coords::Union{Matrix{Float64}, Nothing}

    function AdjacencyGraph(adjacency_list::Vector{Vector{Int}};
                            coords::Union{AbstractMatrix{<:Real}, Nothing} = nothing)
        n = length(adjacency_list)

        if n == 0
            throw(ArgumentError("Graph must have at least one node"))
        end

        # Validate adjacency list structure
        _validate_adjacency_list(adjacency_list)

        # Pre-compute node degrees for performance
        node_degrees = [length(neighbors) for neighbors in adjacency_list]

        # Initialize all nodes as susceptible
        states = zeros(Int8, n)

        coords_mat = _validate_coords(coords, n)

        new(adjacency_list, n, states, node_degrees, coords_mat)
    end
end

"""Validate optional node coordinates and normalize to `Matrix{Float64}` (dim × n)."""
function _validate_coords(coords::Union{AbstractMatrix{<:Real}, Nothing}, n::Int)
    coords === nothing && return nothing
    dim = size(coords, 1)
    if dim != 2 && dim != 3
        throw(ArgumentError("coords must have 2 or 3 rows (got $dim); shape is dim × n"))
    end
    if size(coords, 2) != n
        throw(ArgumentError("coords must have one column per node: expected $n, got $(size(coords, 2))"))
    end
    return Matrix{Float64}(coords)
end

# =============================================================================
# Core Interface Implementation
# =============================================================================

@inline function num_nodes(graph::AdjacencyGraph)::Int
    return graph.n_nodes
end

function get_neighbors(graph::AdjacencyGraph, node_id::Int)::Vector{Int}
    if node_id < 1 || node_id > graph.n_nodes
        throw(BoundsError("Node ID $node_id out of range [1, $(graph.n_nodes)]"))
    end
    return graph.adjacency_list[node_id]
end

# The adjacency list is already stored, so get_neighbors returns it without
# allocating. The buffer-filling variant therefore just hands back the stored
# (read-only) vector and ignores the scratch buffer.
function get_neighbors!(::Vector{Int}, graph::AdjacencyGraph, node_id::Int)::Vector{Int}
    return get_neighbors(graph, node_id)
end

@inline function get_node_degree(graph::AdjacencyGraph, node_id::Int)::Int
    if node_id < 1 || node_id > graph.n_nodes
        throw(BoundsError("Node ID $node_id out of range [1, $(graph.n_nodes)]"))
    end
    return graph.node_degrees[node_id]
end

function node_states_raw(graph::AdjacencyGraph)::Vector{Int8}
    return graph.states
end

function set_node_states_raw!(graph::AdjacencyGraph, states::Vector{Int8})
    if length(states) != graph.n_nodes
        throw(ArgumentError("Expected $(graph.n_nodes) states, got $(length(states))"))
    end
    graph.states = states
end

# Boundary nodes: empty for general graphs (no spatial boundary concept)
function get_boundary_nodes(graph::AdjacencyGraph)::Vector{Int}
    return Int[]
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# A general graph has a layout only if coordinates were attached at construction.
# It never fills space with cells, so the visualizer draws it as a node-link
# diagram (markers + edges), computing a layout when none is attached.

has_layout(graph::AdjacencyGraph)::Bool = graph.coords !== nothing
layout_dim(graph::AdjacencyGraph)::Int = graph.coords === nothing ? 0 : size(graph.coords, 1)

function node_positions(graph::AdjacencyGraph)::Matrix{Float64}
    graph.coords === nothing &&
        error("AdjacencyGraph has no attached coordinates; has_layout() is false")
    return graph.coords
end

"""Attach (or replace) node coordinates on an existing graph (`dim × n`)."""
function set_coords!(graph::AdjacencyGraph, coords::AbstractMatrix{<:Real})
    graph.coords = _validate_coords(coords, graph.n_nodes)
    return graph
end

# =============================================================================
# Performance-Optimized Neighbor Counting
# =============================================================================

"""
Optimized neighbor counting for general graphs.
Uses pre-computed degrees and direct array access for maximum performance.
"""
function count_neighbors_by_state(graph::AdjacencyGraph, node_id::Int, 
                                 target_state::NodeState)::Int
    if graph.node_degrees[node_id] == 0
        return 0  # No neighbors
    end
    
    neighbors = graph.adjacency_list[node_id]
    states = graph.states
    target_int = state_to_int(target_state)
    
    count = 0
    @inbounds for neighbor in neighbors
        if states[neighbor] == target_int
            count += 1
        end
    end
    
    return count
end

# =============================================================================
# Input Validation
# =============================================================================

"""
Validate adjacency list structure and detect common errors (bounds, self-loops,
duplicate neighbors) in a single allocation-free pass.

Duplicates are found with the "last seen" timestamp trick: `last_seen[nb]` records
the node whose neighbor list `nb` was most recently seen in, so a repeat within the
same list is a duplicate. Using `node_id` as the timestamp means the scratch buffer
never needs clearing between nodes — the whole check costs one `Vector{Int}` of
length `n` and is O(total degree).
"""
function _validate_adjacency_list(adjacency_list::Vector{Vector{Int}})
    n = length(adjacency_list)
    last_seen = zeros(Int, n)

    for (node_id, neighbors) in enumerate(adjacency_list)
        for neighbor in neighbors
            if neighbor < 1 || neighbor > n
                throw(ArgumentError("Invalid neighbor $neighbor for node $node_id (must be in [1, $n])"))
            end
            if neighbor == node_id
                throw(ArgumentError("Self-loop detected: node $node_id lists itself as neighbor"))
            end
            if last_seen[neighbor] == node_id
                throw(ArgumentError("Duplicate neighbor $neighbor detected for node $node_id"))
            end
            last_seen[neighbor] = node_id
        end
    end
end

# =============================================================================
# Factory Functions for Common Graph Types
# =============================================================================

"""
Create graph from adjacency matrix.

# Arguments
- `adj_matrix::Matrix{Bool}`: Adjacency matrix (true = edge exists)

# Returns
- `AdjacencyGraph`: Graph representation

# Example
```julia
julia> matrix = [false true false; true false true; false true false]
julia> graph = create_graph_from_matrix(matrix)  # Path graph: 1-2-3
```
"""
function create_graph_from_matrix(adj_matrix::Matrix{Bool};
                                  coords::Union{AbstractMatrix{<:Real}, Nothing} = nothing)::AdjacencyGraph
    n = size(adj_matrix, 1)
    if size(adj_matrix, 2) != n
        throw(ArgumentError("Adjacency matrix must be square, got $(size(adj_matrix))"))
    end

    adjacency_list = Vector{Vector{Int}}(undef, n)

    for i in 1:n
        adjacency_list[i] = findall(adj_matrix[i, :])
    end

    return AdjacencyGraph(adjacency_list; coords = coords)
end

"""
Create graph from edge list.

# Arguments
- `n_nodes::Int`: Number of nodes
- `edges::Vector{Tuple{Int, Int}}`: List of edges as (from, to) pairs

# Returns
- `AdjacencyGraph`: Graph representation

# Example
```julia
julia> edges = [(1, 2), (2, 3), (3, 1)]  # Triangle
julia> graph = create_graph_from_edges(3, edges)
```
"""
function create_graph_from_edges(n_nodes::Int, edges::Vector{Tuple{Int, Int}};
                                 coords::Union{AbstractMatrix{<:Real}, Nothing} = nothing)::AdjacencyGraph
    if n_nodes < 1
        throw(ArgumentError("Number of nodes must be positive"))
    end
    
    # Initialize empty adjacency lists
    adjacency_list = [Int[] for _ in 1:n_nodes]
    
    # Add edges
    for (from, to) in edges
        if from < 1 || from > n_nodes || to < 1 || to > n_nodes
            throw(ArgumentError("Edge ($from, $to) contains invalid node IDs"))
        end
        if from == to
            throw(ArgumentError("Self-loop not allowed: ($from, $to)"))
        end
        
        # Add undirected edge (both directions)
        if to ∉ adjacency_list[from]
            push!(adjacency_list[from], to)
        end
        if from ∉ adjacency_list[to]
            push!(adjacency_list[to], from)
        end
    end
    
    # Sort neighbor lists for consistent ordering
    for neighbors in adjacency_list
        sort!(neighbors)
    end

    return AdjacencyGraph(adjacency_list; coords = coords)
end

# The named structured graphs each have a dedicated implicit type that stores only
# `n` (neighbors computed on demand), rather than a materialized adjacency list:
#   create_complete_graph -> CompleteGraph (graphs/complete.jl)
#   create_path_graph      -> PathGraph     (graphs/path.jl)
#   create_cycle_graph     -> CycleGraph    (graphs/cycle.jl)
#   create_star_graph      -> StarGraph     (graphs/star.jl)

# Erdős–Rényi random graphs live in their own type/file (graphs/erdos_renyi.jl):
# create_erdos_renyi / create_gnp / create_gnm return an ErdosRenyiGraph.

# =============================================================================
# Graph Analysis Utilities
# =============================================================================

"""
Calculate basic graph statistics.
"""
function graph_statistics(graph::AdjacencyGraph)::Dict{Symbol, Any}
    degrees = graph.node_degrees
    n_edges = sum(degrees) ÷ 2  # Each edge counted twice
    
    return Dict{Symbol, Any}(
        :n_nodes => graph.n_nodes,
        :n_edges => n_edges,
        :min_degree => minimum(degrees),
        :max_degree => maximum(degrees),
        :mean_degree => sum(degrees) / graph.n_nodes,
        :density => 2 * n_edges / (graph.n_nodes * (graph.n_nodes - 1))
    )
end

"""
Check if graph is connected (single component).
"""
function is_connected(graph::AdjacencyGraph)::Bool
    if graph.n_nodes <= 1
        return true
    end
    
    visited = falses(graph.n_nodes)
    queue = [1]  # Start from node 1
    visited[1] = true
    visited_count = 1
    
    # BFS to find all reachable nodes
    while !isempty(queue)
        current = popfirst!(queue)
        for neighbor in graph.adjacency_list[current]
            if !visited[neighbor]
                visited[neighbor] = true
                visited_count += 1
                push!(queue, neighbor)
            end
        end
    end
    
    return visited_count == graph.n_nodes
end