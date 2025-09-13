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
"""
mutable struct AdjacencyGraph <: AbstractEpidemicGraph
    adjacency_list::Vector{Vector{Int}}
    n_nodes::Int
    states::Vector{Int8}
    node_degrees::Vector{Int}  # Pre-computed for performance
    
    function AdjacencyGraph(adjacency_list::Vector{Vector{Int}})
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
        
        new(adjacency_list, n, states, node_degrees)
    end
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
Validate adjacency list structure and detect common errors.
"""
function _validate_adjacency_list(adjacency_list::Vector{Vector{Int}})
    n = length(adjacency_list)
    
    for (node_id, neighbors) in enumerate(adjacency_list)
        for neighbor in neighbors
            # Check bounds
            if neighbor < 1 || neighbor > n
                throw(ArgumentError("Invalid neighbor $neighbor for node $node_id (must be in [1, $n])"))
            end
            
            # Check for self-loops
            if neighbor == node_id
                throw(ArgumentError("Self-loop detected: node $node_id lists itself as neighbor"))
            end
        end
        
        # Check for duplicates
        if length(neighbors) != length(unique(neighbors))
            duplicates = [x for x in neighbors if count(==(x), neighbors) > 1]
            throw(ArgumentError("Duplicate neighbors detected for node $node_id: $duplicates"))
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
function create_graph_from_matrix(adj_matrix::Matrix{Bool})::AdjacencyGraph
    n = size(adj_matrix, 1)
    if size(adj_matrix, 2) != n
        throw(ArgumentError("Adjacency matrix must be square, got $(size(adj_matrix))"))
    end
    
    adjacency_list = Vector{Vector{Int}}(undef, n)
    
    for i in 1:n
        adjacency_list[i] = findall(adj_matrix[i, :])
    end
    
    return AdjacencyGraph(adjacency_list)
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
function create_graph_from_edges(n_nodes::Int, edges::Vector{Tuple{Int, Int}})::AdjacencyGraph
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
    
    return AdjacencyGraph(adjacency_list)
end

"""
Create complete graph (all nodes connected to all others).

# Arguments
- `n::Int`: Number of nodes

# Returns
- `AdjacencyGraph`: Complete graph

# Example
```julia
julia> complete = create_complete_graph(5)  # K_5
```
"""
function create_complete_graph(n::Int)::AdjacencyGraph
    if n < 1
        throw(ArgumentError("Number of nodes must be positive"))
    end
    if n > 10_000
        @warn "Creating complete graph with $n nodes ($(n*(n-1)÷2) edges) - this may use significant memory"
    end
    
    adjacency_list = Vector{Vector{Int}}(undef, n)
    
    for i in 1:n
        adjacency_list[i] = [j for j in 1:n if j != i]
    end
    
    return AdjacencyGraph(adjacency_list)
end

"""
Create path graph (nodes connected in a line: 1-2-3-...-n).

# Arguments
- `n::Int`: Number of nodes

# Returns
- `AdjacencyGraph`: Path graph

# Example
```julia
julia> path = create_path_graph(100)  # Linear chain
```
"""
function create_path_graph(n::Int)::AdjacencyGraph
    if n < 1
        throw(ArgumentError("Number of nodes must be positive"))
    end
    
    adjacency_list = Vector{Vector{Int}}(undef, n)
    
    for i in 1:n
        neighbors = Int[]
        if i > 1
            push!(neighbors, i-1)
        end
        if i < n
            push!(neighbors, i+1)
        end
        adjacency_list[i] = neighbors
    end
    
    return AdjacencyGraph(adjacency_list)
end

"""
Create cycle graph (path with ends connected: 1-2-3-...-n-1).

# Arguments
- `n::Int`: Number of nodes

# Returns
- `AdjacencyGraph`: Cycle graph

# Example
```julia
julia> cycle = create_cycle_graph(10)  # 10-node ring
```
"""
function create_cycle_graph(n::Int)::AdjacencyGraph
    if n < 3
        throw(ArgumentError("Cycle graph needs at least 3 nodes"))
    end
    
    adjacency_list = Vector{Vector{Int}}(undef, n)
    
    for i in 1:n
        neighbors = Int[]
        # Previous node (with wraparound)
        prev = i == 1 ? n : i - 1
        push!(neighbors, prev)
        # Next node (with wraparound)
        next = i == n ? 1 : i + 1
        push!(neighbors, next)
        adjacency_list[i] = neighbors
    end
    
    return AdjacencyGraph(adjacency_list)
end

"""
Create star graph (one central node connected to all others).

# Arguments
- `n::Int`: Total number of nodes

# Returns  
- `AdjacencyGraph`: Star graph (node 1 is the center)

# Example
```julia
julia> star = create_star_graph(6)  # 1 center + 5 leaves
```
"""
function create_star_graph(n::Int)::AdjacencyGraph
    if n < 2
        throw(ArgumentError("Star graph needs at least 2 nodes"))
    end
    
    adjacency_list = Vector{Vector{Int}}(undef, n)
    
    # Center node (node 1) connects to all others
    adjacency_list[1] = collect(2:n)
    
    # All other nodes connect only to center
    for i in 2:n
        adjacency_list[i] = [1]
    end
    
    return AdjacencyGraph(adjacency_list)
end

"""
Create random Erdős-Rényi graph.

# Arguments
- `n::Int`: Number of nodes
- `p::Float64`: Probability of edge between any two nodes
- `rng::AbstractRNG`: Random number generator

# Returns
- `AdjacencyGraph`: Random graph

# Example
```julia
julia> random_graph = create_random_graph(100, 0.1)  # 100 nodes, 10% edge probability
```
"""
function create_random_graph(n::Int, p::Float64, 
                            rng::AbstractRNG = Random.default_rng())::AdjacencyGraph
    if n < 1
        throw(ArgumentError("Number of nodes must be positive"))
    end
    if p < 0 || p > 1
        throw(ArgumentError("Edge probability must be in [0, 1]"))
    end
    
    adjacency_list = [Int[] for _ in 1:n]
    
    # Add edges with probability p
    for i in 1:n
        for j in (i+1):n  # Only check upper triangle to avoid duplicates
            if rand(rng) < p
                push!(adjacency_list[i], j)
                push!(adjacency_list[j], i)
            end
        end
    end
    
    # Sort neighbor lists
    for neighbors in adjacency_list
        sort!(neighbors)
    end
    
    return AdjacencyGraph(adjacency_list)
end

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