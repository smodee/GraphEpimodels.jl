"""
Square lattice implementation optimized for epidemic simulations.

This module provides high-performance square lattices with different boundary
conditions, optimized for large-scale stochastic simulations on Z².
"""

using Random
using ..GraphEpimodels: EpidemicGraph, NodeState, BoundaryCondition
using ..GraphEpimodels: SUSCEPTIBLE, INFECTED, REMOVED, ABSORBING, PERIODIC
using ..GraphEpimodels: coord_to_index, index_to_coord, apply_periodic_boundary
using ..GraphEpimodels: is_absorbing_boundary, get_boundary_indices
using ..GraphEpimodels: validate_lattice_size

# =============================================================================
# Square Lattice Implementation
# =============================================================================

"""
High-performance square lattice for epidemic simulations.

Supports different boundary conditions and provides fast neighbor lookups
optimized for epidemic processes on Z².

# Fields
- `width::Int`: Width of the lattice (number of columns)
- `height::Int`: Height of the lattice (number of rows)  
- `boundary::BoundaryCondition`: Boundary condition type
- `states::Vector{NodeState}`: Current state of each node
- `boundary_nodes::Vector{Int}`: Pre-computed boundary node indices
"""
mutable struct SquareLattice <: EpidemicGraph
    width::Int
    height::Int
    boundary::BoundaryCondition
    states::Vector{NodeState}
    boundary_nodes::Vector{Int}
    
    function SquareLattice(width::Int, height::Int, boundary::BoundaryCondition = ABSORBING)
        validate_lattice_size(height, width)
        
        num_nodes = width * height
        states = fill(SUSCEPTIBLE, num_nodes)
        
        # Pre-compute boundary nodes for absorbing boundaries
        boundary_nodes = if boundary == ABSORBING
            get_boundary_indices(height, width)
        else
            Int[]  # No boundary nodes for periodic
        end
        
        new(width, height, boundary, states, boundary_nodes)
    end
end

# =============================================================================
# Required EpidemicGraph Interface
# =============================================================================

function num_nodes(lattice::SquareLattice)::Int
    return lattice.width * lattice.height
end

function node_states(lattice::SquareLattice)::Vector{NodeState}
    return lattice.states
end

function set_node_states!(lattice::SquareLattice, states::Vector{NodeState})
    if length(states) != num_nodes(lattice)
        throw(ArgumentError("Expected $(num_nodes(lattice)) states, got $(length(states))"))
    end
    lattice.states = states
end

function get_node_state(lattice::SquareLattice, node_id::Int)::NodeState
    if node_id < 1 || node_id > num_nodes(lattice)
        throw(BoundsError("Node ID $node_id out of range [1, $(num_nodes(lattice))]"))
    end
    return lattice.states[node_id]
end

function set_node_state!(lattice::SquareLattice, node_id::Int, state::NodeState)
    if node_id < 1 || node_id > num_nodes(lattice)
        throw(BoundsError("Node ID $node_id out of range [1, $(num_nodes(lattice))]"))
    end
    lattice.states[node_id] = state
end

function get_boundary_nodes(lattice::SquareLattice)::Vector{Int}
    return copy(lattice.boundary_nodes)
end

# =============================================================================
# Neighbor Operations (Core Performance Critical Functions)
# =============================================================================

"""
Get neighbors of a node (optimized for square lattice).

# Arguments
- `lattice::SquareLattice`: The lattice
- `node_id::Int`: Node index (1-indexed)

# Returns
- `Vector{Int}`: Array of neighbor indices

# Performance Notes
This function is called frequently in epidemic simulations and is optimized
for both absorbing and periodic boundary conditions.
"""
function get_neighbors(lattice::SquareLattice, node_id::Int)::Vector{Int}
    if node_id < 1 || node_id > num_nodes(lattice)
        throw(BoundsError("Node ID $node_id out of range"))
    end
    
    row, col = index_to_coord(node_id, lattice.width)
    neighbors = Int[]
    
    if lattice.boundary == PERIODIC
        # Periodic boundaries - always 4 neighbors
        sizehint!(neighbors, 4)
        
        # North neighbor
        north_row = apply_periodic_boundary(row - 1, lattice.height)
        push!(neighbors, coord_to_index(north_row, col, lattice.width))
        
        # South neighbor  
        south_row = apply_periodic_boundary(row + 1, lattice.height)
        push!(neighbors, coord_to_index(south_row, col, lattice.width))
        
        # West neighbor
        west_col = apply_periodic_boundary(col - 1, lattice.width)
        push!(neighbors, coord_to_index(row, west_col, lattice.width))
        
        # East neighbor
        east_col = apply_periodic_boundary(col + 1, lattice.width)
        push!(neighbors, coord_to_index(row, east_col, lattice.width))
        
    else  # ABSORBING boundaries
        # Variable number of neighbors (2, 3, or 4)
        sizehint!(neighbors, 4)
        
        # North neighbor
        if row > 1
            push!(neighbors, coord_to_index(row - 1, col, lattice.width))
        end
        
        # South neighbor
        if row < lattice.height
            push!(neighbors, coord_to_index(row + 1, col, lattice.width))
        end
        
        # West neighbor  
        if col > 1
            push!(neighbors, coord_to_index(row, col - 1, lattice.width))
        end
        
        # East neighbor
        if col < lattice.width
            push!(neighbors, coord_to_index(row, col + 1, lattice.width))
        end
    end
    
    return neighbors
end

"""
Get degree (number of neighbors) of a node without allocating neighbor array.

More efficient than `length(get_neighbors(lattice, node_id))` for degree queries.

# Arguments
- `lattice::SquareLattice`: The lattice
- `node_id::Int`: Node index

# Returns
- `Int`: Number of neighbors
"""
function get_node_degree(lattice::SquareLattice, node_id::Int)::Int
    if lattice.boundary == PERIODIC
        return 4  # Always 4 neighbors with periodic boundaries
    else
        # Count neighbors without allocation
        row, col = index_to_coord(node_id, lattice.width)
        degree = 0
        
        if row > 1; degree += 1; end                    # North
        if row < lattice.height; degree += 1; end      # South  
        if col > 1; degree += 1; end                    # West
        if col < lattice.width; degree += 1; end       # East
        
        return degree
    end
end

"""
Count neighbors in a specific state (optimized to avoid allocations).

Critical function for ZIM rate calculations.

# Arguments
- `lattice::SquareLattice`: The lattice
- `node_id::Int`: Node to query
- `target_state::NodeState`: State to count

# Returns
- `Int`: Number of neighbors in target state
"""
function count_neighbors_by_state(lattice::SquareLattice, node_id::Int, 
                                 target_state::NodeState)::Int
    row, col = index_to_coord(node_id, lattice.width)
    count = 0
    
    if lattice.boundary == PERIODIC
        # Check all 4 neighbors with wrapping
        neighbors = [
            coord_to_index(apply_periodic_boundary(row - 1, lattice.height), col, lattice.width),
            coord_to_index(apply_periodic_boundary(row + 1, lattice.height), col, lattice.width), 
            coord_to_index(row, apply_periodic_boundary(col - 1, lattice.width), lattice.width),
            coord_to_index(row, apply_periodic_boundary(col + 1, lattice.width), lattice.width)
        ]
        
        for neighbor in neighbors
            if lattice.states[neighbor] == target_state
                count += 1
            end
        end
        
    else  # ABSORBING boundaries
        # Check neighbors that exist
        if row > 1 && lattice.states[coord_to_index(row - 1, col, lattice.width)] == target_state
            count += 1
        end
        if row < lattice.height && lattice.states[coord_to_index(row + 1, col, lattice.width)] == target_state
            count += 1  
        end
        if col > 1 && lattice.states[coord_to_index(row, col - 1, lattice.width)] == target_state
            count += 1
        end
        if col < lattice.width && lattice.states[coord_to_index(row, col + 1, lattice.width)] == target_state
            count += 1
        end
    end
    
    return count
end

# =============================================================================
# Lattice-Specific Utility Functions
# =============================================================================

"""
Convert lattice coordinates to linear index.

# Arguments  
- `lattice::SquareLattice`: The lattice
- `row::Int`: Row coordinate (1-indexed)
- `col::Int`: Column coordinate (1-indexed)

# Returns
- `Int`: Linear index
"""
function coord_to_index(lattice::SquareLattice, row::Int, col::Int)::Int
    if row < 1 || row > lattice.height || col < 1 || col > lattice.width
        throw(BoundsError("Coordinates ($row, $col) out of bounds"))
    end
    return coord_to_index(row, col, lattice.width)
end

"""
Convert linear index to lattice coordinates.

# Arguments
- `lattice::SquareLattice`: The lattice  
- `index::Int`: Linear index

# Returns
- `Tuple{Int, Int}`: (row, col) coordinates
"""
function index_to_coord(lattice::SquareLattice, index::Int)::Tuple{Int, Int}
    if index < 1 || index > num_nodes(lattice)
        throw(BoundsError("Index $index out of range"))
    end
    return index_to_coord(index, lattice.width)
end

"""
Get the center node of the lattice.

# Arguments
- `lattice::SquareLattice`: The lattice

# Returns  
- `Int`: Index of center node
"""
function get_center_node(lattice::SquareLattice)::Int
    center_row = (lattice.height + 1) ÷ 2
    center_col = (lattice.width + 1) ÷ 2
    return coord_to_index(lattice, center_row, center_col)
end

"""
Get random nodes from the lattice.

# Arguments
- `lattice::SquareLattice`: The lattice
- `count::Int`: Number of nodes to sample
- `rng::AbstractRNG`: Random number generator

# Returns
- `Vector{Int}`: Array of random node indices
"""
function get_random_nodes(lattice::SquareLattice, count::Int, 
                         rng::AbstractRNG = Random.default_rng())::Vector{Int}
    return rand(rng, 1:num_nodes(lattice), count)
end

"""
Get distance from node to nearest boundary.

# Arguments
- `lattice::SquareLattice`: The lattice  
- `node_id::Int`: Node index

# Returns
- `Int`: Distance to boundary (Inf for periodic lattices)
"""
function distance_to_boundary(lattice::SquareLattice, node_id::Int)::Float64
    if lattice.boundary == PERIODIC
        return Inf  # No boundary in periodic lattice
    end
    
    row, col = index_to_coord(lattice, node_id)
    
    # Distance to each boundary
    dist_north = row - 1
    dist_south = lattice.height - row  
    dist_west = col - 1
    dist_east = lattice.width - col
    
    return Float64(min(dist_north, dist_south, dist_west, dist_east))
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
Create a square lattice with specified boundary conditions.

# Arguments
- `width::Int`: Width of lattice
- `height::Int`: Height of lattice  
- `boundary::Symbol`: :absorbing or :periodic

# Returns
- `SquareLattice`: Configured square lattice

# Example
```julia
julia> lattice = create_square_lattice(100, 100, :periodic)
julia> center = get_center_node(lattice)
```
"""
function create_square_lattice(width::Int, height::Int, 
                              boundary::Symbol = :absorbing)::SquareLattice
    boundary_condition = if boundary == :absorbing
        ABSORBING
    elseif boundary == :periodic  
        PERIODIC
    else
        throw(ArgumentError("Unknown boundary type: $boundary. Use :absorbing or :periodic"))
    end
    
    return SquareLattice(width, height, boundary_condition)
end

"""
Create a square torus (periodic boundaries).

# Arguments
- `size::Int`: Size of square torus

# Returns
- `SquareLattice`: Square lattice with periodic boundaries
"""
function create_torus(size::Int)::SquareLattice
    return create_square_lattice(size, size, :periodic)
end