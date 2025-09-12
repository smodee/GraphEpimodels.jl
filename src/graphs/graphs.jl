"""
High-performance square lattice implementation for epidemic modeling.

Optimized for large-scale simulations with efficient neighbor lookups,
coordinate conversions, and boundary handling. Uses primitive Int8 states
for maximum performance.
"""

using Random

# Import the graph interface
# include("graphs.jl")  # Assumes graphs.jl is loaded first

# =============================================================================
# Boundary Condition Types
# =============================================================================

@enum BoundaryCondition ABSORBING PERIODIC

# =============================================================================
# High-Performance Square Lattice
# =============================================================================

"""
Optimized square lattice implementation for epidemic simulations.

Uses efficient coordinate arithmetic and pre-computed boundary node lists.
All state operations use primitive Int8 arrays for maximum performance.

# Fields
- `width::Int`: Number of columns
- `height::Int`: Number of rows  
- `boundary::BoundaryCondition`: Boundary condition type
- `states::Vector{Int8}`: Node states (primitive array)
- `boundary_nodes::Vector{Int}`: Pre-computed boundary node indices (for absorbing)
"""
mutable struct SquareLattice <: AbstractEpidemicGraph
    width::Int
    height::Int
    n_nodes::Int
    boundary::BoundaryCondition
    states::Vector{Int8}
    boundary_nodes::Vector{Int}  # Pre-computed for efficiency
    
    function SquareLattice(width::Int, height::Int, boundary::BoundaryCondition = ABSORBING)
        if width < 1 || height < 1
            throw(ArgumentError("Lattice dimensions must be positive"))
        end
        if width > 50_000 || height > 50_000
            throw(ArgumentError("Lattice dimensions too large (>50,000)"))
        end
        
        n_nodes = width * height
        states = zeros(Int8, n_nodes)  # All start SUSCEPTIBLE
        
        # Pre-compute boundary nodes for absorbing boundaries
        boundary_nodes = if boundary == ABSORBING
            _compute_boundary_nodes(width, height)
        else
            Int[]  # No boundary concept for periodic
        end
        
        new(width, height, n_nodes, boundary, states, boundary_nodes)
    end
end

# =============================================================================
# Core Interface Implementation (Required Methods)
# =============================================================================

@inline function num_nodes(lattice::SquareLattice)::Int
    return lattice.n_nodes
end

function node_states_raw(lattice::SquareLattice)::Vector{Int8}
    return lattice.states
end

function set_node_states_raw!(lattice::SquareLattice, states::Vector{Int8})
    if length(states) != num_nodes(lattice)
        throw(ArgumentError("Expected $(num_nodes(lattice)) states, got $(length(states))"))
    end
    lattice.states = states
end

function get_neighbors(lattice::SquareLattice, node_id::Int)::Vector{Int}
    if node_id < 1 || node_id > num_nodes(lattice)
        throw(BoundsError("Node ID $node_id out of range [1, $(num_nodes(lattice))]"))
    end
    
    row, col = _index_to_coord(node_id, lattice.height)
    neighbors = Int[]
    
    if lattice.boundary == ABSORBING
        # Absorbing: only add neighbors within bounds
        _add_neighbor_if_valid!(neighbors, row-1, col, lattice)    # North
        _add_neighbor_if_valid!(neighbors, row+1, col, lattice)    # South  
        _add_neighbor_if_valid!(neighbors, row, col-1, lattice)    # West
        _add_neighbor_if_valid!(neighbors, row, col+1, lattice)    # East
    else  # PERIODIC
        # Periodic: wrap around boundaries
        north_row = row == 1 ? lattice.height : row - 1
        south_row = row == lattice.height ? 1 : row + 1
        west_col = col == 1 ? lattice.width : col - 1
        east_col = col == lattice.width ? 1 : col + 1
        
        push!(neighbors, _coord_to_index(north_row, col, lattice.height))
        push!(neighbors, _coord_to_index(south_row, col, lattice.height))
        push!(neighbors, _coord_to_index(row, west_col, lattice.height))
        push!(neighbors, _coord_to_index(row, east_col, lattice.height))
    end
    
    return neighbors
end

function get_boundary_nodes(lattice::SquareLattice)::Vector{Int}
    return copy(lattice.boundary_nodes)  # Return copy to prevent modification
end

# =============================================================================
# Coordinate Conversion (High Performance)
# =============================================================================

"""
Convert 2D lattice coordinates to linear index.
Uses column-major ordering for cache efficiency: index = col + (row-1)*height

# Arguments  
- `row::Int`: Row coordinate (1-indexed)
- `col::Int`: Column coordinate (1-indexed)
- `height::Int`: Height of lattice

# Returns
- `Int`: Linear index (1-indexed)
"""
@inline function _coord_to_index(row::Int, col::Int, height::Int)::Int
    return col + (row - 1) * height
end

"""
Convert linear index to 2D lattice coordinates.

# Arguments
- `index::Int`: Linear index (1-indexed) 
- `height::Int`: Height of lattice

# Returns
- `Tuple{Int, Int}`: (row, col) coordinates (1-indexed)
"""
@inline function _index_to_coord(index::Int, height::Int)::Tuple{Int, Int}
    row, col = divrem(index - 1, height)
    return (row + 1, col + 1)
end

# Public coordinate conversion functions
function coord_to_index(lattice::SquareLattice, row::Int, col::Int)::Int
    if row < 1 || row > lattice.height || col < 1 || col > lattice.width
        throw(BoundsError("Coordinates ($row, $col) out of bounds"))
    end
    return _coord_to_index(row, col, lattice.height)
end

function index_to_coord(lattice::SquareLattice, index::Int)::Tuple{Int, Int}
    if index < 1 || index > num_nodes(lattice)
        throw(BoundsError("Index $index out of bounds"))
    end
    return _index_to_coord(index, lattice.height)
end

# =============================================================================
# Boundary Computation (Internal)
# =============================================================================

"""
Pre-compute boundary node indices for absorbing lattices.
"""
function _compute_boundary_nodes(width::Int, height::Int)::Vector{Int}
    boundary_nodes = Int[]
    sizehint!(boundary_nodes, 2 * (width + height - 2))  # Pre-allocate
    
    # Top and bottom rows
    for col in 1:width
        push!(boundary_nodes, _coord_to_index(1, col, height))        # Top
        if height > 1  # Avoid duplicates for 1D lattice
            push!(boundary_nodes, _coord_to_index(height, col, height)) # Bottom
        end
    end
    
    # Left and right columns (excluding corners already added)
    if width > 1  # Only if lattice is 2D
        for row in 2:(height-1)
            push!(boundary_nodes, _coord_to_index(row, 1, height))     # Left
            push!(boundary_nodes, _coord_to_index(row, width, height)) # Right
        end
    end
    
    return boundary_nodes
end

"""
Add neighbor to list if coordinates are within lattice bounds (for absorbing).
"""
function _add_neighbor_if_valid!(neighbors::Vector{Int}, row::Int, col::Int, 
                                lattice::SquareLattice)
    if 1 <= row <= lattice.height && 1 <= col <= lattice.width
        push!(neighbors, _coord_to_index(row, col, lattice.height))
    end
end

# =============================================================================
# Lattice-Specific Utility Functions
# =============================================================================

"""
Get the center node of the lattice (useful for initialization).
"""
function get_center_node(lattice::SquareLattice)::Int
    center_row = (lattice.height + 1) ÷ 2
    center_col = (lattice.width + 1) ÷ 2
    return _coord_to_index(center_row, center_col, lattice.height)
end

"""
Get random nodes from the lattice.

# Arguments
- `lattice::SquareLattice`: The lattice
- `count::Int`: Number of nodes to sample
- `rng::AbstractRNG`: Random number generator

# Returns
- `Vector{Int}`: Vector of random node indices
"""
function get_random_nodes(lattice::SquareLattice, count::Int, 
                         rng::AbstractRNG = Random.default_rng())::Vector{Int}
    if count > num_nodes(lattice)
        throw(ArgumentError("Cannot sample $count nodes from $(num_nodes(lattice)) total"))
    end
    return rand(rng, 1:num_nodes(lattice), count)
end

"""
Get distance from node to nearest boundary (for absorbing lattices).
Returns Inf for periodic lattices.

# Arguments
- `lattice::SquareLattice`: The lattice
- `node_id::Int`: Node index

# Returns  
- `Float64`: Distance to nearest boundary
"""
function distance_to_boundary(lattice::SquareLattice, node_id::Int)::Float64
    if lattice.boundary == PERIODIC
        return Inf  # No boundary in periodic lattice
    end
    
    row, col = index_to_coord(lattice, node_id)
    
    # Distance to each boundary edge
    dist_north = row - 1
    dist_south = lattice.height - row
    dist_west = col - 1  
    dist_east = lattice.width - col
    
    return Float64(min(dist_north, dist_south, dist_west, dist_east))
end

"""
Check if a node is on the boundary (for absorbing lattices).
"""
function is_boundary_node(lattice::SquareLattice, node_id::Int)::Bool
    if lattice.boundary == PERIODIC
        return false  # No boundary concept
    end
    return node_id in lattice.boundary_nodes
end

# =============================================================================
# Performance-Optimized Neighbor Counting
# =============================================================================

"""
Optimized neighbor counting for epidemic processes.
This is a performance-critical function used heavily by ZIM.

# Arguments
- `lattice::SquareLattice`: The lattice
- `node_id::Int`: Node to query
- `target_state::NodeState`: State to count

# Returns
- `Int`: Number of neighbors in target state
"""
function count_neighbors_by_state(lattice::SquareLattice, node_id::Int, 
                                 target_state::NodeState)::Int
    # Use optimized inline coordinate arithmetic instead of get_neighbors()
    row, col = _index_to_coord(node_id, lattice.height)
    states = lattice.states
    target_int = state_to_int(target_state)
    count = 0
    
    if lattice.boundary == ABSORBING
        # Check each direction with bounds checking
        if row > 1  # North
            north_idx = _coord_to_index(row-1, col, lattice.height)
            if states[north_idx] == target_int
                count += 1
            end
        end
        if row < lattice.height  # South
            south_idx = _coord_to_index(row+1, col, lattice.height)
            if states[south_idx] == target_int
                count += 1
            end
        end
        if col > 1  # West
            west_idx = _coord_to_index(row, col-1, lattice.height)
            if states[west_idx] == target_int
                count += 1
            end
        end
        if col < lattice.width  # East
            east_idx = _coord_to_index(row, col+1, lattice.height)
            if states[east_idx] == target_int
                count += 1
            end
        end
    else  # PERIODIC
        # Periodic boundaries with wraparound
        north_row = row == 1 ? lattice.height : row - 1
        south_row = row == lattice.height ? 1 : row + 1
        west_col = col == 1 ? lattice.width : col - 1
        east_col = col == lattice.width ? 1 : col + 1
        
        indices = [
            _coord_to_index(north_row, col, lattice.height),
            _coord_to_index(south_row, col, lattice.height),
            _coord_to_index(row, west_col, lattice.height),
            _coord_to_index(row, east_col, lattice.height)
        ]
        
        for idx in indices
            if states[idx] == target_int
                count += 1
            end
        end
    end
    
    return count
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
Create square lattice with specified boundary conditions.

# Arguments
- `width::Int`: Width of lattice (columns)
- `height::Int`: Height of lattice (rows)  
- `boundary::Symbol`: :absorbing or :periodic

# Returns
- `SquareLattice`: Configured lattice

# Example
```julia
julia> lattice = create_square_lattice(100, 100, :absorbing)
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
Create square torus (periodic boundaries).

# Arguments
- `size::Int`: Size of square torus

# Returns
- `SquareLattice`: Square lattice with periodic boundaries
"""
function create_torus(size::Int)::SquareLattice
    return create_square_lattice(size, size, :periodic)
end