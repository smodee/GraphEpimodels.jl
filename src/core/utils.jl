"""
Utility functions for epidemic model simulations.

This module provides common utilities for coordinate conversion,
boundary conditions, parameter validation, and random number generation.
"""

using Random

# =============================================================================
# Coordinate Conversion Utilities (1-indexed for Julia)
# =============================================================================

"""
Convert 2D lattice coordinates to linear index.

Uses row-major ordering: index = (row-1) * width + col

# Arguments
- `row::Int`: Row coordinate (1-indexed)
- `col::Int`: Column coordinate (1-indexed)
- `width::Int`: Width of the lattice

# Returns
- `Int`: Linear index (1-indexed)

# Example
```julia
julia> coord_to_index(2, 3, 5)  # Row 2, Col 3 in 5-wide grid
8
```
"""
function coord_to_index(row::Int, col::Int, width::Int)::Int
    return (row - 1) * width + col
end

"""
Convert linear index to 2D lattice coordinates.

# Arguments
- `index::Int`: Linear index (1-indexed)
- `width::Int`: Width of the lattice

# Returns
- `Tuple{Int, Int}`: (row, col) coordinates (1-indexed)

# Example
```julia
julia> index_to_coord(8, 5)  # Index 8 in 5-wide grid
(2, 3)
```
"""
function index_to_coord(index::Int, width::Int)::Tuple{Int, Int}
    row, col = divrem(index - 1, width)
    return (row + 1, col + 1)
end

# =============================================================================
# Boundary Condition Utilities
# =============================================================================

"""
Apply periodic boundary conditions to a coordinate.

# Arguments
- `coord::Int`: Coordinate value (can be < 1 or > size)
- `size::Int`: Size of the dimension

# Returns
- `Int`: Wrapped coordinate in range [1, size]

# Example
```julia
julia> apply_periodic_boundary(0, 10)   # Wraps to 10
10
julia> apply_periodic_boundary(12, 10)  # Wraps to 2
2
```
"""
function apply_periodic_boundary(coord::Int, size::Int)::Int
    return mod1(coord, size)  # Julia's mod1 gives range [1, size]
end

"""
Check if a position is on the absorbing boundary.

# Arguments
- `row::Int`: Row coordinate
- `col::Int`: Column coordinate
- `height::Int`: Height of the lattice
- `width::Int`: Width of the lattice

# Returns
- `Bool`: true if position is on the boundary
"""
function is_absorbing_boundary(row::Int, col::Int, height::Int, width::Int)::Bool
    return row == 1 || row == height || col == 1 || col == width
end

"""
Get indices of boundary nodes for absorbing boundaries.

# Arguments
- `height::Int`: Height of the lattice
- `width::Int`: Width of the lattice

# Returns
- `Vector{Int}`: Array of boundary node indices
"""
function get_boundary_indices(height::Int, width::Int)::Vector{Int}
    boundary_nodes = Int[]
    
    # Top and bottom rows
    for col in 1:width
        push!(boundary_nodes, coord_to_index(1, col, width))        # Top row
        push!(boundary_nodes, coord_to_index(height, col, width))   # Bottom row
    end
    
    # Left and right columns (excluding corners already added)
    for row in 2:(height-1)
        push!(boundary_nodes, coord_to_index(row, 1, width))        # Left column  
        push!(boundary_nodes, coord_to_index(row, width, width))    # Right column
    end
    
    return boundary_nodes
end

# =============================================================================
# Parameter Validation
# =============================================================================

"""
Validate epidemic model parameters.

# Arguments
- `lambda_param::Float64`: Infection rate (λ)
- `mu_param::Float64`: Recovery/removal rate (μ)
- `min_value::Float64`: Minimum allowed value (default: 0.0)

# Throws
- `ArgumentError`: If parameters are invalid
"""
function validate_epidemic_parameters(lambda_param::Float64, mu_param::Float64 = 1.0; 
                                     min_value::Float64 = 0.0)
    if lambda_param ≤ min_value
        throw(ArgumentError("λ must be > $min_value, got $lambda_param"))
    end
    if mu_param ≤ min_value
        throw(ArgumentError("μ must be > $min_value, got $mu_param"))
    end
end

"""
Validate lattice dimensions.

# Arguments
- `height::Int`: Lattice height
- `width::Int`: Lattice width
- `min_size::Int`: Minimum dimension size (default: 1)
- `max_size::Int`: Maximum dimension size (default: 10000)

# Throws
- `ArgumentError`: If dimensions are invalid
"""
function validate_lattice_size(height::Int, width::Int; 
                              min_size::Int = 1, max_size::Int = 10000)
    if height < min_size || width < min_size
        throw(ArgumentError("Lattice dimensions must be >= $min_size"))
    end
    if height > max_size || width > max_size
        throw(ArgumentError("Lattice dimensions must be <= $max_size"))
    end
end

"""
Validate and convert node list.

# Arguments
- `nodes::Vector{Int}`: List of node indices
- `num_nodes::Int`: Total number of nodes in graph

# Returns
- `Vector{Int}`: Validated node array

# Throws
- `ArgumentError`: If node indices are invalid
"""
function validate_node_list(nodes::Vector{Int}, num_nodes::Int)::Vector{Int}
    if isempty(nodes)
        return nodes
    end
    
    if any(node -> node < 1 || node > num_nodes, nodes)
        throw(ArgumentError("Node indices must be in range [1, $num_nodes]"))
    end
    
    if length(unique(nodes)) != length(nodes)
        throw(ArgumentError("Duplicate node indices not allowed"))
    end
    
    return nodes
end

# =============================================================================
# Random Number Generation  
# =============================================================================

"""
Create a reproducible random number generator.

# Arguments
- `seed::Union{Int, Nothing}`: Random seed (nothing for non-reproducible)

# Returns
- `AbstractRNG`: Random number generator
"""
function create_rng(seed::Union{Int, Nothing} = nothing)::AbstractRNG
    if seed === nothing
        return Random.default_rng()
    else
        return Random.Xoshiro(seed)  # Julia's default high-performance RNG
    end
end

"""
Set global random seed for reproducibility.

# Arguments
- `seed::Int`: Random seed
"""
function set_global_seed!(seed::Int)
    Random.seed!(seed)
end

# =============================================================================
# Statistical Analysis Helpers
# =============================================================================

"""
Compute survival probability with standard error.

# Arguments
- `outcomes::Vector{Bool}`: Boolean array of survival outcomes

# Returns
- `Tuple{Float64, Float64}`: (probability, standard_error)
"""
function compute_survival_probability(outcomes::Vector{Bool})::Tuple{Float64, Float64}
    n = length(outcomes)
    if n == 0
        return (0.0, 0.0)
    end
    
    p = mean(outcomes)
    se = sqrt(p * (1 - p) / n)
    
    return (p, se)
end

"""
Estimate critical parameter value by linear interpolation.

# Arguments
- `lambda_values::Vector{Float64}`: Array of parameter values
- `survival_probs::Vector{Float64}`: Array of corresponding survival probabilities  
- `threshold::Float64`: Threshold probability for criticality (default: 0.5)

# Returns
- `Union{Float64, Nothing}`: Estimated critical parameter, or nothing if not found
"""
function estimate_critical_parameter(lambda_values::Vector{Float64},
                                    survival_probs::Vector{Float64};
                                    threshold::Float64 = 0.5)::Union{Float64, Nothing}
    if length(lambda_values) != length(survival_probs)
        throw(ArgumentError("Arrays must have same length"))
    end
    
    # Sort by lambda values
    sorted_indices = sortperm(lambda_values)
    lambdas = lambda_values[sorted_indices]
    probs = survival_probs[sorted_indices]
    
    # Find crossing point
    below_threshold = probs .< threshold
    above_threshold = probs .>= threshold
    
    if !any(below_threshold) || !any(above_threshold)
        return nothing  # No crossing point found
    end
    
    # Find the transition
    for i in 1:(length(probs)-1)
        if below_threshold[i] && above_threshold[i+1]
            # Linear interpolation
            x1, y1 = lambdas[i], probs[i]
            x2, y2 = lambdas[i+1], probs[i+1]
            
            if y2 ≈ y1  # Avoid division by zero
                return (x1 + x2) / 2
            end
            
            # Interpolate to find where probability equals threshold
            critical_lambda = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            return critical_lambda
        end
    end
    
    return nothing
end

# =============================================================================
# Performance Monitoring
# =============================================================================

"""
Simple timing macro for performance monitoring.

# Example
```julia
julia> @time_it "simulation" begin
           # simulation code
       end
```
"""
macro time_it(description, expr)
    quote
        local start_time = time()
        local result = $(esc(expr))
        local elapsed = time() - start_time
        println("$($description) took $(elapsed:.4f)s")
        result
    end
end