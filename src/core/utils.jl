"""
Utility functions for epidemic model simulations.

This module provides common utilities for parameter validation and 
random number generation.
"""

using Random
using Statistics  # For mean() function

# =============================================================================
# Validation Functions
# =============================================================================

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
        println("$($description) took $(round(elapsed, digits=4))s")
        result
    end
end
