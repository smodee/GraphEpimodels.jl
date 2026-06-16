"""
Utility functions for epidemic model simulations.

This module provides common utilities for parameter validation and 
random number generation.
"""

using Random
using Statistics  # For mean() function

# =============================================================================
# Random Number Generation
# =============================================================================

"""
Create a reproducible random number generator.

# Arguments
- `seed::Union{Int, Nothing}`: Random seed (nothing for non-reproducible)

# Returns
- `Random.Xoshiro`: Concrete high-performance RNG

The concrete return type matters: an `::AbstractRNG` annotation here would make
the inferred return abstract, which propagates into the process constructors and
reintroduces the per-step rand() boxing the RNG type parameter is meant to avoid.
"""
function create_rng(seed::Union{Int, Nothing} = nothing)
    if seed === nothing
        return Random.Xoshiro()
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
# Initial Node Resolution
# =============================================================================

"""
Resolve an `initial_nodes` spec into a concrete `Vector{Int}`.

Accepts `:center` (graph center or midpoint fallback), `:random` (one random
node), or an explicit `Vector{Int}` that is returned as-is.
"""
function resolve_initial_nodes(graph::AbstractEpidemicGraph,
                               spec::Union{Symbol, Vector{Int}},
                               rng::AbstractRNG)::Vector{Int}
    if spec == :center
        hasmethod(get_center_node, (typeof(graph),)) ?
            [get_center_node(graph)] : [num_nodes(graph) ÷ 2]
    elseif spec == :random
        [rand(rng, 1:num_nodes(graph))]
    else
        spec
    end
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
