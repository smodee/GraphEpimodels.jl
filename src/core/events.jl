"""
Event scheduling and time evolution for epidemic processes.

This module implements the Gillespie algorithm for exact stochastic simulation
of continuous-time Markov processes. Focuses on efficient implementation for
square lattice simulations with general graph support.
"""

using Random

# =============================================================================
# Basic Gillespie Implementation
# =============================================================================

"""
Standard Gillespie algorithm for exact stochastic simulation.

Uses the direct method:
1. Calculate total rate of all possible events
2. Sample time to next event from exponential distribution  
3. Sample which event occurs proportional to individual rates
"""
mutable struct GillespieScheduler
    rng::AbstractRNG
    
    GillespieScheduler(rng::AbstractRNG = Random.default_rng()) = new(rng)
end

"""
Perform one Gillespie step with rate array.

# Arguments
- `scheduler::GillespieScheduler`: The scheduler
- `rates::Vector{Float64}`: Array of event rates

# Returns
- `Tuple{Float64, Int}`: (time_increment, event_index) or (Inf, 0) if no events
"""
function gillespie_step(scheduler::GillespieScheduler, rates::Vector{Float64})::Tuple{Float64, Int}
    n_events = length(rates)
    
    if n_events == 0
        return (Inf, 0)
    end
    
    # Calculate total rate
    total_rate = sum(rates)
    
    if total_rate ≤ 0.0
        return (Inf, 0)
    end
    
    # Sample time to next event
    dt = randexp(scheduler.rng) / total_rate
    
    # Sample which event occurs
    cumsum_rates = cumsum(rates)
    threshold = rand(scheduler.rng) * total_rate
    event_index = searchsortedfirst(cumsum_rates, threshold)
    
    # Handle edge case
    event_index = min(event_index, n_events)
    
    return (dt, event_index)
end

# =============================================================================
# Performance-Optimized Version for Large Simulations
# =============================================================================

"""
High-performance Gillespie scheduler using pre-allocated arrays.

Minimizes memory allocations for maximum speed in large simulations
and Monte Carlo studies. Uses working buffers to avoid repeated allocations.
"""
mutable struct PerformanceScheduler
    rng::AbstractRNG
    max_events::Int
    
    # Pre-allocated working arrays
    cumsum_buffer::Vector{Float64}
    
    function PerformanceScheduler(max_events::Int = 100_000, rng::AbstractRNG = Random.default_rng())
        new(rng, max_events, zeros(Float64, max_events))
    end
end

"""
High-performance Gillespie step using pre-allocated buffers.

# Arguments
- `scheduler::PerformanceScheduler`: The scheduler
- `rates::Vector{Float64}`: Event rates (only first num_active used)
- `num_active::Int`: Number of active events

# Returns
- `Tuple{Float64, Int}`: (time_increment, event_index) or (Inf, 0) if no events
"""
function gillespie_step(scheduler::PerformanceScheduler, rates::Vector{Float64}, 
                       num_active::Int)::Tuple{Float64, Int}
    if num_active ≤ 0
        return (Inf, 0)
    end
    
    if num_active > scheduler.max_events
        error("Too many events ($num_active) for scheduler capacity ($(scheduler.max_events))")
    end
    
    # Calculate total rate
    total_rate = sum(view(rates, 1:num_active))
    
    if total_rate ≤ 0.0
        return (Inf, 0)
    end
    
    # Sample time increment
    dt = randexp(scheduler.rng) / total_rate
    
    # Compute cumulative sums in pre-allocated buffer
    cumsum!(view(scheduler.cumsum_buffer, 1:num_active), view(rates, 1:num_active))
    
    # Sample event
    threshold = rand(scheduler.rng) * total_rate
    event_index = searchsortedfirst(view(scheduler.cumsum_buffer, 1:num_active), threshold)
    event_index = min(event_index, num_active)
    
    return (dt, event_index)
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
Simple Gillespie step without creating scheduler object.

Convenience function for one-off calculations.

# Arguments
- `rates::Vector{Float64}`: Array of event rates
- `rng::AbstractRNG`: Random number generator

# Returns  
- `Tuple{Float64, Int}`: (time_increment, event_index)
"""
function gillespie_step(rates::Vector{Float64}, rng::AbstractRNG = Random.default_rng())::Tuple{Float64, Int}
    scheduler = GillespieScheduler(rng)
    return gillespie_step(scheduler, rates)
end