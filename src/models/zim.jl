"""
Zombie Infection Model (ZIM) implementation.

This module implements the Zombie Infection Model described in Bethuelsen,
Broman & Modée (2024). The ZIM is a variant of the SIR model where infected
individuals (zombies) attempt to bite susceptible neighbors, and susceptible
individuals fight back to kill zombies.

Key differences from SIR:
- Infected nodes don't recover spontaneously
- Recovery happens when susceptible neighbors kill the infected node  
- Kill rate: μ × (number of susceptible neighbors)
- Infection rate: λ × (number of susceptible neighbors) for each infected node
- Outcome probability: λ/(λ+μ) for infection, μ/(λ+μ) for kill
"""

using Random
using ..GraphEpimodels: EpidemicProcess, SIRLikeProcess, SquareLattice
using ..GraphEpimodels: NodeState, SUSCEPTIBLE, INFECTED, REMOVED
using ..GraphEpimodels: GillespieScheduler, PerformanceScheduler, gillespie_step
using ..GraphEpimodels: get_nodes_in_state, count_neighbors_by_state, get_neighbors
using ..GraphEpimodels: validate_epidemic_parameters, create_rng, compute_survival_probability
using ..GraphEpimodels: create_square_lattice, get_center_node, get_random_nodes

# =============================================================================
# ZIM Process Implementation
# =============================================================================

"""
Zombie Infection Model process.

The ZIM evolves according to:
1. Each infected node (zombie) interacts with susceptible neighbors at rate (λ + μ) × #susceptible_neighbors
2. Outcome: infection with probability λ/(λ+μ), kill with probability μ/(λ+μ)
3. Process stops when no infected nodes remain

# Fields
- `lattice::SquareLattice`: The underlying graph
- `λ::Float64`: Infection rate (bite rate)
- `μ::Float64`: Kill rate  
- `infection_prob::Float64`: Probability λ/(λ+μ) that zombie wins interaction
- `scheduler::Union{GillespieScheduler, PerformanceScheduler}`: Event scheduler
- `time::Float64`: Current simulation time
- `steps::Int`: Number of steps executed
- `rng::AbstractRNG`: Random number generator
"""
mutable struct ZIMProcess <: SIRLikeProcess
    lattice::SquareLattice
    λ::Float64
    μ::Float64
    infection_prob::Float64
    scheduler::Union{GillespieScheduler, PerformanceScheduler}
    time::Float64
    steps::Int
    rng::AbstractRNG
    
    function ZIMProcess(lattice::SquareLattice, λ::Float64, μ::Float64 = 1.0;
                       rng::AbstractRNG = Random.default_rng())
        validate_epidemic_parameters(λ, μ)
        
        infection_prob = λ / (λ + μ)
        
        # Choose scheduler based on lattice size
        scheduler = if num_nodes(lattice) > 100_000
            PerformanceScheduler(num_nodes(lattice), rng)
        else
            GillespieScheduler(rng)
        end
        
        new(lattice, λ, μ, infection_prob, scheduler, 0.0, 0, rng)
    end
end

# =============================================================================
# Required EpidemicProcess Interface
# =============================================================================

function get_graph(process::ZIMProcess)::SquareLattice
    return process.lattice
end

function current_time(process::ZIMProcess)::Float64
    return process.time
end

function step_count(process::ZIMProcess)::Int
    return process.steps
end

function is_active(process::ZIMProcess)::Bool
    infected_nodes = get_nodes_in_state(process.lattice, INFECTED)
    return !isempty(infected_nodes)
end

function get_total_rate(process::ZIMProcess)::Float64
    infected_nodes = get_nodes_in_state(process.lattice, INFECTED)
    total_rate = 0.0
    
    for node in infected_nodes
        susceptible_neighbors = count_neighbors_by_state(process.lattice, node, SUSCEPTIBLE)
        total_rate += (process.λ + process.μ) * susceptible_neighbors
    end
    
    return total_rate
end

"""
Execute one ZIM simulation step.

# Arguments
- `process::ZIMProcess`: The ZIM process

# Returns  
- `Float64`: Time increment for this step (Inf if no events possible)
"""
function step!(process::ZIMProcess)::Float64
    # Get all infected nodes and their rates
    infected_nodes = get_nodes_in_state(process.lattice, INFECTED)
    
    if isempty(infected_nodes)
        return Inf  # No more infected nodes
    end
    
    # Calculate rates for each infected node
    rates = Float64[]
    sizehint!(rates, length(infected_nodes))
    
    for node in infected_nodes
        susceptible_neighbors = count_neighbors_by_state(process.lattice, node, SUSCEPTIBLE)
        rate = (process.λ + process.μ) * susceptible_neighbors
        push!(rates, rate)
    end
    
    # Sample next event using Gillespie algorithm
    dt, zombie_idx = if process.scheduler isa PerformanceScheduler
        gillespie_step(process.scheduler, rates, length(rates))
    else
        gillespie_step(process.scheduler, rates)
    end
    
    if dt == Inf
        return dt  # No positive rates
    end
    
    # Update time and step count
    process.time += dt
    process.steps += 1
    
    # Execute the selected event
    zombie_node = infected_nodes[zombie_idx]
    
    # Determine outcome: infection vs kill
    if rand(process.rng) < process.infection_prob
        # Zombie wins - infect a susceptible neighbor
        _zombie_wins!(process, zombie_node)
    else
        # Zombie loses - gets killed
        _zombie_loses!(process, zombie_node)
    end
    
    return dt
end

"""
Reset process to initial conditions.

# Arguments
- `process::ZIMProcess`: The ZIM process
- `initial_infected::Vector{Int}`: Nodes to start as infected
"""
function reset!(process::ZIMProcess, initial_infected::Vector{Int})
    # Reset time and counters
    process.time = 0.0
    process.steps = 0
    
    # Reset all nodes to susceptible
    fill!(node_states(process.lattice), SUSCEPTIBLE)
    
    # Set initial infected nodes
    for node_id in initial_infected
        set_node_state!(process.lattice, node_id, INFECTED)
    end
end

# =============================================================================
# ZIM-Specific Event Handlers (Internal Functions)
# =============================================================================

"""
Handle zombie victory: infect a random susceptible neighbor.

# Arguments
- `process::ZIMProcess`: The ZIM process
- `zombie_node::Int`: Index of the attacking zombie
"""
function _zombie_wins!(process::ZIMProcess, zombie_node::Int)
    # Get susceptible neighbors
    neighbors = get_neighbors(process.lattice, zombie_node)
    susceptible_neighbors = filter(n -> get_node_state(process.lattice, n) == SUSCEPTIBLE, 
                                  neighbors)
    
    if !isempty(susceptible_neighbors)
        # Randomly choose a susceptible neighbor to infect
        target = rand(process.rng, susceptible_neighbors)
        set_node_state!(process.lattice, target, INFECTED)
    end
end

"""
Handle zombie defeat: zombie is killed and removed.

# Arguments
- `process::ZIMProcess`: The ZIM process  
- `zombie_node::Int`: Index of the zombie that was killed
"""
function _zombie_loses!(process::ZIMProcess, zombie_node::Int)
    set_node_state!(process.lattice, zombie_node, REMOVED)
end

# =============================================================================
# ZIM-Specific Analysis Functions
# =============================================================================

"""
Check if infection has reached the boundary (zombie outbreak).

# Arguments
- `process::ZIMProcess`: The ZIM process

# Returns
- `Bool`: true if outbreak has occurred
"""
function has_escaped(process::ZIMProcess)::Bool
    return has_reached_boundary(process)
end

"""
Get current ZIM statistics.

# Arguments
- `process::ZIMProcess`: The ZIM process

# Returns
- `Dict{Symbol, Any}`: Dictionary with detailed statistics
"""
function get_zim_statistics(process::ZIMProcess)::Dict{Symbol, Any}
    base_stats = get_statistics(process)
    
    # Add ZIM-specific statistics
    boundary_infected = count(node -> get_node_state(process.lattice, node) == INFECTED,
                             get_boundary_nodes(process.lattice))
    
    base_stats[:λ] = process.λ
    base_stats[:μ] = process.μ
    base_stats[:infection_probability] = process.infection_prob
    base_stats[:has_escaped] = has_escaped(process)
    base_stats[:boundary_infected] = boundary_infected
    
    return base_stats
end

"""
Estimate survival probability via Monte Carlo simulation.

# Arguments
- `process::ZIMProcess`: The ZIM process
- `initial_infected::Vector{Int}`: Initial infected nodes
- `num_simulations::Int`: Number of simulation runs (default: 1000)
- `max_time::Float64`: Maximum time per simulation (default: 1000.0)

# Returns
- `Dict{Symbol, Any}`: Dictionary with survival statistics
"""
function estimate_survival_probability(process::ZIMProcess, initial_infected::Vector{Int};
                                      num_simulations::Int = 1000,
                                      max_time::Float64 = 1000.0)::Dict{Symbol, Any}
    
    survival_outcomes = Bool[]
    escape_times = Float64[]
    final_sizes = Int[]
    
    sizehint!(survival_outcomes, num_simulations)
    
    for i in 1:num_simulations
        # Reset simulation
        reset!(process, initial_infected)
        
        # Run until escape, extinction, or timeout
        results = run_simulation(process; max_time=max_time)
        
        # Record outcomes
        survived = has_escaped(process)
        push!(survival_outcomes, survived)
        
        if survived
            push!(escape_times, results[:time])
        end
        
        push!(final_sizes, results[:total_ever_infected])
    end
    
    # Compute statistics
    survival_prob, survival_se = compute_survival_probability(survival_outcomes)
    
    return Dict{Symbol, Any}(
        :survival_probability => survival_prob,
        :survival_std_error => survival_se,
        :mean_escape_time => isempty(escape_times) ? NaN : sum(escape_times) / length(escape_times),
        :mean_final_size => sum(final_sizes) / length(final_sizes),
        :num_escapes => length(escape_times),
        :num_extinctions => num_simulations - length(escape_times)
    )
end

# =============================================================================
# Factory Functions and Convenience Interface
# =============================================================================

"""
Create a complete ZIM simulation setup.

# Arguments  
- `width::Int`: Lattice width
- `height::Int`: Lattice height
- `λ::Float64`: Infection rate
- `μ::Float64`: Kill rate (default: 1.0)
- `boundary::Symbol`: :absorbing or :periodic (default: :absorbing)  
- `initial_infected::Union{Symbol, Vector{Int}}`: :center, :random, or node indices
- `rng_seed::Union{Int, Nothing}`: Random seed (default: nothing)

# Returns
- `ZIMProcess`: Configured ZIM process ready to run

# Example
```julia  
julia> zim = create_zim_simulation(100, 100, 2.0)
julia> results = run_simulation(zim; max_time=100.0)
julia> println("Survival: ", has_escaped(zim))
```
"""
function create_zim_simulation(width::Int, height::Int, λ::Float64, μ::Float64 = 1.0;
                              boundary::Symbol = :absorbing,
                              initial_infected::Union{Symbol, Vector{Int}} = :center,
                              rng_seed::Union{Int, Nothing} = nothing)::ZIMProcess
    
    # Create lattice
    lattice = create_square_lattice(width, height, boundary)
    
    # Create RNG
    rng = create_rng(rng_seed)
    
    # Create process
    process = ZIMProcess(lattice, λ, μ; rng=rng)
    
    # Set initial conditions
    infected_nodes = if initial_infected == :center
        [get_center_node(lattice)]
    elseif initial_infected == :random
        get_random_nodes(lattice, 1, rng)
    else
        initial_infected
    end
    
    reset!(process, infected_nodes)
    
    return process
end

"""
Run survival probability analysis across multiple λ values.

# Arguments
- `λ_values::Vector{Float64}`: Array of λ values to test  
- `width::Int`: Lattice width (default: 100)
- `height::Int`: Lattice height (default: 100)
- `num_simulations::Int`: Number of simulations per λ (default: 100)
- `μ::Float64`: Kill rate (default: 1.0)
- `kwargs...`: Additional arguments for create_zim_simulation

# Returns
- `Dict{Symbol, Any}`: Results including survival probabilities and errors

# Example
```julia
julia> λs = 1.0:0.2:3.0
julia> results = run_survival_analysis(λs; num_simulations=1000)
julia> # Plot results...
```
"""
function run_survival_analysis(λ_values::Vector{Float64};
                              width::Int = 100, height::Int = 100,
                              num_simulations::Int = 100, μ::Float64 = 1.0,
                              kwargs...)::Dict{Symbol, Any}
    
    survival_probs = Float64[]
    std_errors = Float64[]
    
    sizehint!(survival_probs, length(λ_values))
    sizehint!(std_errors, length(λ_values))
    
    for (i, λ) in enumerate(λ_values)
        println("Testing λ = $(λ) ($(i)/$(length(λ_values)))")
        
        # Create simulation
        zim = create_zim_simulation(width, height, λ, μ; kwargs...)
        initial_infected = [get_center_node(zim.lattice)]
        
        # Run analysis
        stats = estimate_survival_probability(zim, initial_infected; 
                                            num_simulations=num_simulations)
        
        push!(survival_probs, stats[:survival_probability])
        push!(std_errors, stats[:survival_std_error])
    end
    
    return Dict{Symbol, Any}(
        :λ_values => λ_values,
        :survival_probs => survival_probs,
        :std_errors => std_errors,
        :num_simulations => num_simulations
    )
end