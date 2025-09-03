"""
GraphEpimodels.jl - Fast, extensible epidemic modeling on graphs.

This package provides high-performance implementations of epidemic processes
on graphs, with specialized support for the Zombie Infection Model (ZIM),
SIR models, and other interacting particle systems.

# Main Components
- Core framework for epidemic processes and graphs
- Optimized square lattice implementations
- Gillespie algorithm for exact stochastic simulation  
- ZIM (Zombie Infection Model) implementation
- Statistical analysis and parameter estimation tools

# Example Usage
```julia
using GraphEpimodels

# Create ZIM simulation on 100×100 lattice
zim = create_zim_simulation(100, 100, 2.0)  # λ = 2.0

# Run single simulation  
results = run_simulation(zim; max_time=50.0)
println("Final size: ", results[:total_ever_infected])
println("Escaped: ", has_escaped(zim))

# Estimate survival probability
initial = [get_center_node(zim.lattice)]
stats = estimate_survival_probability(zim, initial; num_simulations=1000)
println("Survival probability: ", stats[:survival_probability])
```

Based on research by Bethuelsen, Broman & Modée (2024).
"""
module GraphEpimodels

# Re-export Random for convenience
using Random
export Random

# =============================================================================
# Analysis Tools  
# =============================================================================

# Visualization
include("analysis/visualization.jl")
export LatticeVisualizer, visualize_state
export plot_state, plot_comparison, plot_spread_pattern, set_color_scheme!
export plot_survival_curve, plot_phase_diagram
export quick_plot, save_plot, save_visualization_demo, setup_publication_plots

# =============================================================================
# Core Framework
# =============================================================================

# Abstract types and enums
include("core/base.jl")
export EpidemicGraph, EpidemicProcess
export SIRLikeProcess, ContactLikeProcess, VoterLikeProcess
export NodeState, BoundaryCondition
export SUSCEPTIBLE, INFECTED, REMOVED, S, I, R
export ABSORBING, PERIODIC

# Core interface functions
export num_nodes, node_states, set_node_states!
export get_node_state, set_node_state!
export get_neighbors, get_node_degree, get_boundary_nodes
export count_states, get_nodes_in_state, count_neighbors_by_state
export get_active_edges
export step!, is_active, get_total_rate
export current_time, step_count, get_graph
export get_statistics, reset!, has_reached_boundary, get_cluster_size
export run_simulation

# Event scheduling (Gillespie algorithm)
include("core/events.jl")
export GillespieScheduler, PerformanceScheduler
export gillespie_step

# Utility functions
include("core/utils.jl")
export coord_to_index, index_to_coord
export apply_periodic_boundary, is_absorbing_boundary, get_boundary_indices
export validate_epidemic_parameters, validate_lattice_size, validate_node_list
export create_rng, set_global_seed!
export compute_survival_probability, estimate_critical_parameter
export @time_it

# =============================================================================
# Graph Implementations
# =============================================================================

# Square lattice
include("graphs/lattice.jl")
export SquareLattice
export get_center_node, get_random_nodes, distance_to_boundary
export create_square_lattice, create_torus

# =============================================================================
# Epidemic Models
# =============================================================================

# Zombie Infection Model
include("models/zim.jl")
export ZIMProcess
export has_escaped, get_zim_statistics
export estimate_survival_probability
export create_zim_simulation, run_survival_analysis

# =============================================================================
# Package Information and Convenience Functions
# =============================================================================

"""Package version"""
const VERSION = v"0.1.0"

"""Package authors"""
const AUTHORS = "Samuel Modée, Stein Andreas Bethuelsen, Erik Broman"

"""
Print package information and quick start guide.
"""
function print_info()
    println("GraphEpimodels.jl v$VERSION")
    println("Authors: $AUTHORS")
    println()
    println("Fast, extensible epidemic modeling on graphs")
    println("Specialized for the Zombie Infection Model (ZIM)")
    println()
    println("Quick start:")
    println("  julia> using GraphEpimodels")
    println("  julia> zim = create_zim_simulation(100, 100, 2.0)")
    println("  julia> results = run_simulation(zim)")
    println("  julia> println(\"Escaped: \", has_escaped(zim))")
    println()
    println("For more examples, see the documentation.")
end

export print_info

"""
Create a simple ZIM simulation for quick testing.

# Arguments
- `size::Int`: Lattice size (creates size×size lattice) (default: 50)
- `λ::Float64`: Infection rate (default: 2.0)

# Returns  
- `ZIMProcess`: Ready-to-run ZIM simulation

# Example
```julia
julia> zim = quick_zim()
julia> @time results = run_simulation(zim)
```
"""
function quick_zim(size::Int = 50, λ::Float64 = 2.0)::ZIMProcess
    return create_zim_simulation(size, size, λ; rng_seed=42)
end

export quick_zim

"""
Run a quick demonstration of ZIM functionality.

Shows basic simulation, survival probability estimation, and timing.
"""
function demo()
    println("GraphEpimodels.jl Demo")
    println("=" ^ 50)
    println()
    
    # Basic simulation
    println("1. Creating ZIM simulation (50×50, λ=2.0)...")
    zim = create_zim_simulation(50, 50, 2.0; rng_seed=42)
    println("   ✓ Created ZIM with $(num_nodes(zim.lattice)) nodes")
    
    # Single run
    println("\n2. Running single simulation...")
    @time_it "Single simulation" begin
        results = run_simulation(zim; max_time=50.0)
    end
    
    println("   Final statistics:")
    println("     Time: $(results[:time]:.2f)")
    println("     Steps: $(results[:step_count])")  
    println("     Final size: $(results[:total_ever_infected])")
    println("     Escaped: $(has_escaped(zim))")
    
    # Survival probability  
    println("\n3. Estimating survival probability (100 runs)...")
    initial = [get_center_node(zim.lattice)]
    
    @time_it "Survival analysis" begin
        stats = estimate_survival_probability(zim, initial; num_simulations=100)
    end
    
    println("   Results:")
    println("     Survival probability: $(stats[:survival_probability]:.3f) ± $(stats[:survival_std_error]:.3f)")
    println("     Escapes: $(stats[:num_escapes])")
    println("     Extinctions: $(stats[:num_extinctions])")
    if !isnan(stats[:mean_escape_time])
        println("     Mean escape time: $(stats[:mean_escape_time]:.2f)")
    end
    
    println("\n4. Testing different λ values...")
    λ_test = [1.5, 2.0, 2.5]
    for λ in λ_test
        zim_test = create_zim_simulation(30, 30, λ; rng_seed=42)
        test_stats = estimate_survival_probability(zim_test, [get_center_node(zim_test.lattice)]; 
                                                  num_simulations=50)
        println("     λ=$(λ): P(survival)=$(test_stats[:survival_probability]:.3f)")
    end
    
    println("\n✓ Demo completed successfully!")
    println("\nNext steps:")
    println("  - Try: quick_zim() for fast testing")
    println("  - Try: run_survival_analysis([1.0, 1.5, 2.0, 2.5, 3.0])")
    println("  - Scale up: create_zim_simulation(200, 200, 2.0)")
end

export demo

# =============================================================================
# Module Initialization
# =============================================================================

function __init__()
    # Set up default random seed for reproducible examples
    # (Users can override this)
    Random.seed!(12345)
end

end  # module GraphEpimodels