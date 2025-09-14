"""
GraphEpimodels.jl - Fast, extensible epidemic modeling on graphs.

This package provides high-performance implementations of epidemic processes
on graphs, with specialized support for the Zombie Infection Model (ZIM)
and other interacting particle systems.

# Main Components
- Core framework for epidemic processes and graphs
- Optimized square lattice implementations
- Gillespie algorithm for exact stochastic simulation  
- ZIM (Zombie Infection Model) implementation
- Statistical analysis tools

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

using Random
using Statistics
using Plots
using Colors

# Re-export commonly used modules
export Random

# =============================================================================
# Graph Interface and Implementations
# =============================================================================

# Abstract graph interface and core types
include("graphs/graphs.jl")
export AbstractEpidemicGraph
export NodeState, SUSCEPTIBLE, INFECTED, REMOVED, S, I, R
export BoundaryCondition, ABSORBING, PERIODIC
export state_to_int, int_to_state

# Core graph interface functions
export num_nodes, get_neighbors, node_states_raw, set_node_states_raw!
export node_states, set_node_states!
export get_node_state, set_node_state!
export get_node_degree, get_boundary_nodes, has_boundary
export count_states, get_nodes_in_state, count_neighbors_by_state
export get_active_edges

# Square lattice implementation
include("graphs/lattice.jl")
export SquareLattice
export coord_to_index, index_to_coord
export get_center_node, get_random_nodes, distance_to_boundary
export create_square_lattice, create_torus

# General graph implementation (adjacency lists)
include("graphs/adjacency.jl")
export AdjacencyGraph
export create_graph_from_matrix, create_graph_from_edges
export create_complete_graph, create_path_graph, create_cycle_graph, create_star_graph

# =============================================================================
# Epidemic Process Framework
# =============================================================================

# Abstract process interface
include("models/epiprocess.jl")
export AbstractEpidemicProcess
export SIRLikeProcess, ContactLikeProcess, VoterLikeProcess

# Core process interface functions
export get_graph, step!, reset!
export current_time, step_count, is_active, get_total_rate
export get_statistics, has_reached_boundary, get_cluster_size
export run_simulation

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
# Utilities
# =============================================================================

include("core/utils.jl")
export validate_node_list, create_rng, set_global_seed!
export compute_survival_probability
export @time_it

# =============================================================================
# Visualization
# =============================================================================

# Abstract visualization interface
include("visualization/visualization.jl")
export AbstractVisualizer, StaticVisualizer, InteractiveVisualizer, TimeSeriesVisualizer
export visualize_state, supported_graph_types, can_visualize
export COLOR_SCHEMES, get_state_color, available_color_schemes, print_color_schemes
export extract_visualization_data, generate_visualization_title
export FIGURE_SIZES, get_figure_size

# Lattice visualization
include("visualization/lattice_viz.jl")
export LatticeVisualizer
export plot_state, plot_comparison, plot_spread_pattern, set_color_scheme!
export quick_lattice_plot, save_lattice_plot

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
    println("     Time: $(results[:time])")
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
    println("     Survival probability: $(stats[:survival_probability]) ± $(stats[:survival_std_error])")
    println("     Escapes: $(stats[:num_escapes])")
    println("     Extinctions: $(stats[:num_extinctions])")
    if !isnan(stats[:mean_escape_time])
        println("     Mean escape time: $(stats[:mean_escape_time])")
    end
    
    println("\n4. Testing different λ values...")
    λ_test = [1.5, 2.0, 2.5]
    for λ in λ_test
        zim_test = create_zim_simulation(30, 30, λ; rng_seed=42)
        test_stats = estimate_survival_probability(zim_test, [get_center_node(zim_test.lattice)]; 
                                                  num_simulations=50)
        println("     λ=$(λ): P(survival)=$(test_stats[:survival_probability])")
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