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
using Distributed, ProgressMeter

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
export get_statistics, has_escaped, get_cluster_size
export run_simulation

# =============================================================================
# Epidemic Models
# =============================================================================

# Zombie Infection Model
include("models/zim.jl")
export ZIMProcess
export has_escaped, get_zim_statistics
export create_zim_simulation

# =============================================================================
# Utilities
# =============================================================================

include("core/utils.jl")
export validate_node_list, create_rng, set_global_seed!
export @time_it

# =============================================================================
# Analysis Tools  
# =============================================================================

# Survival probability analysis
include("analysis/survival_analysis.jl")
export AnalysisMode, MINIMAL, DETAILED
export SurvivalCriterion, EscapeCriterion, PersistenceCriterion, ThresholdCriterion
export estimate_survival_probability, run_parameter_sweep, run_zim_survival_analysis

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
const AUTHORS = "Samuel Modée"

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

# =============================================================================
# Module Initialization
# =============================================================================

function __init__()
    # Set up default random seed for reproducible examples
    # (Users can override this)
    Random.seed!(12345)
end

end  # module GraphEpimodels