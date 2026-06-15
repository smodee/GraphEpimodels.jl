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

# Threading Setup
For parallel survival probability analysis, start Julia with multiple threads:
  julia --threads=4

# Example Usage
```julia
using GraphEpimodels

# Create ZIM simulation on 100×100 lattice
zim = create_zim_simulation(100, 100, 2.0)  # λ = 2.0

# Run single simulation  
results = run_simulation(zim; max_time=50.0)
println("Final size: ", results[:total_ever_infected])
println("Escaped: ", has_escaped(zim))

# Parallel survival analysis
λ_values = [1.5, 2.0, 2.5]
sweep_results = run_zim_lattice_survival_analysis(λ_values, 100, 100; num_simulations=1000)
```

Based on research by Bethuelsen, Broman & Modée (2024).
"""
module GraphEpimodels

using Random
using Statistics
using ProgressMeter

# Plotting/animation (Makie) and CSV persistence are optional: their code lives
# in package extensions (see ext/) that load only when the user brings in
# `CairoMakie` / `CSV` + `DataFrames`. Loading GraphEpimodels alone stays light.

# Re-export commonly used modules
export Random

# =============================================================================
# Graph Interface and Implementations
# =============================================================================

# Abstract graph interface and core types
include("graphs/graphs.jl")
export AbstractEpidemicGraph, AbstractImplicitGraph, AbstractLatticeGraph
export NodeState, SUSCEPTIBLE, INFECTED, REMOVED, S, I, R
export BoundaryCondition, ABSORBING, PERIODIC
export state_to_int, int_to_state

# Core graph interface functions
export num_nodes, get_neighbors, get_neighbors!, node_states_raw, set_node_states_raw!
export node_states, set_node_states!
export get_node_state, set_node_state!
export get_node_degree, get_boundary_nodes, has_boundary
export count_states, get_nodes_in_state, count_neighbors_by_state
export get_active_edges

# Geometry interface (consumed by visualization)
export has_layout, layout_dim, node_positions, has_cells, cell_polygons

# Square lattice implementation
include("graphs/lattice.jl")
export SquareLattice
export coord_to_index, index_to_coord
export get_center_node, get_random_nodes, distance_to_boundary
export create_square_lattice, create_torus

# Triangular lattice (6-neighbor) and hexagonal/honeycomb lattice (3-neighbor)
include("graphs/triangular_lattice.jl")
export TriangularLattice, create_triangular_lattice
include("graphs/hexagonal_lattice.jl")
export HexagonalLattice, create_hexagonal_lattice

# General graph implementation (adjacency lists)
include("graphs/adjacency.jl")
export AdjacencyGraph, set_coords!
export create_graph_from_matrix, create_graph_from_edges

# Erdős–Rényi random graph (adjacency-list backed)
include("graphs/erdos_renyi.jl")
export ErdosRenyiGraph
export create_erdos_renyi, create_gnp, create_gnm

# Structured graphs as implicit types (store only n; neighbors computed on demand)
include("graphs/complete.jl")
export CompleteGraph, create_complete_graph
include("graphs/path.jl")
export PathGraph, create_path_graph
include("graphs/cycle.jl")
export CycleGraph, create_cycle_graph
include("graphs/star.jl")
export StarGraph, create_star_graph

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

# SIR Model
include("models/sir.jl")
export SIRProcess
export get_sir_statistics
export create_sir_simulation

# Maki-Thompson Rumor Spreading Model
include("models/maki_thompson.jl")
export MakiThompsonProcess
export get_maki_thompson_statistics
export create_maki_thompson_simulation

# Chase-Escape Model
include("models/chasescape.jl")
export ChaseEscapeProcess
export get_chase_escape_statistics
export create_chase_escape_simulation

# =============================================================================
# Utilities
# =============================================================================

# Basic utility functions
include("core/utils.jl")
export validate_node_list, create_rng, set_global_seed!
export @time_it

# Process serialization and CSV utilities
include("core/persistence.jl")
export extract_process_info, process_info_to_config_string
export process_info_to_json, parse_process_info_json
export get_next_start_seed, update_or_append_survival_result

# =============================================================================
# Analysis Tools  
# =============================================================================

# Survival probability analysis
include("analysis/survival_analysis.jl")
export AnalysisMode, MINIMAL, DETAILED
export SurvivalCriterion, EscapeCriterion, PersistenceCriterion, ThresholdCriterion
export estimate_survival_probability, run_parameter_sweep, run_zim_lattice_survival_analysis
export check_threading_setup, get_recommended_threads

# =============================================================================
# Visualization
# =============================================================================

# Abstract visualization interface + visualizer dispatch
include("visualization/visualization.jl")
export AbstractVisualizer, StaticVisualizer, InteractiveVisualizer
export visualize_state, supported_graph_types, can_visualize
export visualizer_for, create_auto_visualizer, render_frame
export COLOR_SCHEMES, available_color_schemes, print_color_schemes, default_color_scheme
export generate_visualization_title
export FIGURE_SIZES, get_figure_size

# Lattice visualization (square / triangular / hexagonal)
include("visualization/lattice_viz.jl")
export LatticeVisualizer, save_plot

# Network visualization (general adjacency graphs)
include("visualization/network_viz.jl")
export NetworkVisualizer

# Animated visualization
include("visualization/animation.jl")
export FrameSampler, EveryStep, EveryNSteps, TimeInterval
export SimulationRecording, record_simulation, num_frames
export animate_recording, animate_simulation

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
    println("Threading: $(Threads.nthreads()) threads available")
    Random.seed!(12345)
end

end  # module GraphEpimodels