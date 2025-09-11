"""
Visualization tools for epidemic simulations.

This module provides plotting and animation capabilities for visualizing
epidemic processes on graphs, with special focus on lattice-based models
like ZIM and SIR.
"""

using Plots
using Colors
using ..GraphEpimodels: EpidemicProcess, SquareLattice, NodeState
using ..GraphEpimodels: SUSCEPTIBLE, INFECTED, REMOVED, ABSORBING
using ..GraphEpimodels: num_nodes, node_states, get_boundary_nodes
using ..GraphEpimodels: index_to_coord, get_statistics

# =============================================================================
# Color Schemes for Different Models
# =============================================================================

const COLOR_SCHEMES = Dict(
    :zim => Dict(
        :susceptible => colorant"lightgray",     # Light gray
        :infected => colorant"forestgreen",     # Green (zombies)
        :removed => colorant"red",              # Red (dead)
        :boundary => colorant"darkgray",        # Dark gray for boundary
        :name => "ZIM (S=gray, I=green, R=red)"
    ),
    :sir => Dict(
        :susceptible => colorant"white",         # White
        :infected => colorant"red",             # Red (infected)
        :removed => colorant"black",            # Black (recovered)
        :boundary => colorant"gray",            # Gray for boundary
        :name => "SIR (S=white, I=red, R=black)"
    ),
    :custom => Dict(
        :susceptible => colorant"white",
        :infected => colorant"blue",            # Blue
        :removed => colorant"orange",           # Orange
        :boundary => colorant"black",           # Black
        :name => "Custom"
    )
)

# =============================================================================
# Lattice Visualizer
# =============================================================================

"""
Visualization tools for epidemic processes on square lattices.

Provides high-quality heatmap plotting with configurable color schemes,
boundary highlighting, and support for publication-ready figures.
"""
mutable struct LatticeVisualizer
    color_scheme::Symbol
    colors::Dict{Symbol, Any}  # Changed from Colorant to Any to allow strings
    figsize::Tuple{Int, Int}
    
    function LatticeVisualizer(color_scheme::Symbol = :zim, figsize::Tuple{Int, Int} = (800, 800))
        if color_scheme ∉ keys(COLOR_SCHEMES)
            throw(ArgumentError("Unknown color scheme '$color_scheme'. Available: $(keys(COLOR_SCHEMES))"))
        end
        
        colors = COLOR_SCHEMES[color_scheme]
        new(color_scheme, colors, figsize)
    end
end

"""
Get 2D array representation of lattice states for visualization.

# Arguments
- `lattice::SquareLattice`: The lattice to visualize

# Returns
- `Matrix{Int}`: 2D array with state values (height × width)
"""
function visualize_state(lattice::SquareLattice)::Matrix{Int}
    states = node_states(lattice)
    state_matrix = Matrix{Int}(undef, lattice.height, lattice.width)
    
    for i in 1:num_nodes(lattice)
        row, col = index_to_coord(lattice, i)
        state_matrix[row, col] = Int(states[i])
    end
    
    return state_matrix
end

"""
Plot the current state of an epidemic process on a lattice.

# Arguments
- `visualizer::LatticeVisualizer`: The visualizer instance
- `process::EpidemicProcess`: Epidemic process to visualize (must have SquareLattice)
- `title::Union{String, Nothing}`: Plot title (auto-generated if nothing)
- `show_boundary::Bool`: Whether to highlight boundary nodes (default: true)

# Returns
- `Plots.Plot`: The plot object

# Example
```julia
julia> vis = LatticeVisualizer(:zim)
julia> p = plot_state(vis, zim_process)
julia> display(p)
```
"""
function plot_state(visualizer::LatticeVisualizer, 
                   process::EpidemicProcess;
                   title::Union{String, Nothing} = nothing,
                   show_boundary::Bool = true)
    
    if !isa(get_graph(process), SquareLattice)
        throw(ArgumentError("plot_state requires a SquareLattice graph"))
    end
    
    lattice = get_graph(process)
    state_grid = visualize_state(lattice)
    
    # Create color mapping
    state_colors = [
        visualizer.colors[:susceptible],  # 0 = SUSCEPTIBLE
        visualizer.colors[:infected],     # 1 = INFECTED  
        visualizer.colors[:removed]       # 2 = REMOVED
    ]
    
    # Create heatmap
    p = heatmap(state_grid, 
               c = state_colors,
               aspect_ratio = :equal,
               size = visualizer.figsize,
               showaxis = false,
               grid = false,
               colorbar = false)
    
    # Set title
    if title === nothing
        stats = get_statistics(process)
        title = string(visualizer.colors[:name], "\n",
                      "Time: $(round(stats[:time], digits=2)), ",
                      "Steps: $(stats[:step_count]), ",
                      "I: $(stats[:infected]), R: $(stats[:removed])")
    end
    
    plot!(title = title)
    
    # Add boundary highlighting if requested
    if show_boundary && lattice.boundary == ABSORBING
        _add_boundary_highlight!(p, lattice, state_grid, visualizer.colors[:boundary])
    end
    
    return p
end

"""
Add boundary highlighting to an existing plot.
"""
function _add_boundary_highlight!(p, lattice::SquareLattice, state_grid::Matrix{Int}, boundary_color)
    height, width = size(state_grid)
    
    # Add boundary rectangle overlay
    plot!(p, [0.5, width+0.5, width+0.5, 0.5, 0.5], 
          [0.5, 0.5, height+0.5, height+0.5, 0.5],
          line = (3, boundary_color, 0.7),
          fill = false,
          label = "")
end

"""
Plot multiple epidemic states side by side for comparison.

# Arguments  
- `visualizer::LatticeVisualizer`: The visualizer instance
- `processes::Vector{EpidemicProcess}`: List of processes to compare
- `titles::Union{Vector{String}, Nothing}`: Titles for each subplot (auto-generated if nothing)

# Returns
- `Plots.Plot`: Combined plot with subplots

# Example
```julia
julia> processes = [zim1, zim2, zim3]
julia> titles = ["λ=1.5", "λ=2.0", "λ=2.5"]  
julia> p = plot_comparison(vis, processes, titles)
```
"""
function plot_comparison(visualizer::LatticeVisualizer,
                        processes::Vector{EpidemicProcess};
                        titles::Union{Vector{String}, Nothing} = nothing)
    
    n_plots = length(processes)
    if n_plots == 0
        throw(ArgumentError("At least one process required"))
    end
    
    # Auto-generate titles if not provided
    if titles === nothing
        titles = ["Process $i" for i in 1:n_plots]
    elseif length(titles) != n_plots
        throw(ArgumentError("Need $n_plots titles, got $(length(titles))"))
    end
    
    # Determine layout
    cols = min(n_plots, 3)  # Max 3 columns
    rows = ceil(Int, n_plots / cols)
    
    # Create individual plots
    plots = []
    for i in 1:n_plots
        p = plot_state(visualizer, processes[i]; title = titles[i])
        push!(plots, p)
    end
    
    # Combine into layout
    combined_plot = plot(plots..., layout = (rows, cols), 
                        size = (cols * 400, rows * 400))
    
    return combined_plot
end

"""
Plot current state with special highlighting for spread analysis.

# Arguments
- `visualizer::LatticeVisualizer`: The visualizer instance  
- `process::EpidemicProcess`: Process to visualize
- `highlight_infected::Bool`: Whether to add borders around infected nodes (default: true)
- `highlight_boundary_infected::Bool`: Whether to highlight boundary infections (default: true)
- `title::Union{String, Nothing}`: Plot title

# Returns
- `Plots.Plot`: Plot with spread pattern highlighting

# Example
```julia
julia> p = plot_spread_pattern(vis, zim, title="ZIM Spread Pattern (λ=2.5)")
```
"""
function plot_spread_pattern(visualizer::LatticeVisualizer,
                            process::EpidemicProcess;
                            highlight_infected::Bool = true,
                            highlight_boundary_infected::Bool = true,
                            title::Union{String, Nothing} = nothing)
    
    if !isa(get_graph(process), SquareLattice)
        throw(ArgumentError("plot_spread_pattern requires a SquareLattice graph"))
    end
    
    # Create base plot
    p = plot_state(visualizer, process; title = title, show_boundary = true)
    
    lattice = get_graph(process)
    state_grid = visualize_state(lattice)
    
    # Highlight currently infected nodes
    if highlight_infected
        infected_positions = findall(x -> x == Int(INFECTED), state_grid)
        for pos in infected_positions
            row, col = pos.I
            # Add rectangle around infected node
            plot!(p, [col-0.4, col+0.4, col+0.4, col-0.4, col-0.4],
                  [row-0.4, row-0.4, row+0.4, row+0.4, row-0.4],
                  line = (2, :black), fill = false, label = "")
        end
    end
    
    # Highlight infected nodes on boundary (escape points)
    if highlight_boundary_infected && lattice.boundary == ABSORBING
        boundary_nodes = get_boundary_nodes(lattice)
        states = node_states(lattice)
        infected_boundary = filter(i -> states[i] == INFECTED, boundary_nodes)
        
        for node_idx in infected_boundary
            row, col = index_to_coord(lattice, node_idx)
            # Add prominent highlighting for boundary escape
            plot!(p, [col-0.45, col+0.45, col+0.45, col-0.45, col-0.45],
                  [row-0.45, row-0.45, row+0.45, row+0.45, row-0.45],
                  line = (4, :yellow), fill = false, label = "")
        end
    end
    
    return p
end

"""
Change the color scheme of the visualizer.

# Arguments
- `visualizer::LatticeVisualizer`: The visualizer to modify
- `scheme::Symbol`: New color scheme (:zim, :sir, :custom)
"""
function set_color_scheme!(visualizer::LatticeVisualizer, scheme::Symbol)
    if scheme ∉ keys(COLOR_SCHEMES)
        throw(ArgumentError("Unknown color scheme '$scheme'. Available: $(keys(COLOR_SCHEMES))"))
    end
    
    visualizer.color_scheme = scheme
    visualizer.colors = COLOR_SCHEMES[scheme]
end

# =============================================================================
# Statistical Plotting Functions
# =============================================================================

"""
Plot survival probability as a function of infection rate.

# Arguments
- `λ_values::Vector{Float64}`: Array of λ values
- `survival_probs::Vector{Float64}`: Array of survival probabilities
- `std_errors::Union{Vector{Float64}, Nothing}`: Standard errors for error bars (optional)
- `title::String`: Plot title (default: "Survival Probability vs λ")
- `xlabel::String`: X-axis label (default: "Infection rate (λ)")
- `ylabel::String`: Y-axis label (default: "P(survival)")

# Returns
- `Plots.Plot`: The survival curve plot

# Example
```julia
julia> results = run_survival_analysis([1.0, 1.5, 2.0, 2.5, 3.0])
julia> p = plot_survival_curve(results[:λ_values], results[:survival_probs], results[:std_errors])
```
"""
function plot_survival_curve(λ_values::Vector{Float64},
                             survival_probs::Vector{Float64};
                             std_errors::Union{Vector{Float64}, Nothing} = nothing,
                             title::String = "Survival Probability vs λ",
                             xlabel::String = "Infection rate (λ)",
                             ylabel::String = "P(survival)")
    
    # Create base plot
    if std_errors !== nothing
        p = plot(λ_values, survival_probs, 
                yerror = std_errors,
                marker = :circle,
                linewidth = 2,
                markersize = 6,
                capsize = 5,
                label = "Survival probability")
    else
        p = plot(λ_values, survival_probs,
                marker = :circle,
                linewidth = 2,
                markersize = 6,
                label = "Survival probability")
    end
    
    # Customize plot
    plot!(p, xlabel = xlabel,
          ylabel = ylabel,
          title = title,
          grid = true,
          gridwidth = 1,
          gridcolor = :gray,
          gridalpha = 0.3,
          ylims = (-0.05, 1.05),
          legend = :bottomright)
    
    # Add horizontal reference line at 0.5
    hline!(p, [0.5], line = (:dash, :gray, 1), alpha = 0.5, label = "")
    
    return p
end

"""
Create a phase diagram showing parameter relationships.

# Arguments
- `results::Dict{Symbol, Any}`: Results from parameter sweep
- `xlabel::String`: X-axis parameter name
- `ylabel::String`: Y-axis parameter name
- `title::String`: Plot title

# Returns
- `Plots.Plot`: Phase diagram plot
"""
function plot_phase_diagram(results::Dict{Symbol, Any};
                           xlabel::String = "Parameter 1",
                           ylabel::String = "Parameter 2", 
                           title::String = "Phase Diagram")
    
    # Extract data - assumes results has the right structure
    # This is a placeholder - would need to be customized based on actual results structure
    x_vals = get(results, :x_values, Float64[])
    y_vals = get(results, :y_values, Float64[])
    z_vals = get(results, :z_values, Float64[])
    
    if isempty(x_vals) || isempty(y_vals) || isempty(z_vals)
        @warn "Phase diagram requires x_values, y_values, and z_values in results"
        return plot()
    end
    
    p = heatmap(x_vals, y_vals, z_vals,
               xlabel = xlabel,
               ylabel = ylabel,
               title = title,
               aspect_ratio = :equal,
               colorbar_title = "Value")
    
    return p
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
Quickly plot an epidemic process state.

# Arguments  
- `process::EpidemicProcess`: Process to plot
- `kwargs...`: Additional arguments passed to plot_state

# Returns
- `Plots.Plot`: The plot

# Example
```julia
julia> p = quick_plot(zim_process)
julia> display(p)
```
"""
function quick_plot(process::EpidemicProcess; kwargs...)
    visualizer = LatticeVisualizer()
    return plot_state(visualizer, process; kwargs...)
end

"""
Save a plot to file with high DPI for publication.

# Arguments
- `p::Plots.Plot`: Plot to save
- `filename::String`: Output filename
- `dpi::Int`: Resolution (default: 300)
"""
function save_plot(p::Plots.Plot, filename::String; dpi::Int = 300)
    savefig(p, filename)
    println("Plot saved to: $filename")
end

"""
Create and save a demonstration plot showing visualization capabilities.

# Arguments
- `save_dir::String`: Directory to save figures (default: "figures/")

# Example
```julia
julia> save_visualization_demo("my_figures/")
```
"""
function save_visualization_demo(save_dir::String = "figures/")
    mkpath(save_dir)
    println("Generating visualization examples...")
    
    # Example 1: Different λ values comparison
    println("Creating λ comparison plot...")
    λ_vals = [0.8, 1.5, 3.0]
    processes = EpidemicProcess[]
    
    for λ in λ_vals
        zim = create_zim_simulation(30, 30, λ; rng_seed=42)
        # Run for a bit to see some spread
        run_simulation(zim; max_time=20.0, max_steps=1000)
        push!(processes, zim)
    end
    
    titles = ["λ = $λ" for λ in λ_vals]
    visualizer = LatticeVisualizer(:zim)
    p1 = plot_comparison(visualizer, processes; titles=titles)
    save_plot(p1, joinpath(save_dir, "lambda_comparison.png"))
    
    # Example 2: Spread pattern analysis  
    println("Creating spread pattern plot...")
    zim_spread = create_zim_simulation(40, 40, 2.5; rng_seed=123)
    run_simulation(zim_spread; max_time=30.0, max_steps=2000)
    
    p2 = plot_spread_pattern(visualizer, zim_spread; 
                            title="ZIM Spread Pattern (λ=2.5)")
    save_plot(p2, joinpath(save_dir, "spread_pattern.png"))
    
    # Example 3: Survival curve
    println("Creating survival curve...")
    λ_range = [1.0, 1.5, 2.0, 2.5, 3.0]
    # Mock data for demonstration - replace with actual analysis
    survival_data = [0.1, 0.3, 0.7, 0.9, 0.95]
    errors = [0.05, 0.1, 0.1, 0.05, 0.03]
    
    p3 = plot_survival_curve(λ_range, survival_data; std_errors=errors)
    save_plot(p3, joinpath(save_dir, "survival_curve.png"))
    
    println("Visualization examples saved to: $save_dir")
end

# =============================================================================
# Setup for Publication-Quality Plots
# =============================================================================

"""
Configure Plots.jl for publication-quality output.
"""
function setup_publication_plots()
    # Set default plot attributes for publication quality
    default(fontfamily = "Computer Modern",
           titlefontsize = 14,
           guidefontsize = 12, 
           tickfontsize = 10,
           legendfontsize = 10,
           linewidth = 2,
           gridwidth = 1,
           dpi = 300,
           size = (600, 400))
    
    println("Publication plot settings applied")
end