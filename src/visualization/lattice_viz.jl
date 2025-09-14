"""
Heatmap visualization for epidemic processes on square lattices.

Provides efficient heatmap rendering for large lattices with customizable
colors and styling options.
"""

# Import required packages and interfaces
# Note: This assumes Plots.jl is available (commented out in main module)
using Plots

# Import visualization interface (assumes visualization.jl is loaded)

# =============================================================================
# Lattice Heatmap Visualizer
# =============================================================================

"""
Static heatmap visualizer for square lattices.

Creates 2D heatmap visualizations showing the spatial distribution of epidemic
states across the lattice. Optimized for large lattices with efficient rendering.

# Fields
- `color_scheme::Symbol`: Color scheme to use (from visualization.jl)
- `show_boundary::Bool`: Whether to highlight boundary nodes
- `figure_size::Tuple{Int, Int}`: Figure dimensions in pixels
- `show_grid::Bool`: Whether to show grid lines between nodes
"""
mutable struct LatticeVisualizer <: StaticVisualizer
    color_scheme::Symbol
    show_boundary::Bool
    figure_size::Tuple{Int, Int}
    show_grid::Bool
    
    function LatticeVisualizer(; 
                              color_scheme::Symbol = :zim,
                              show_boundary::Bool = false,
                              figure_size::Tuple{Int, Int} = (600, 600),
                              show_grid::Bool = false)
        
        # Validate color scheme
        if color_scheme ∉ available_color_schemes()
            throw(ArgumentError("Unknown color scheme: $color_scheme. Available: $(available_color_schemes())"))
        end
        
        new(color_scheme, show_boundary, figure_size, show_grid)
    end
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

function supported_graph_types(viz::LatticeVisualizer)::Vector{Type}
    return [SquareLattice]
end

function visualize_state(viz::LatticeVisualizer, process::AbstractEpidemicProcess)
    # Validate compatibility
    validate_visualizer_compatibility(viz, process)
    
    graph = get_graph(process)
    
    # Extract lattice dimensions and states
    width, height = graph.width, graph.height
    states_raw = node_states_raw(graph)
    
    # Convert linear state array to 2D matrix for heatmap
    state_matrix = _states_to_matrix(states_raw, height, width)
    
    # Create color matrix based on states
    color_matrix = _create_color_matrix(state_matrix, viz.color_scheme)
    
    # Create the heatmap plot
    plot_title = generate_visualization_title(process)
    
    p = heatmap(color_matrix,
               title = plot_title,
               size = viz.figure_size,
               aspect_ratio = :equal,
               showaxis = false,
               grid = viz.show_grid,
               colorbar = false)  # We use discrete colors, not continuous
    
    # Add boundary highlighting if requested
    if viz.show_boundary && has_boundary(graph)
        _add_boundary_overlay!(p, graph, viz)
    end
    
    return p
end

# =============================================================================
# Optional Interface Implementation  
# =============================================================================

function get_visualization_settings(viz::LatticeVisualizer)::Dict{Symbol, Any}
    return Dict{Symbol, Any}(
        :color_scheme => viz.color_scheme,
        :show_boundary => viz.show_boundary,
        :figure_size => viz.figure_size,
        :show_grid => viz.show_grid
    )
end

function set_visualization_settings!(viz::LatticeVisualizer, settings::Dict{Symbol, Any})
    for (key, value) in settings
        if key == :color_scheme
            if value ∉ available_color_schemes()
                throw(ArgumentError("Unknown color scheme: $value"))
            end
            viz.color_scheme = value
        elseif key == :show_boundary
            viz.show_boundary = value
        elseif key == :figure_size
            viz.figure_size = value
        elseif key == :show_grid
            viz.show_grid = value
        else
            @warn "Unknown setting: $key"
        end
    end
end

# =============================================================================
# Internal Helper Functions
# =============================================================================

"""
Convert linear state array to 2D matrix for heatmap visualization.

The state array uses column-major indexing (as in our lattice implementation),
so we need to reshape and transpose appropriately.
"""
function _states_to_matrix(states_raw::Vector{Int8}, height::Int, width::Int)::Matrix{Int8}
    # Reshape from column-major linear array to 2D matrix
    # lattice uses: index = col + (row-1)*height
    state_matrix = reshape(states_raw, height, width)
    
    # Transpose so that matrix[row, col] corresponds to lattice position (row, col)
    return transpose(state_matrix)
end

"""
Create a color matrix from state matrix using the specified color scheme.

Returns a matrix of colors that Plots.jl can use directly for heatmap rendering.
"""
function _create_color_matrix(state_matrix::Matrix{Int8}, scheme::Symbol)
    rows, cols = size(state_matrix)
    color_matrix = Matrix{Symbol}(undef, rows, cols)
    
    # Get color scheme
    colors = COLOR_SCHEMES[scheme]
    
    # Map each state to its color
    for i in 1:rows, j in 1:cols
        state_int = state_matrix[i, j]
        
        color_matrix[i, j] = if state_int == state_to_int(SUSCEPTIBLE)
            colors[:susceptible]
        elseif state_int == state_to_int(INFECTED)
            colors[:infected]
        elseif state_int == state_to_int(REMOVED)
            colors[:removed]
        else
            colors[:background]  # Fallback
        end
    end
    
    return color_matrix
end

"""
Add boundary highlighting overlay to the plot.

Draws a border around the lattice to emphasize boundary conditions.
"""
function _add_boundary_overlay!(p, lattice::SquareLattice, viz::LatticeVisualizer)
    # Only add boundary for absorbing lattices
    if lattice.boundary != ABSORBING
        return
    end
    
    width, height = lattice.width, lattice.height
    
    # Add boundary rectangle
    plot!(p, 
          [0.5, width + 0.5, width + 0.5, 0.5, 0.5],
          [0.5, 0.5, height + 0.5, height + 0.5, 0.5],
          line = (3, :black, 0.8),
          fill = false,
          label = "")
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
Quick heatmap visualization with default settings.

Convenience function for rapid visualization without creating a visualizer object.

# Arguments
- `process::AbstractEpidemicProcess`: Process with SquareLattice graph
- `color_scheme::Symbol`: Color scheme to use (default: :zim)

# Returns
- Plots.jl plot object

# Example
```julia
julia> zim = create_zim_simulation(50, 50, 2.0)
julia> run_simulation(zim; max_time=20.0)
julia> p = quick_lattice_plot(zim)
julia> display(p)
```
"""
function quick_lattice_plot(process::AbstractEpidemicProcess, color_scheme::Symbol = :zim)
    viz = LatticeVisualizer(color_scheme=color_scheme)
    return visualize_state(viz, process)
end

"""
Create a comparison plot showing multiple lattice states side by side.

# Arguments
- `processes::Vector{AbstractEpidemicProcess}`: Processes to compare
- `titles::Vector{String}`: Titles for each subplot (optional)
- `color_scheme::Symbol`: Color scheme to use (default: :zim)

# Returns
- Plots.jl plot object with subplots

# Example  
```julia
julia> processes = [zim1, zim2, zim3]
julia> titles = ["λ=1.5", "λ=2.0", "λ=2.5"]  
julia> p = plot_lattice_comparison(processes, titles)
```
"""
function plot_lattice_comparison(processes::Vector{<:AbstractEpidemicProcess},
                                titles::Vector{String} = String[],
                                color_scheme::Symbol = :zim)
    
    n_plots = length(processes)
    if isempty(titles)
        titles = ["Process $i" for i in 1:n_plots]
    elseif length(titles) != n_plots
        throw(ArgumentError("Number of titles must match number of processes"))
    end
    
    # Create individual plots
    viz = LatticeVisualizer(color_scheme=color_scheme, figure_size=(300, 300))
    plots = []
    
    for (i, process) in enumerate(processes)
        # Validate each process
        validate_visualizer_compatibility(viz, process)
        
        p = visualize_state(viz, process)
        plot!(p, title=titles[i])
        push!(plots, p)
    end
    
    # Combine into layout
    layout = if n_plots <= 3
        (1, n_plots)  # Single row
    else
        # Multiple rows, try to make roughly square
        rows = Int(ceil(sqrt(n_plots)))
        cols = Int(ceil(n_plots / rows))
        (rows, cols)
    end
    
    return plot(plots..., layout=layout, size=(300 * layout[2], 300 * layout[1]))
end

"""
Save lattice visualization to file.

# Arguments
- `process::AbstractEpidemicProcess`: Process to visualize
- `filename::String`: Output filename (extension determines format)
- `color_scheme::Symbol`: Color scheme to use (default: :zim)
- `dpi::Int`: Resolution for raster formats (default: 300)

# Example
```julia
julia> save_lattice_plot(zim, "simulation_result.png")
```
"""
function save_lattice_plot(process::AbstractEpidemicProcess, filename::String;
                          color_scheme::Symbol = :zim, dpi::Int = 300)
    viz = LatticeVisualizer(color_scheme=color_scheme, figure_size=(800, 800))
    p = visualize_state(viz, process)
    
    savefig(p, filename)
    println("Lattice visualization saved to: $filename")
end