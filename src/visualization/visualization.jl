"""
Abstract visualization interface for epidemic processes on graphs.

This module defines the interface that all graph visualizers must implement,
along with shared utilities for colors, plotting, and common visualization tasks.
"""

# Import required interfaces (assumes graphs.jl and epiprocess.jl are loaded)

# =============================================================================
# Abstract Visualizer Types
# =============================================================================

"""
Abstract base type for all epidemic process visualizers.

All concrete visualizer implementations must inherit from this type and 
implement the required interface methods.
"""
abstract type AbstractVisualizer end

"""
Abstract type for static visualizers that produce single images/plots.

Examples:
- Heatmap visualizers for lattices
- Node-link diagrams for networks
- State distribution plots
"""
abstract type StaticVisualizer <: AbstractVisualizer end

"""
Abstract type for interactive visualizers with dynamic capabilities.

Examples:
- Interactive network browsers
- Animation controllers
- Real-time simulation viewers
"""
abstract type InteractiveVisualizer <: AbstractVisualizer end

# =============================================================================
# Required Interface Methods (must be implemented by all visualizer types)
# =============================================================================

"""
Create a visualization of the current process state.

# Arguments
- `visualizer::AbstractVisualizer`: The visualizer
- `process::AbstractEpidemicProcess`: The process to visualize

# Returns
- Visualization object (type depends on implementation - could be Plot, Figure, etc.)
"""
function visualize_state(visualizer::AbstractVisualizer, process::AbstractEpidemicProcess)
    error("visualize_state must be implemented by concrete visualizer type $(typeof(visualizer))")
end

"""
Get supported graph types for this visualizer.

# Arguments
- `visualizer::AbstractVisualizer`: The visualizer

# Returns
- `Vector{Type}`: Vector of supported AbstractEpidemicGraph types
"""
function supported_graph_types(visualizer::AbstractVisualizer)::Vector{Type}
    error("supported_graph_types must be implemented by concrete visualizer type $(typeof(visualizer))")
end

# =============================================================================
# Optional Interface Methods (have default implementations)
# =============================================================================

"""
Check if this visualizer can handle the given graph type.

# Arguments
- `visualizer::AbstractVisualizer`: The visualizer
- `graph::AbstractEpidemicGraph`: The graph to check

# Returns
- `Bool`: true if visualizer supports this graph type
"""
function can_visualize(visualizer::AbstractVisualizer, graph::AbstractEpidemicGraph)::Bool
    graph_type = typeof(graph)
    return graph_type in supported_graph_types(visualizer)
end

"""
Get visualization settings/parameters for this visualizer.

Default implementation returns empty dictionary. Override for configurable visualizers.

# Returns
- `Dict{Symbol, Any}`: Current visualization settings
"""
function get_visualization_settings(visualizer::AbstractVisualizer)::Dict{Symbol, Any}
    return Dict{Symbol, Any}()
end

"""
Update visualization settings/parameters.

Default implementation does nothing. Override for configurable visualizers.

# Arguments
- `settings::Dict{Symbol, Any}`: New settings to apply
"""
function set_visualization_settings!(visualizer::AbstractVisualizer, settings::Dict{Symbol, Any})
    # Default: do nothing
end

# =============================================================================
# Shared Color Schemes and Styling
# =============================================================================

"""
Standard color schemes for epidemic visualizations.

Each scheme maps epidemic states to colors appropriate for different contexts.
"""
const COLOR_SCHEMES = Dict{Symbol, Dict{Symbol, Any}}(
    :zim => Dict(
        :susceptible => :lightgray,
        :infected => :forestgreen,     # Green for zombies
        :removed => :red,              # Red for killed  
        :boundary => :darkgray,
        :background => :white,
        :name => "ZIM (Zombies=green, Dead=red)"
    ),
    
    :sir => Dict(
        :susceptible => :lightblue,
        :infected => :red,             # Red for infected
        :removed => :darkgray,         # Gray for recovered
        :boundary => :black,
        :background => :white,
        :name => "SIR (Infected=red, Recovered=gray)"
    ),
    
    :medical => Dict(
        :susceptible => :white,
        :infected => :orange,          # Orange for symptomatic  
        :removed => :purple,           # Purple for immune
        :boundary => :gray,
        :background => :white,
        :name => "Medical (Orange=infected, Purple=immune)"
    ),
    
    :contrast => Dict(
        :susceptible => :white,
        :infected => :black,           # High contrast
        :removed => :gray,
        :boundary => :red,
        :background => :white,
        :name => "High Contrast (Black/white)"
    ),
    
    :colorblind => Dict(
        :susceptible => :white,
        :infected => :blue,            # Blue/orange for colorblind accessibility
        :removed => :darkorange,       
        :boundary => :black,
        :background => :white,
        :name => "Colorblind Safe (Blue/orange)"
    )
)

"""
Get color for a specific epidemic state from a color scheme.

# Arguments
- `state::NodeState`: The epidemic state
- `scheme::Symbol`: Color scheme name (default: :zim)

# Returns
- Color value (type depends on plotting backend)
"""
function get_state_color(state::NodeState, scheme::Symbol = :zim)
    if scheme ∉ keys(COLOR_SCHEMES)
        throw(ArgumentError("Unknown color scheme: $scheme. Available: $(keys(COLOR_SCHEMES))"))
    end
    
    color_map = COLOR_SCHEMES[scheme]
    
    return if state == SUSCEPTIBLE
        color_map[:susceptible]
    elseif state == INFECTED  
        color_map[:infected]
    elseif state == REMOVED
        color_map[:removed]
    else
        color_map[:background]  # Fallback
    end
end

"""
Get all available color scheme names.

# Returns
- `Vector{Symbol}`: Available color scheme names
"""
function available_color_schemes()::Vector{Symbol}
    return collect(keys(COLOR_SCHEMES))
end

"""
Print information about available color schemes.
"""
function print_color_schemes()
    println("Available color schemes:")
    for (scheme_name, scheme) in COLOR_SCHEMES
        println("  :$scheme_name - $(scheme[:name])")
    end
end

# =============================================================================
# Shared Plotting Utilities
# =============================================================================

"""
Standard figure size presets for different use cases.
"""
const FIGURE_SIZES = Dict{Symbol, Tuple{Int, Int}}(
    :small => (400, 400),
    :medium => (600, 600),  
    :large => (800, 800),
    :wide => (1000, 600),
    :presentation => (1200, 800),
    :paper => (600, 400)
)

"""
Get figure size for a given preset.

# Arguments
- `size_preset::Symbol`: Size preset name

# Returns
- `Tuple{Int, Int}`: (width, height) in pixels
"""
function get_figure_size(size_preset::Symbol)::Tuple{Int, Int}
    if size_preset ∉ keys(FIGURE_SIZES)
        throw(ArgumentError("Unknown size preset: $size_preset. Available: $(keys(FIGURE_SIZES))"))
    end
    return FIGURE_SIZES[size_preset]
end

# =============================================================================
# Validation and Error Handling
# =============================================================================

"""
Validate that a visualizer can handle the given process.

# Arguments
- `visualizer::AbstractVisualizer`: The visualizer
- `process::AbstractEpidemicProcess`: The process to visualize

# Throws
- `ArgumentError`: If visualizer cannot handle this process type
"""
function validate_visualizer_compatibility(visualizer::AbstractVisualizer, 
                                         process::AbstractEpidemicProcess)
    graph = get_graph(process)
    
    if !can_visualize(visualizer, graph)
        supported_types = supported_graph_types(visualizer)
        graph_type = typeof(graph)
        
        throw(ArgumentError(
            "Visualizer $(typeof(visualizer)) cannot handle graph type $graph_type. " *
            "Supported types: $supported_types"
        ))
    end
end

# =============================================================================
# Factory Functions for Common Visualization Tasks
# =============================================================================

"""
Create an appropriate visualizer for the given graph type.

Automatically selects the best visualizer implementation based on graph type.
This is a convenience function for users who don't want to choose manually.

# Arguments
- `graph::AbstractEpidemicGraph`: The graph to visualize
- `color_scheme::Symbol`: Color scheme to use (default: :zim)
- `kwargs...`: Additional arguments passed to visualizer constructor

# Returns
- `AbstractVisualizer`: Appropriate visualizer for this graph type
"""
function create_auto_visualizer(graph::AbstractEpidemicGraph, 
                               color_scheme::Symbol = :zim; 
                               kwargs...)::AbstractVisualizer
    
    # This would be implemented once we have concrete visualizer types
    # For now, throw an informative error
    graph_type = typeof(graph)
    
    error("""
    Auto-visualizer creation not yet implemented for graph type: $graph_type
    
    Please use a specific visualizer:
    - For SquareLattice: use LatticeVisualizer from lattice_viz.jl  
    - For AdjacencyGraph: use NetworkVisualizer from network_viz.jl
    
    Example:
      visualizer = LatticeVisualizer(color_scheme=:$color_scheme)
      plot = visualize_state(visualizer, process)
    """)
end

# =============================================================================
# Utility Functions for Process Analysis
# =============================================================================

"""
Extract visualization data from process state.

Converts process state into a format suitable for visualization.
This is a common operation that many visualizers need.

# Arguments
- `process::AbstractEpidemicProcess`: The process

# Returns
- `Dict{Symbol, Any}`: Visualization data including states, statistics, etc.
"""
function extract_visualization_data(process::AbstractEpidemicProcess)::Dict{Symbol, Any}
    graph = get_graph(process)
    states = node_states_raw(graph)  # Use raw states for performance
    statistics = get_statistics(process)
    
    # Count states for summary info
    state_counts = count_states(graph)
    
    return Dict{Symbol, Any}(
        :node_states => states,
        :state_counts => state_counts,
        :statistics => statistics,
        :graph => graph,
        :num_nodes => num_nodes(graph),
        :has_boundary => has_boundary(graph),
        :boundary_nodes => get_boundary_nodes(graph)
    )
end

"""
Generate title text for epidemic visualizations.

Creates informative titles based on process state and parameters.

# Arguments  
- `process::AbstractEpidemicProcess`: The process
- `custom_title::Union{String, Nothing}`: Custom title prefix (optional)

# Returns
- `String`: Generated title text
"""
function generate_visualization_title(process::AbstractEpidemicProcess,
                                    custom_title::Union{String, Nothing} = nothing)::String
    stats = get_statistics(process)
    
    base_title = if custom_title !== nothing
        custom_title
    else
        process_name = if isa(process, ZIMProcess)
            "ZIM"
        else
            string(typeof(process))
        end
        "Epidemic Simulation ($process_name)"
    end
    
    info_parts = String[]
    
    # Add time info
    if stats[:time] > 0
        push!(info_parts, "t=$(round(stats[:time], digits=2))")
    end
    
    # Add step info  
    if stats[:step_count] > 0
        push!(info_parts, "steps=$(stats[:step_count])")
    end
    
    # Add state counts
    if stats[:infected] > 0 || stats[:removed] > 0
        push!(info_parts, "I=$(stats[:infected]), R=$(stats[:removed])")
    end
    
    if !isempty(info_parts)
        return base_title * " (" * join(info_parts, ", ") * ")"
    else
        return base_title
    end
end