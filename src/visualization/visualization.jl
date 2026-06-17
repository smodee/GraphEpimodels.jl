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

# =============================================================================
# Shared Color Schemes and Styling
# =============================================================================

"""
Standard color schemes for epidemic visualizations.

Each scheme maps epidemic states to colors appropriate for different contexts.
"""
const COLOR_SCHEMES = Dict{Symbol, Dict{Symbol, Any}}(
    :zim => Dict(
        :susceptible => :white,
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
    ),

    :chaseescape => Dict(
        :susceptible => :white,        # empty
        :infected => :red,             # prey (red)
        :removed => :blue,             # predator (blue)
        :boundary => :black,
        :background => :white,
        :name => "Chase-Escape (Prey=red, Predator=blue)"
    ),

    :maki_thompson => Dict(
        :susceptible => :white,        # ignorant
        :infected => :orange,          # spreader
        :removed => :gray,             # stifler
        :boundary => :black,
        :background => :white,
        :name => "Maki-Thompson (Spreader=orange, Stifler=gray)"
    ),

    # Neutral default for when the model is unknown / unspecified.
    :general => Dict(
        :susceptible => :lightgray,
        :infected => :crimson,
        :removed => :steelblue,
        :boundary => :black,
        :background => :white,
        :name => "General (neutral default)"
    )
)

"""
Pick the color scheme that best matches an epidemic model.

Used by the rendering entry points so that omitting `color_scheme` yields a
model-appropriate palette (the user can always override with an explicit scheme).
Falls back to `:general` for unrecognized models.

Accepts either a process or a short process name (e.g. the `process_name` stored
on a `SimulationRecording`: "SIR", "ZIM", …).
"""
function default_color_scheme(name::AbstractString)::Symbol
    n = lowercase(name)
    if n == "zim"
        return :zim
    elseif n == "sir"
        return :sir
    elseif n in ("chaseescape", "chase-escape", "chase_escape")
        return :chaseescape
    elseif n in ("makithompson", "maki-thompson", "maki_thompson")
        return :maki_thompson
    else
        return :general
    end
end

default_color_scheme(process::AbstractEpidemicProcess)::Symbol =
    default_color_scheme(_process_name(process))

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
    return visualizer_for(graph; color_scheme = color_scheme, kwargs...)
end

# =============================================================================
# Visualizer Dispatch (graph type -> appropriate visualizer)
# =============================================================================
#
# Multiple dispatch picks the most specific method: all lattices (square,
# triangular, hexagonal) get the LatticeVisualizer; general graphs get the
# NetworkVisualizer. Adding a 3D branch later is purely additive. The concrete
# visualizer types are defined in lattice_viz.jl / network_viz.jl (included
# after this file); these bodies run only when called, so the forward reference
# is fine.

"""
Return an appropriate visualizer for `graph`, selected by its type.

- a *cell* lattice (square / triangular / hexagonal — `has_cells(graph)`) →
  `LatticeVisualizer` (dual-tiling cells)
- any other `AbstractEpidemicGraph` → `NetworkVisualizer` (node-link diagram)

The lattice method is the more specific one, but it routes by `has_cells`: a
lattice with a space-filling cell tiling uses the `LatticeVisualizer`, while a
lattice *without* cells (a 3D `CubeLattice`, or a d≥4 hypercubic lattice) can't be
drawn as cells and falls through to the `NetworkVisualizer`. Everything else
(general `AdjacencyGraph`, `ErdosRenyiGraph`, and the structured implicit graphs —
complete / cycle / path / star) also routes to the node-link `NetworkVisualizer`,
which can draw any graph: it reads `node_positions` + `get_neighbors` (in 3D where
the graph provides a 3D layout) and falls back to a spring layout otherwise.

Extra keyword arguments are forwarded to the visualizer constructor.
"""
function visualizer_for(graph::AbstractLatticeGraph; kwargs...)::AbstractVisualizer
    return has_cells(graph) ? LatticeVisualizer(; kwargs...) : NetworkVisualizer(; kwargs...)
end

function visualizer_for(graph::AbstractEpidemicGraph; kwargs...)::AbstractVisualizer
    return NetworkVisualizer(; kwargs...)
end

# =============================================================================
# Utility Functions for Process Analysis
# =============================================================================

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
        "Epidemic Simulation ($(_process_name(process)))"
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

# =============================================================================
# Makie-backed entry points (implemented in the CairoMakie extension)
# =============================================================================
#
# `render_frame`, `save_plot`, `animate_recording`, and
# `animate_simulation` are implemented in ext/GraphEpimodelsCairoMakieExt.jl,
# which loads only when the user runs `using CairoMakie`. They are declared here
# (as generic functions) so the package can export them and the extension can add
# the concrete, type-specialized methods. Without CairoMakie loaded, only these
# fallbacks exist and give an actionable error. (`visualize_state` already has a
# generic fallback above.)

const _MAKIE_HINT = "requires CairoMakie. Run `using CairoMakie` to enable plotting/animation."

render_frame(args...; kwargs...)      = error("render_frame $_MAKIE_HINT")
save_plot(args...; kwargs...)         = error("save_plot $_MAKIE_HINT")
animate_recording(args...; kwargs...) = error("animate_recording $_MAKIE_HINT")
animate_simulation(args...; kwargs...) = error("animate_simulation $_MAKIE_HINT")