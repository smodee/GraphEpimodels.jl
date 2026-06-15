"""
`NetworkVisualizer` — visualizer for epidemic processes on general graphs.

Draws an `AdjacencyGraph` (or an `ErdosRenyiGraph`, which wraps one) as a
node-link diagram: edges as line segments, nodes as colored markers. Node
positions come from coordinates attached to the graph (`has_layout`); otherwise a
force-directed layout is computed with NetworkLayout.

This file holds only the type and its backend-independent interface methods. The
Makie rendering (`render_frame`, `visualize_state`, layout/drawing) lives in
ext/GraphEpimodelsCairoMakieExt.jl and loads with `using CairoMakie`.
"""

# =============================================================================
# Network Visualizer
# =============================================================================

"""
Static node-link visualizer for `AdjacencyGraph`.

# Fields
- `color_scheme::Symbol`: Color scheme (from visualization.jl)
- `figure_size::Tuple{Int, Int}`: Figure dimensions in pixels
- `node_size::Float64`: Marker size for nodes
- `show_edges::Bool`: Draw edges between nodes
- `edge_color`: Color for edges
"""
mutable struct NetworkVisualizer <: StaticVisualizer
    color_scheme::Symbol
    figure_size::Tuple{Int, Int}
    node_size::Float64
    show_edges::Bool
    edge_color

    function NetworkVisualizer(;
                              color_scheme::Symbol = :general,
                              figure_size::Tuple{Int, Int} = (700, 700),
                              node_size::Real = 12.0,
                              show_edges::Bool = true,
                              edge_color = (:gray, 0.5))
        color_scheme ∈ available_color_schemes() ||
            throw(ArgumentError("Unknown color scheme: $color_scheme"))
        new(color_scheme, figure_size, Float64(node_size), show_edges, edge_color)
    end
end

function supported_graph_types(viz::NetworkVisualizer)::Vector{Type}
    return [AdjacencyGraph, ErdosRenyiGraph,
            CompleteGraph, CycleGraph, PathGraph, StarGraph, RegularTree]
end

# A node-link visualizer can draw *any* graph: positions come from an attached
# layout when present, or a force-directed fallback otherwise.
can_visualize(viz::NetworkVisualizer, graph::AbstractEpidemicGraph)::Bool = true

function get_visualization_settings(viz::NetworkVisualizer)::Dict{Symbol, Any}
    return Dict{Symbol, Any}(
        :color_scheme => viz.color_scheme,
        :figure_size => viz.figure_size,
        :node_size => viz.node_size,
        :show_edges => viz.show_edges
    )
end

# Visualizer dispatch lives in visualization.jl: the generic
# `visualizer_for(::AbstractEpidemicGraph)` fallback routes every non-lattice graph
# (general `AdjacencyGraph`, `ErdosRenyiGraph`, and the structured implicit graphs)
# to the `NetworkVisualizer`.
