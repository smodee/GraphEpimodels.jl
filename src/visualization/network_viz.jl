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
- `node_size::Float64`: Marker size for nodes. The auto-selected visualizer
  (`visualizer_for`) scales this to the node count so large graphs don't crowd; an
  explicit `node_size` keyword overrides that.
- `show_edges::Bool`: Draw edges between nodes
- `edge_color`: Color for edges (a color or `(color, alpha)` tuple)
- `edge_width::Float64`: Line width for edges
- `dim::Int`: Drawing dimension, 2 or 3. With `dim = 3` the graph is laid out and
  drawn in 3D (intrinsic 3D layout where the graph provides one — star, complete,
  tree — otherwise a 3D force-directed layout). Lattices are 2D only and are not
  drawn by this visualizer.
- `dark::Bool`: Dark theme — render on a dark background with light title text and
  contrast-adjusted basemap / node outlines (default `false`).
"""
mutable struct NetworkVisualizer <: StaticVisualizer
    color_scheme::Symbol
    figure_size::Tuple{Int, Int}
    node_size::Float64
    show_edges::Bool
    edge_color
    edge_width::Float64
    dim::Int
    dark::Bool

    function NetworkVisualizer(;
                              color_scheme::Symbol = :general,
                              figure_size::Tuple{Int, Int} = (700, 700),
                              node_size::Real = 12.0,
                              show_edges::Bool = true,
                              edge_color = (:gray, 0.5),
                              edge_width::Real = 1.0,
                              dim::Int = 2,
                              dark::Bool = false)
        color_scheme ∈ available_color_schemes() ||
            throw(ArgumentError("Unknown color scheme: $color_scheme"))
        dim == 2 || dim == 3 ||
            throw(ArgumentError("NetworkVisualizer dim must be 2 or 3, got $dim"))
        new(color_scheme, figure_size, Float64(node_size), show_edges, edge_color,
            Float64(edge_width), dim, dark)
    end
end

"""
Adaptive default marker size for a node-link graph of `n` nodes.

The historical fixed default (`12.0`) crowds large graphs (a 100+-node country, a
several-hundred-node Erdős–Rényi). This shrinks the marker as the node count grows
(roughly `∝ 1/√n` past a small reference count), with a floor so markers never
vanish and a cap at the historical default so small graphs render exactly as before.
Used by `visualizer_for`; pass an explicit `node_size` to bypass it.
"""
_adaptive_node_size(n::Integer)::Float64 = n <= 1 ? 12.0 : clamp(12.0 * sqrt(30 / n), 3.0, 12.0)

function supported_graph_types(viz::NetworkVisualizer)::Vector{Type}
    return [AdjacencyGraph, ErdosRenyiGraph,
            CompleteGraph, CycleGraph, PathGraph, StarGraph, RegularTree]
end

# A node-link visualizer can draw *any* graph: positions come from an attached
# layout when present, or a force-directed fallback otherwise.
can_visualize(viz::NetworkVisualizer, graph::AbstractEpidemicGraph)::Bool = true

# Visualizer dispatch lives in visualization.jl: the generic
# `visualizer_for(::AbstractEpidemicGraph)` fallback routes every non-lattice graph
# (general `AdjacencyGraph`, `ErdosRenyiGraph`, and the structured implicit graphs)
# to the `NetworkVisualizer`.
