"""
Makie visualization for epidemic processes on general graphs.

`NetworkVisualizer` draws an `AdjacencyGraph` as a node-link diagram: edges as
line segments, nodes as colored markers. Node positions come from coordinates
attached to the graph (`has_layout`); otherwise a force-directed layout is
computed with NetworkLayout.
"""

using CairoMakie
import NetworkLayout

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
                              color_scheme::Symbol = :sir,
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
    return [AdjacencyGraph]
end

can_visualize(viz::NetworkVisualizer, graph::AdjacencyGraph)::Bool = true

# =============================================================================
# Layout
# =============================================================================

"""
Resolve node positions for a graph: attached coordinates if present, else a
force-directed (spring) layout. Returns a `2 × n` `Matrix{Float64}`.
"""
function _resolve_positions(graph::AdjacencyGraph)::Matrix{Float64}
    if has_layout(graph)
        pos = node_positions(graph)
        return size(pos, 1) == 2 ? pos : pos[1:2, :]   # drop z for 2D draw
    end
    n = num_nodes(graph)
    adj = falses(n, n)
    @inbounds for i in 1:n, j in get_neighbors(graph, i)
        adj[i, j] = true
    end
    pts = NetworkLayout.spring(adj; seed = 1)
    mat = Matrix{Float64}(undef, 2, n)
    @inbounds for i in 1:n
        mat[1, i] = pts[i][1]
        mat[2, i] = pts[i][2]
    end
    return mat
end

# =============================================================================
# Rendering core
# =============================================================================

"""Draw a network frame into an existing axis (edges + colored node markers)."""
function _draw_network!(ax, viz::NetworkVisualizer, graph::AdjacencyGraph,
                        states_raw::Vector{Int8};
                        positions::Union{Matrix{Float64}, Nothing} = nothing)
    pos = positions === nothing ? _resolve_positions(graph) : positions

    if viz.show_edges
        segs = Point2f[]
        @inbounds for i in 1:num_nodes(graph), j in get_neighbors(graph, i)
            i < j || continue
            push!(segs, Point2f(pos[1, i], pos[2, i]))
            push!(segs, Point2f(pos[1, j], pos[2, j]))
        end
        isempty(segs) || linesegments!(ax, segs; color = viz.edge_color)
    end

    colors = _node_colors(viz.color_scheme, states_raw)
    scatter!(ax, pos[1, :], pos[2, :]; color = colors, markersize = viz.node_size,
             strokewidth = 0.5, strokecolor = :black)
    return ax
end

function render_frame(viz::NetworkVisualizer, graph::AdjacencyGraph,
                      states_raw::Vector{Int8}; title::String = "",
                      positions::Union{Matrix{Float64}, Nothing} = nothing)
    fig = Figure(size = viz.figure_size)
    ax = Axis(fig[1, 1]; title = title, aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    _draw_network!(ax, viz, graph, states_raw; positions = positions)
    return fig
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

function visualize_state(viz::NetworkVisualizer, process::AbstractEpidemicProcess)
    validate_visualizer_compatibility(viz, process)
    graph = get_graph(process)
    return render_frame(viz, graph, node_states_raw(graph);
                        title = generate_visualization_title(process))
end

function get_visualization_settings(viz::NetworkVisualizer)::Dict{Symbol, Any}
    return Dict{Symbol, Any}(
        :color_scheme => viz.color_scheme,
        :figure_size => viz.figure_size,
        :node_size => viz.node_size,
        :show_edges => viz.show_edges
    )
end
