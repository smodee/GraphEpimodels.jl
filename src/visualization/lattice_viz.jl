"""
`LatticeVisualizer` — visualizer for epidemic processes on regular lattices.

Renders any `AbstractLatticeGraph` — square, triangular, or hexagonal — using the
*dual-tiling* convention: each node is drawn as the polygonal cell dual to the
lattice, so the cell's side count equals the node's degree (square→square,
triangular→hexagon, hexagonal→triangle). Colors come from the shared
`COLOR_SCHEMES` table; state 0/1/2 maps to susceptible/infected/removed.

This file holds only the type and its backend-independent interface methods. The
Makie rendering (`render_frame`, `visualize_state`, `save_plot`) lives in
ext/GraphEpimodelsCairoMakieExt.jl and loads with `using CairoMakie`.
"""

# =============================================================================
# Lattice Visualizer
# =============================================================================

"""
Static visualizer for regular lattices (square, triangular, hexagonal).

# Fields
- `color_scheme::Symbol`: Color scheme (from visualization.jl)
- `show_boundary::Bool`: Outline the lattice boundary
- `figure_size::Tuple{Int, Int}`: Figure dimensions in pixels
- `show_grid::Bool`: Stroke cell outlines (cell path only)
"""
mutable struct LatticeVisualizer <: StaticVisualizer
    color_scheme::Symbol
    show_boundary::Bool
    figure_size::Tuple{Int, Int}
    show_grid::Bool

    function LatticeVisualizer(;
                              color_scheme::Symbol = :general,
                              show_boundary::Bool = false,
                              figure_size::Tuple{Int, Int} = (600, 600),
                              show_grid::Bool = false)
        if color_scheme ∉ available_color_schemes()
            throw(ArgumentError("Unknown color scheme: $color_scheme. Available: $(available_color_schemes())"))
        end
        new(color_scheme, show_boundary, figure_size, show_grid)
    end
end

function supported_graph_types(viz::LatticeVisualizer)::Vector{Type}
    return [SquareLattice, TriangularLattice, HexagonalLattice]
end

# The LatticeVisualizer draws dual-tiling cells, so it can handle a lattice only
# if that lattice actually supplies a cell tiling. A cell-less lattice (3D cube,
# d≥4 hypercubic) is routed to the NetworkVisualizer by `visualizer_for`; this
# guard turns a forced mismatch into a clear error instead of a `cell_polygons`
# failure deep in rendering.
can_visualize(viz::LatticeVisualizer, graph::AbstractLatticeGraph)::Bool = has_cells(graph)
