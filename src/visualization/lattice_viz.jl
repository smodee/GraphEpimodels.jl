"""
Makie visualization for epidemic processes on regular lattices.

`LatticeVisualizer` renders any `AbstractLatticeGraph` — square, triangular, or
hexagonal — using the *dual-tiling* convention: each node is drawn as the
polygonal cell dual to the lattice, so the cell's side count equals the node's
degree (square→square, triangular→hexagon, hexagonal→triangle). Square lattices
take a fast `image!` path; triangular/hexagonal use `poly!` over their dual
cells (from `cell_polygons`).

Colors come from the shared `COLOR_SCHEMES` table; state 0/1/2 maps to
susceptible/infected/removed.
"""

using CairoMakie

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
                              color_scheme::Symbol = :zim,
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

can_visualize(viz::LatticeVisualizer, graph::AbstractLatticeGraph)::Bool = true

# =============================================================================
# Color helpers
# =============================================================================

"""Resolve a color scheme to concrete `(susceptible, infected, removed)` colors."""
function _state_colors(scheme::Symbol; transparent_background::Bool = false)
    cs = COLOR_SCHEMES[scheme]
    susceptible = transparent_background ? RGBAf(0, 0, 0, 0) : to_color(cs[:susceptible])
    return (susceptible, to_color(cs[:infected]), to_color(cs[:removed]))
end

"""Per-node color vector for the given raw states (0=S, 1=I, 2=R)."""
function _node_colors(scheme::Symbol, states_raw::Vector{Int8};
                      transparent_background::Bool = false)
    cS, cI, cR = _state_colors(scheme; transparent_background = transparent_background)
    palette = (cS, cI, cR)
    return [palette[Int(s) + 1] for s in states_raw]
end

# =============================================================================
# Rendering core
# =============================================================================

"""Create a clean figure + axis (equal aspect, no decorations) for a lattice frame."""
function _lattice_axis(viz::LatticeVisualizer; title::String = "",
                       transparent_background::Bool = false)
    bg = transparent_background ? :transparent : :white
    fig = Figure(size = viz.figure_size, backgroundcolor = bg)
    ax = Axis(fig[1, 1]; title = title, aspect = DataAspect(),
              backgroundcolor = bg)
    hidedecorations!(ax)
    hidespines!(ax)
    return fig, ax
end

"""
Render a single lattice state (raw Int8 vector) to a Makie `Figure`.

Shared by `visualize_state` (live state) and the animation builder (recorded
frames), so animation frames look identical to static snapshots.
"""
function render_frame(viz::LatticeVisualizer, lattice::AbstractLatticeGraph,
                      states_raw::Vector{Int8}; title::String = "",
                      transparent_background::Bool = false)
    fig, ax = _lattice_axis(viz; title = title,
                            transparent_background = transparent_background)
    _draw_lattice!(ax, viz, lattice, states_raw;
                   transparent_background = transparent_background)
    return fig
end

# Square lattice — two paths depending on whether transparency is needed.
#
# `image!` is fast but composites alpha against the Cairo surface before writing
# pixels, so transparent susceptible cells (RGBAf(0,0,0,0)) appear as the
# background colour rather than as true transparency in the saved PNG.
# `poly!` renders each cell as a vector path fill; alpha=0 fills are genuinely
# absent from the output, giving real per-cell transparency.
function _draw_lattice!(ax, viz::LatticeVisualizer, lattice::SquareLattice,
                        states_raw::Vector{Int8}; transparent_background::Bool = false)
    width, height = lattice.width, lattice.height
    cS, cI, cR = _state_colors(viz.color_scheme;
                               transparent_background = transparent_background)
    palette = (cS, cI, cR)

    if transparent_background
        polys  = Vector{Vector{Point2f}}(undef, lattice.n_nodes)
        colors = Vector{RGBAf}(undef, lattice.n_nodes)
        @inbounds for idx in 1:lattice.n_nodes
            r, c = index_to_coord(lattice, idx)
            xf, yf = Float32(c), Float32(r)
            polys[idx]  = [Point2f(xf - 0.5f0, yf - 0.5f0),
                           Point2f(xf + 0.5f0, yf - 0.5f0),
                           Point2f(xf + 0.5f0, yf + 0.5f0),
                           Point2f(xf - 0.5f0, yf + 0.5f0)]
            colors[idx] = palette[Int(states_raw[idx]) + 1]
        end
        strokewidth = viz.show_grid ? 0.5 : 0.0
        poly!(ax, polys; color = colors, strokewidth = strokewidth,
              strokecolor = (:gray, 0.5))
    else
        # Fast path via a per-pixel color image.
        img = Matrix{RGBAf}(undef, width, height)
        @inbounds for idx in 1:lattice.n_nodes
            r, c = index_to_coord(lattice, idx)
            img[c, r] = palette[Int(states_raw[idx]) + 1]
        end
        image!(ax, (0.5, width + 0.5), (0.5, height + 0.5), img; interpolate = false)
    end

    if viz.show_boundary && has_boundary(lattice)
        _draw_boundary_box!(ax, 0.5, width + 0.5, 0.5, height + 0.5)
    end
    return ax
end

# Triangular / hexagonal: draw the dual cells as polygons.
function _draw_lattice!(ax, viz::LatticeVisualizer, lattice::AbstractLatticeGraph,
                        states_raw::Vector{Int8}; transparent_background::Bool = false)
    cells = cell_polygons(lattice)
    colors = _node_colors(viz.color_scheme, states_raw;
                          transparent_background = transparent_background)

    polys = [Point2f.(eachcol(c)) for c in cells]
    strokewidth = viz.show_grid ? 0.5 : 0.0
    poly!(ax, polys; color = colors, strokewidth = strokewidth, strokecolor = (:gray, 0.5))

    if viz.show_boundary
        pos = node_positions(lattice)
        xmin, xmax = extrema(@view pos[1, :])
        ymin, ymax = extrema(@view pos[2, :])
        _draw_boundary_box!(ax, xmin - 0.6, xmax + 0.6, ymin - 0.6, ymax + 0.6)
    end
    return ax
end

"""Stroke a rectangle outline (boundary emphasis)."""
function _draw_boundary_box!(ax, xlo, xhi, ylo, yhi)
    lines!(ax, [xlo, xhi, xhi, xlo, xlo], [ylo, ylo, yhi, yhi, ylo];
           color = :black, linewidth = 2)
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

function visualize_state(viz::LatticeVisualizer, process::AbstractEpidemicProcess;
                         transparent_background::Bool = false)
    validate_visualizer_compatibility(viz, process)
    graph = get_graph(process)
    return render_frame(viz, graph, node_states_raw(graph);
                        title = generate_visualization_title(process),
                        transparent_background = transparent_background)
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
            value ∈ available_color_schemes() ||
                throw(ArgumentError("Unknown color scheme: $value"))
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
# Convenience Functions
# =============================================================================

"""
Save a lattice visualization to file (format from the extension).

# Arguments
- `process::AbstractEpidemicProcess`: Process to visualize
- `filename::String`: Output path (`.png`, `.pdf`, `.svg`)
- `color_scheme::Symbol`: Color scheme (default: `:zim`)
- `figure_size::Tuple{Int, Int}`: Plot dimensions in pixels (default: (800, 800))
- `transparent_background::Bool`: Transparent background + susceptible cells (default: false)
- `show_boundary::Bool`: Outline the lattice boundary (default: false)
"""
function save_lattice_plot(process::AbstractEpidemicProcess, filename::String;
                          color_scheme::Symbol = :zim,
                          figure_size::Tuple{Int, Int} = (800, 800),
                          transparent_background::Bool = false,
                          show_boundary::Bool = false,
                          show_grid::Bool = false)
    viz = LatticeVisualizer(color_scheme = color_scheme,
                            figure_size = figure_size,
                            show_boundary = show_boundary,
                            show_grid = show_grid)
    fig = visualize_state(viz, process; transparent_background = transparent_background)
    save(filename, fig)
    println("Lattice visualization saved to: $filename")
    return fig
end
