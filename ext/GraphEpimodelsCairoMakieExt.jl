"""
GraphEpimodelsCairoMakieExt — Makie-backed plotting and animation.

Loads automatically when the user runs `using CairoMakie` alongside
GraphEpimodels. Provides the rendering methods for the `LatticeVisualizer` /
`NetworkVisualizer` types and the `animate_*` entry points declared in the
package. Without CairoMakie loaded, those entry points raise a friendly error
(see src/visualization/visualization.jl).
"""
module GraphEpimodelsCairoMakieExt

using GraphEpimodels
using CairoMakie
import NetworkLayout

# Non-exported package internals reused here.
using GraphEpimodels: validate_visualizer_compatibility, _frame_title

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
# Lattice rendering
# =============================================================================

"""Create a clean figure + axis (equal aspect, no decorations)."""
function _make_axis(figure_size::Tuple{Int,Int}; title::String = "",
                    transparent_background::Bool = false)
    bg = transparent_background ? :transparent : :white
    fig = Figure(size = figure_size, backgroundcolor = bg)
    ax = Axis(fig[1, 1]; title = title, aspect = DataAspect(), backgroundcolor = bg)
    hidedecorations!(ax)
    hidespines!(ax)
    return fig, ax
end

"""Create a clean figure + axis (equal aspect, no decorations) for a lattice frame."""
function _lattice_axis(viz::LatticeVisualizer; title::String = "",
                       transparent_background::Bool = false)
    return _make_axis(viz.figure_size; title = title,
                      transparent_background = transparent_background)
end

"""Create a clean figure + 3D axis (equal aspect, no decorations) for node-link frames."""
function _make_axis3(figure_size::Tuple{Int,Int}; title::String = "",
                     transparent_background::Bool = false)
    bg = transparent_background ? :transparent : :white
    fig = Figure(size = figure_size, backgroundcolor = bg)
    ax = Axis3(fig[1, 1]; title = title, aspect = :data, backgroundcolor = bg)
    hidedecorations!(ax)
    return fig, ax
end

"""
Render a single lattice state (raw Int8 vector) to a Makie `Figure`.

Shared by `visualize_state` (live state) and the animation builder (recorded
frames), so animation frames look identical to static snapshots.
"""
function GraphEpimodels.render_frame(viz::LatticeVisualizer, lattice::AbstractLatticeGraph,
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
    width, height = lattice.dims[1], lattice.dims[2]
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
        # `image!` draws no cell borders, so honour `show_grid` with an explicit
        # overlay (the transparent `poly!` path above gets grid lines for free).
        if viz.show_grid
            vlines!(ax, 0.5:1:(width + 0.5); color = (:gray, 0.5), linewidth = 0.5)
            hlines!(ax, 0.5:1:(height + 0.5); color = (:gray, 0.5), linewidth = 0.5)
        end
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

function GraphEpimodels.visualize_state(viz::LatticeVisualizer, process::AbstractEpidemicProcess;
                                        transparent_background::Bool = false)
    validate_visualizer_compatibility(viz, process)
    graph = get_graph(process)
    return render_frame(viz, graph, node_states_raw(graph);
                        title = generate_visualization_title(process),
                        transparent_background = transparent_background)
end

"""
Save a static visualization of a process to file (format from the filename extension).

The visualizer is chosen by graph type via `visualizer_for`, so this works for
lattices (square / triangular / hexagonal, drawn as dual-tiling cells) and for
general / structured / random graphs (drawn as node-link diagrams) alike. The
lattice-only options (`transparent_background`, `show_boundary`, `show_grid`) are
ignored for node-link graphs.

# Arguments
- `process::AbstractEpidemicProcess`: Process to visualize
- `filename::String`: Output path (`.png`, `.pdf`, `.svg`)
- `color_scheme::Union{Symbol, Nothing}`: Color scheme; `nothing` (default) picks a
  model-appropriate scheme via `default_color_scheme`
- `figure_size::Tuple{Int, Int}`: Plot dimensions in pixels (default: (800, 800))
- `transparent_background::Bool`: Transparent background + susceptible cells (lattices only; default: false)
- `show_boundary::Bool`: Outline the lattice boundary (lattices only; default: false)
- `show_grid::Bool`: Stroke cell outlines (cell lattices only; default: false)
- `dim::Union{Int, Nothing}`: Drawing dimension, 2 or 3 (node-link graphs only).
  Default `nothing` picks the graph's natural dimension — 3D for graphs with a 3D
  layout (`CubeLattice`), 2D otherwise. `dim = 3` forces 3D — an intrinsic 3D layout
  where available (cube / star / complete / tree), else a 3D force-directed layout.
  Cell lattices (square / triangular / hexagonal) render in 2D only.
"""
function GraphEpimodels.save_plot(process::AbstractEpidemicProcess, filename::String;
                                          color_scheme::Union{Symbol, Nothing} = nothing,
                                          figure_size::Tuple{Int, Int} = (800, 800),
                                          transparent_background::Bool = false,
                                          show_boundary::Bool = false,
                                          show_grid::Bool = false,
                                          dim::Union{Int, Nothing} = nothing)
    scheme = color_scheme === nothing ? default_color_scheme(process) : color_scheme
    graph = get_graph(process)
    viz = _prepare_visualizer(graph, scheme, figure_size;
                              dim = dim, show_boundary = show_boundary, show_grid = show_grid)
    fig = viz isa LatticeVisualizer ?
        visualize_state(viz, process; transparent_background = transparent_background) :
        visualize_state(viz, process)
    save(filename, fig)
    println("Visualization saved to: $filename")
    return fig
end

# =============================================================================
# Network rendering
# =============================================================================

# Resolve the node-link drawing dimension: an explicit `dim` wins; `nothing` picks
# the graph's natural dimension — 3 when it has a 3D layout (CubeLattice), else 2.
# `layout_dim` is 3 for a 3D-native graph, 2 for a 2D one, and 0 when there's no
# intrinsic layout; clamping maps that 0 (spring-layout fallback) to 2.
_resolve_draw_dim(graph, dim::Union{Int, Nothing}) =
    something(dim, clamp(layout_dim(graph), 2, 3))

"""
Resolve and configure the auto-selected visualizer for rendering `graph`: pick a
`LatticeVisualizer` or `NetworkVisualizer` via [`visualizer_for`](@ref), reject
`dim ≠ 2` for cell lattices (which have no 3D layout), set the drawing dimension
for node-link graphs, and apply the lattice cell options. Shared by `save_plot`
and `animate_recording` so the visualizer/dimension resolution lives in one place.
"""
function _prepare_visualizer(graph, scheme::Symbol, figure_size::Tuple{Int, Int};
                             dim::Union{Int, Nothing}, show_boundary::Bool, show_grid::Bool)
    viz = visualizer_for(graph; color_scheme = scheme, figure_size = figure_size)
    if viz isa LatticeVisualizer
        something(dim, 2) == 2 || throw(ArgumentError(
            "$(typeof(graph)) renders in 2D only (got dim=$dim); cell lattices have no 3D layout"))
        viz.show_boundary = show_boundary
        viz.show_grid = show_grid
    else
        viz.dim = _resolve_draw_dim(graph, dim)
    end
    return viz
end

"""
Resolve node positions for a graph in `dim` dimensions (2 or 3). Returns a
`dim × n` `Matrix{Float64}`.

Resolution order:
1. If the graph has an intrinsic layout for `dim` (`dim ∈ supported_layout_dims`),
   use it (star sphere, tree shells, lattice grid, attached coordinates, …).
2. Else if it has an intrinsic layout of *higher* dimension, project onto the
   first `dim` axes (e.g. draw a graph with attached 3D coords in 2D).
3. Else fall back to a force-directed (spring) layout computed in `dim` dimensions.

Works for any `AbstractEpidemicGraph`: an `ErdosRenyiGraph` forwards the layout of
its wrapped graph, and a bare `AdjacencyGraph` with no coordinates falls back to a
spring layout.
"""
function _resolve_positions(graph::AbstractEpidemicGraph; dim::Int = 2)::Matrix{Float64}
    if dim in supported_layout_dims(graph)
        return node_positions(graph; dim = dim)
    end
    ld = layout_dim(graph)
    if ld > dim
        return node_positions(graph; dim = ld)[1:dim, :]   # project higher-dim layout
    end
    n = num_nodes(graph)
    adj = falses(n, n)
    @inbounds for i in 1:n, j in get_neighbors(graph, i)
        adj[i, j] = true
    end
    pts = NetworkLayout.spring(adj; dim = dim, seed = 1)
    mat = Matrix{Float64}(undef, dim, n)
    @inbounds for i in 1:n, d in 1:dim
        mat[d, i] = pts[i][d]
    end
    return mat
end

# Convert a `dim × n` coordinate matrix to a vector of Makie points, picking
# Point3f for a 3-row matrix and Point2f otherwise. Used for both nodes and edges
# so the 2D and 3D draw paths share one code path.
function _to_points(pos::AbstractMatrix{<:Real})
    n = size(pos, 2)
    if size(pos, 1) == 3
        return [Point3f(pos[1, i], pos[2, i], pos[3, i]) for i in 1:n]
    end
    return [Point2f(pos[1, i], pos[2, i]) for i in 1:n]
end

"""Create a node-link axis: 2D `Axis` (equal aspect) or 3D `Axis3` per `viz.dim`."""
function _network_axis(fig, viz::NetworkVisualizer; title::String = "")
    if viz.dim == 3
        ax = Axis3(fig[1, 1]; title = title, aspect = :data)
        hidedecorations!(ax)
        return ax
    end
    ax = Axis(fig[1, 1]; title = title, aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

"""
Draw a network frame into an existing axis (edges + colored node markers).

Works in 2D and 3D: the drawing dimension follows the coordinate matrix (2 or 3
rows), so the same code renders into an `Axis` or an `Axis3`.
"""
function _draw_network!(ax, viz::NetworkVisualizer, graph::AbstractEpidemicGraph,
                        states_raw::Vector{Int8};
                        positions::Union{Matrix{Float64}, Nothing} = nothing)
    pos = positions === nothing ? _resolve_positions(graph; dim = viz.dim) : positions
    pts = _to_points(pos)

    if viz.show_edges
        segs = eltype(pts)[]
        @inbounds for i in 1:num_nodes(graph), j in get_neighbors(graph, i)
            i < j || continue
            push!(segs, pts[i])
            push!(segs, pts[j])
        end
        isempty(segs) || linesegments!(ax, segs; color = viz.edge_color)
    end

    colors = _node_colors(viz.color_scheme, states_raw)
    scatter!(ax, pts; color = colors, markersize = viz.node_size,
             strokewidth = 0.5, strokecolor = :black)
    return ax
end

# =============================================================================
# Geographic basemap (drawn behind a GeoGraph's node-link diagram)
# =============================================================================
#
# A geographic graph carries node positions as raw [lon, lat]. Plotting those
# directly squashes the map, because 1° of longitude is shorter on the ground than
# 1° of latitude by a factor cos(lat). Rather than reproject (which would need a
# projection library and a basemap in matching coordinates), the axis box is given
# the aspect ratio (Δlon·cos lat₀)/Δlat and framed to the basemap's bbox, so lon/lat
# fill the box in correct proportion — the same trick whether drawing nodes, edges
# or the coastline, all of which live in one lon/lat space.

"""Axis-box aspect (width/height) that renders a lon/lat bbox in true proportion."""
function _geo_aspect(bbox::NTuple{4,Float64})::Float64
    lonmin, lonmax, latmin, latmax = bbox
    lat0 = deg2rad((latmin + latmax) / 2)
    dlon = max(lonmax - lonmin, eps())
    dlat = max(latmax - latmin, eps())
    return (dlon * cos(lat0)) / dlat
end

"""A clean (decoration-free) 2D axis with the geographic aspect for a basemap."""
function _geo_axis(fig, bm::GraphEpimodels.Basemap; title::String = "",
                   transparent_background::Bool = false)
    bg = transparent_background ? :transparent : :white
    ax = Axis(fig[1, 1]; title = title, aspect = _geo_aspect(bm.bbox), backgroundcolor = bg)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

"""Frame the axis to the basemap's geographic bbox so the whole map is visible."""
_apply_geo_limits!(ax, bm::GraphEpimodels.Basemap) =
    limits!(ax, bm.bbox[1], bm.bbox[2], bm.bbox[3], bm.bbox[4])

"""Convert a GeoJSON coordinate ring (`[[lon, lat], …]`) to `Point2f`s."""
function _ring_points(ring)::Vector{Point2f}
    pts = Vector{Point2f}(undef, length(ring))
    @inbounds for (i, p) in enumerate(ring)
        pts[i] = Point2f(Float64(p[1]), Float64(p[2]))
    end
    return pts
end

"""Append every polygon/line ring of one GeoJSON geometry to `rings`."""
function _collect_rings!(rings::Vector{Vector{Point2f}}, gtype::String, coords)
    coords === nothing && return
    if gtype == "Polygon"
        for ring in coords
            push!(rings, _ring_points(ring))
        end
    elseif gtype == "MultiPolygon"
        for poly in coords, ring in poly
            push!(rings, _ring_points(ring))
        end
    elseif gtype == "LineString"
        push!(rings, _ring_points(coords))
    elseif gtype == "MultiLineString"
        for line in coords
            push!(rings, _ring_points(line))
        end
    end
    return
end

"""
Parse the GeoJSON file behind a [`Basemap`](@ref) into drawable rings.

Reads with the package's dependency-free JSON reader and pulls coordinate rings
out of `Polygon` / `MultiPolygon` / `LineString` / `MultiLineString` geometries
(point geometries are ignored). Missing/unreadable file → no rings (the node-link
diagram still draws). Parse once and reuse across animation frames.
"""
function _load_basemap_rings(bm::GraphEpimodels.Basemap)::Vector{Vector{Point2f}}
    isfile(bm.path) || return Vector{Point2f}[]
    data = GraphEpimodels.parse_json(read(bm.path, String))
    rings = Vector{Point2f}[]
    feats = data isa Dict ? get(data, "features", nothing) : nothing
    if feats isa Vector
        for f in feats
            f isa Dict || continue
            geom = get(f, "geometry", nothing)
            geom isa Dict || continue
            _collect_rings!(rings, String(get(geom, "type", "")), get(geom, "coordinates", nothing))
        end
    elseif data isa Dict
        _collect_rings!(rings, String(get(data, "type", "")), get(data, "coordinates", nothing))
    end
    return rings
end

"""Draw basemap rings (filled land + coastline stroke) into an axis."""
function _draw_basemap!(ax, rings::Vector{Vector{Point2f}};
                        fill = (:gray, 0.08), stroke = (:gray, 0.55), strokewidth = 0.8)
    for ring in rings
        if length(ring) >= 3
            poly!(ax, ring; color = fill, strokecolor = stroke, strokewidth = strokewidth)
        elseif length(ring) >= 2
            lines!(ax, ring; color = stroke, linewidth = strokewidth)
        end
    end
    return ax
end

# A geographic graph is drawn as a node-link diagram on a map backdrop, but only
# in 2D (lon/lat has no meaningful 3D embedding here). Accepts any visualizer so
# the animation path can ask regardless of type — a LatticeVisualizer is never
# geographic.
_is_geographic(viz, graph) = viz isa NetworkVisualizer && viz.dim == 2 && has_basemap(graph)

function GraphEpimodels.render_frame(viz::NetworkVisualizer, graph::AbstractEpidemicGraph,
                                     states_raw::Vector{Int8}; title::String = "",
                                     positions::Union{Matrix{Float64}, Nothing} = nothing)
    fig = Figure(size = viz.figure_size)
    if _is_geographic(viz, graph)
        bm = basemap(graph)
        ax = _geo_axis(fig, bm; title = title)
        _draw_basemap!(ax, _load_basemap_rings(bm))
        _draw_network!(ax, viz, graph, states_raw; positions = positions)
        _apply_geo_limits!(ax, bm)
    else
        ax = _network_axis(fig, viz; title = title)
        _draw_network!(ax, viz, graph, states_raw; positions = positions)
    end
    return fig
end

function GraphEpimodels.visualize_state(viz::NetworkVisualizer, process::AbstractEpidemicProcess)
    validate_visualizer_compatibility(viz, process)
    graph = get_graph(process)
    return render_frame(viz, graph, node_states_raw(graph);
                        title = generate_visualization_title(process))
end

# =============================================================================
# Animation builder
# =============================================================================

# Draw one recorded frame into an existing axis, dispatched on visualizer type.
# Reuses the same drawing cores as the static `render_frame`, so animation frames
# look identical to static snapshots.
_draw_frame!(ax, viz::LatticeVisualizer, graph, states;
             positions = nothing, transparent_background::Bool = false) =
    _draw_lattice!(ax, viz, graph, states; transparent_background = transparent_background)
_draw_frame!(ax, viz::NetworkVisualizer, graph, states; positions = nothing,
             transparent_background::Bool = false) =
    _draw_network!(ax, viz, graph, states; positions = positions)

"""Fixed drawing limits (with margin) so the view doesn't jitter between frames."""
function _frame_limits(positions::Matrix{Float64})
    xlo, xhi = extrema(@view positions[1, :])
    ylo, yhi = extrema(@view positions[2, :])
    mx = 0.05 * (xhi - xlo) + 1.0
    my = 0.05 * (yhi - ylo) + 1.0
    return (xlo - mx, xhi + mx, ylo - my, yhi + my)
end

"""Apply fixed 3D limits (with margin) to an `Axis3` so the view stays steady."""
function _apply_limits3!(ax, positions::Matrix{Float64})
    xlo, xhi = extrema(@view positions[1, :])
    ylo, yhi = extrema(@view positions[2, :])
    zlo, zhi = extrema(@view positions[3, :])
    mx = 0.05 * (xhi - xlo) + 0.5
    my = 0.05 * (yhi - ylo) + 0.5
    mz = 0.05 * (zhi - zlo) + 0.5
    limits!(ax, xlo - mx, xhi + mx, ylo - my, yhi + my, zlo - mz, zhi + mz)
    return ax
end

"""
Render a recording to an animated GIF or MP4 (format from the filename extension).

Each frame is drawn with the same drawing core used by `visualize_state`, so
frames look identical to static snapshots. The visualizer is chosen by graph type
via `visualizer_for`, so this works for square / triangular / hexagonal lattices
and for general / structured / random graphs alike. Pass an explicit `visualizer`
to override that choice — e.g. a `NetworkVisualizer` to draw a lattice as a
node-link diagram.

# Arguments
- `rec::SimulationRecording`: The recording to animate
- `color_scheme::Union{Symbol, Nothing}`: Color scheme; `nothing` (default) picks a
  model-appropriate scheme from the recording's process name via `default_color_scheme`.
  Ignored when an explicit `visualizer` is supplied (the visualizer carries its own).
- `fps::Int`: Frames per second of the output (default: 15)
- `filename::String`: Output path; `.gif` or `.mp4` (default: "simulation.gif")
- `figure_size::Tuple{Int, Int}`: Frame size in pixels (default: (600, 600))
- `show_boundary::Bool`: Outline the lattice boundary (lattices only; default: false)
- `show_grid::Bool`: Stroke cell outlines (cell lattices only; default: false)
- `dim::Union{Int, Nothing}`: Drawing dimension, 2 or 3 (node-link graphs only).
  Default `nothing` picks the graph's natural dimension — 3D for graphs with a 3D
  layout (`CubeLattice`), 2D otherwise. `dim = 3` forces an intrinsic 3D layout where
  available (cube / star / complete / tree), else a 3D force-directed one. Ignored
  when an explicit `visualizer` is supplied (it carries its own `dim`). Cell lattices
  render in 2D only.
- `turntable::Bool`: For 3D animations, slowly rotate the camera one full turn over
  the clip (default: false). No effect in 2D.
- `visualizer::Union{AbstractVisualizer, Nothing}`: Override the auto-selected
  visualizer (default: `nothing` → chosen by graph type).

# Returns
- `String`: The output filename.
"""
function GraphEpimodels.animate_recording(rec::SimulationRecording;
                                          color_scheme::Union{Symbol, Nothing} = nothing,
                                          fps::Int = 15,
                                          filename::String = "simulation.gif",
                                          figure_size::Tuple{Int, Int} = (600, 600),
                                          show_boundary::Bool = false,
                                          show_grid::Bool = false,
                                          show_title::Bool = true,
                                          transparent_background::Bool = false,
                                          dim::Union{Int, Nothing} = nothing,
                                          turntable::Bool = false,
                                          visualizer::Union{AbstractVisualizer, Nothing} = nothing)
    graph = rec.graph
    if visualizer === nothing
        scheme = color_scheme === nothing ? default_color_scheme(rec.process_name) : color_scheme
        viz = _prepare_visualizer(graph, scheme, figure_size;
                                  dim = dim, show_boundary = show_boundary, show_grid = show_grid)
    else
        # An explicitly supplied visualizer carries its own dim; still honor the
        # call's lattice cell options.
        viz = visualizer
        if viz isa LatticeVisualizer
            viz.show_boundary = show_boundary
            viz.show_grid = show_grid
        end
    end

    is3d = viz isa NetworkVisualizer && viz.dim == 3
    geo  = _is_geographic(viz, graph)

    # A general (node-link) graph gets a single fixed layout so nodes don't move
    # between frames; lattices use their intrinsic node positions.
    positions = viz isa NetworkVisualizer ? _resolve_positions(graph; dim = viz.dim) :
                node_positions(graph)

    # Geographic graphs draw a map backdrop; parse its rings once and redraw them
    # each frame (the per-frame `empty!(ax)` clears everything, basemap included).
    bm = geo ? basemap(graph) : nothing
    basemap_rings = geo ? _load_basemap_rings(bm) : Vector{Vector{Point2f}}()

    title0 = show_title ? _frame_title(rec, 1) : ""
    if is3d
        fig, ax = _make_axis3(figure_size; title = title0,
                              transparent_background = transparent_background)
        _apply_limits3!(ax, positions)
        base_azimuth = ax.azimuth[]
    elseif geo
        fig = Figure(size = figure_size,
                     backgroundcolor = transparent_background ? :transparent : :white)
        ax = _geo_axis(fig, bm; title = title0, transparent_background = transparent_background)
        _apply_geo_limits!(ax, bm)
        base_azimuth = 0.0
    else
        fig, ax = _make_axis(figure_size; title = title0,
                             transparent_background = transparent_background)
        xlo, xhi, ylo, yhi = _frame_limits(positions)
        limits!(ax, xlo, xhi, ylo, yhi)
        base_azimuth = 0.0
    end

    layout_pos = viz isa NetworkVisualizer ? positions : nothing
    n = num_frames(rec)
    record(fig, filename, 1:n; framerate = fps) do idx
        empty!(ax)
        if show_title
            ax.title = _frame_title(rec, idx)
        end
        if is3d
            _apply_limits3!(ax, positions)   # keep limits steady across frames
            if turntable
                ax.azimuth[] = base_azimuth + 2π * (idx - 1) / n
            end
        elseif geo
            _draw_basemap!(ax, basemap_rings)   # redraw backdrop under the new frame
            _apply_geo_limits!(ax, bm)          # re-assert limits after empty!
        end
        _draw_frame!(ax, viz, graph, rec.frames[idx];
                     positions = layout_pos,
                     transparent_background = transparent_background)
    end

    println("Animation saved to: $filename ($n frames, $fps fps)")
    return filename
end

"""
Run a process and save an animated GIF of its evolution in one call.

Records the run with `record_simulation` and renders it with `animate_recording`.
Returns the `SimulationRecording` so the animation can be re-rendered at a
different fps / color scheme without re-simulating.

Runs from the process's *current* state (like `run_simulation`); pass a freshly
created `create_*_simulation(...)` process for a clean run.

# Arguments
See `record_simulation` (run/sampling) and `animate_recording` (rendering).

# Returns
- `SimulationRecording`
"""
function GraphEpimodels.animate_simulation(process::AbstractEpidemicProcess;
                                           sampler::FrameSampler = TimeInterval(1.0),
                                           max_time::Float64 = Inf,
                                           max_steps::Int = typemax(Int),
                                           stop_on_escape::Bool = false,
                                           color_scheme::Union{Symbol, Nothing} = nothing,
                                           fps::Int = 15,
                                           filename::String = "simulation.gif",
                                           figure_size::Tuple{Int, Int} = (600, 600),
                                           show_boundary::Bool = false,
                                           show_grid::Bool = false,
                                           show_title::Bool = true,
                                           transparent_background::Bool = false,
                                           dim::Union{Int, Nothing} = nothing,
                                           turntable::Bool = false,
                                           visualizer::Union{AbstractVisualizer, Nothing} = nothing)::SimulationRecording
    rec = record_simulation(process;
                            sampler = sampler,
                            max_time = max_time,
                            max_steps = max_steps,
                            stop_on_escape = stop_on_escape)

    animate_recording(rec;
                      color_scheme = color_scheme,
                      fps = fps,
                      filename = filename,
                      figure_size = figure_size,
                      show_boundary = show_boundary,
                      show_grid = show_grid,
                      show_title = show_title,
                      transparent_background = transparent_background,
                      dim = dim,
                      turntable = turntable,
                      visualizer = visualizer)

    return rec
end

end  # module GraphEpimodelsCairoMakieExt
