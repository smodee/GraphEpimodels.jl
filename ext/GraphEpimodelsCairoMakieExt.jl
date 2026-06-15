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
"""
function GraphEpimodels.save_plot(process::AbstractEpidemicProcess, filename::String;
                                          color_scheme::Union{Symbol, Nothing} = nothing,
                                          figure_size::Tuple{Int, Int} = (800, 800),
                                          transparent_background::Bool = false,
                                          show_boundary::Bool = false,
                                          show_grid::Bool = false)
    scheme = color_scheme === nothing ? default_color_scheme(process) : color_scheme
    graph = get_graph(process)
    viz = visualizer_for(graph; color_scheme = scheme, figure_size = figure_size)
    if viz isa LatticeVisualizer
        viz.show_boundary = show_boundary
        viz.show_grid = show_grid
        fig = visualize_state(viz, process; transparent_background = transparent_background)
    else
        fig = visualize_state(viz, process)
    end
    save(filename, fig)
    println("Visualization saved to: $filename")
    return fig
end

# =============================================================================
# Network rendering
# =============================================================================

"""
Resolve node positions for a graph: attached coordinates if present, else a
force-directed (spring) layout. Returns a `2 × n` `Matrix{Float64}`.

Works for any `AbstractEpidemicGraph`: structured implicit graphs and lattices
carry an intrinsic layout (`has_layout`), an `ErdosRenyiGraph` forwards the layout
of its wrapped graph, and a bare `AdjacencyGraph` with no coordinates falls back
to a spring layout.
"""
function _resolve_positions(graph::AbstractEpidemicGraph)::Matrix{Float64}
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

"""Draw a network frame into an existing axis (edges + colored node markers)."""
function _draw_network!(ax, viz::NetworkVisualizer, graph::AbstractEpidemicGraph,
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

function GraphEpimodels.render_frame(viz::NetworkVisualizer, graph::AbstractEpidemicGraph,
                                     states_raw::Vector{Int8}; title::String = "",
                                     positions::Union{Matrix{Float64}, Nothing} = nothing)
    fig = Figure(size = viz.figure_size)
    ax = Axis(fig[1, 1]; title = title, aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    _draw_network!(ax, viz, graph, states_raw; positions = positions)
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
                                          visualizer::Union{AbstractVisualizer, Nothing} = nothing)
    graph = rec.graph
    if visualizer === nothing
        scheme = color_scheme === nothing ? default_color_scheme(rec.process_name) : color_scheme
        viz = visualizer_for(graph; color_scheme = scheme, figure_size = figure_size)
    else
        viz = visualizer
    end
    if viz isa LatticeVisualizer
        viz.show_boundary = show_boundary
        viz.show_grid = show_grid
    end

    # A general (node-link) graph gets a single fixed layout so nodes don't move
    # between frames; lattices use their intrinsic node positions.
    positions = viz isa NetworkVisualizer ? _resolve_positions(graph) :
                node_positions(graph)
    xlo, xhi, ylo, yhi = _frame_limits(positions)

    title0 = show_title ? _frame_title(rec, 1) : ""
    fig, ax = _make_axis(figure_size; title = title0,
                         transparent_background = transparent_background)
    limits!(ax, xlo, xhi, ylo, yhi)

    layout_pos = viz isa NetworkVisualizer ? positions : nothing
    n = num_frames(rec)
    record(fig, filename, 1:n; framerate = fps) do idx
        empty!(ax)
        if show_title
            ax.title = _frame_title(rec, idx)
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
                      visualizer = visualizer)

    return rec
end

end  # module GraphEpimodelsCairoMakieExt
