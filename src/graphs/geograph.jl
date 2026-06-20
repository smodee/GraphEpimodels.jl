"""
Geographic graph type — real-world settlements connected by multi-modal transport.

`GeoGraph` is the package's type for graphs that live on a map: nodes are
settlements (with names, populations and `[lon, lat]` coordinates) and edges come
in *labeled layers* — e.g. roads, railways, ferries, flights. The example data
shipped with the package is a country (`:norway_mock`), hence the "country graph" name
in the explorer UI, but the type itself is general (a region, a metro, …).

Like [`ErdosRenyiGraph`](@ref), `GeoGraph` is a thin wrapper: it stores an inner
[`AdjacencyGraph`](@ref) — built from the *union of the selected layers* — and
delegates the whole simulation interface to it. So once a `GeoGraph` is built,
every model runs on it exactly as on any other graph, at the same speed (the inner
graph's neighbor lists are a plain merged adjacency; the engine cannot tell it is
geographic).

The canonical layer data is kept alongside, so a different subset of layers can be
re-selected cheaply with [`with_layers`](@ref) without re-reading the file. The
node coordinates are geographic (`[lon, lat]`, advertised as a 2D layout); the
visualization layer corrects the longitude/latitude aspect when drawing.

Load the bundled example with [`load_geograph`](@ref); discover what is available
with [`available_country_graphs`](@ref) / [`country_edge_sets`](@ref).
"""

# =============================================================================
# Type
# =============================================================================

"""
A geographic graph: settlements (nodes) connected by labeled transport layers.

# Fields
- `graph::AdjacencyGraph`: Inner storage — the union of the active layers, with
  `[lon, lat]` coordinates attached. Provides the full simulation interface.
- `name::Symbol`: Stable identifier (e.g. `:norway_mock`), used for the data folder and
  for the persistence config string.
- `display_name::String`: Human-readable name (e.g. `"Norway"`).
- `node_names::Vector{String}`: Settlement name per node id.
- `population::Vector{Int}`: Settlement population per node id (`0` if unknown).
- `layers::Dict{Symbol,Vector{Tuple{Int,Int}}}`: All edge layers (the canonical,
  full data), keyed by layer symbol; each value is a list of undirected `(u, v)`
  edges.
- `layer_order::Vector{Symbol}`: Layers in their declared (UI) order.
- `layer_labels::Dict{Symbol,String}`: Human-readable label per layer.
- `active_layers::Vector{Symbol}`: Which layers are merged into `graph` (a subset
  of `layer_order`, kept in that order).
- `basemap::Union{Basemap,Nothing}`: Geographic backdrop handle, or `nothing`.

Construct via [`load_geograph`](@ref); re-select layers with [`with_layers`](@ref).
"""
struct GeoGraph <: AbstractEpidemicGraph
    graph::AdjacencyGraph
    name::Symbol
    display_name::String
    node_names::Vector{String}
    population::Vector{Int}
    layers::Dict{Symbol,Vector{Tuple{Int,Int}}}
    layer_order::Vector{Symbol}
    layer_labels::Dict{Symbol,String}
    active_layers::Vector{Symbol}
    basemap::Union{Basemap,Nothing}
end

function Base.show(io::IO, g::GeoGraph)
    active = join(string.(g.active_layers), "+")
    print(io, "GeoGraph(", g.display_name, ": n=", num_nodes(g),
          ", m=", num_edges(g), ", layers=[", active, "])")
end

# =============================================================================
# Interface — forwarded to the inner AdjacencyGraph
# =============================================================================
#
# The wrapper is immutable; state mutation flows through the (mutable) inner
# graph. Hot-path methods delegate, reusing AdjacencyGraph's optimized,
# zero-allocation implementations.

@inline num_nodes(g::GeoGraph)::Int = num_nodes(g.graph)
get_neighbors(g::GeoGraph, node_id::Int)::Vector{Int} = get_neighbors(g.graph, node_id)
get_neighbors!(buffer::Vector{Int}, g::GeoGraph, node_id::Int)::Vector{Int} =
    get_neighbors!(buffer, g.graph, node_id)
@inline get_node_degree(g::GeoGraph, node_id::Int)::Int = get_node_degree(g.graph, node_id)
node_states_raw(g::GeoGraph)::Vector{Int8} = node_states_raw(g.graph)
set_node_states_raw!(g::GeoGraph, states::Vector{Int8}) = set_node_states_raw!(g.graph, states)
get_boundary_nodes(g::GeoGraph)::Vector{Int} = get_boundary_nodes(g.graph)
count_neighbors_by_state(g::GeoGraph, node_id::Int, target_state::NodeState)::Int =
    count_neighbors_by_state(g.graph, node_id, target_state)

# Geometry: geographic coordinates are a 2D layout (delegated to the inner graph,
# which carries the [lon, lat] columns).
supported_layout_dims(g::GeoGraph)::Tuple{Vararg{Int}} = supported_layout_dims(g.graph)
node_positions(g::GeoGraph; dim::Int = layout_dim(g))::Matrix{Float64} =
    node_positions(g.graph; dim = dim)

# Basemap: a GeoGraph advertises its bundled backdrop (if any) so the visualizer
# draws a map behind the node-link diagram.
has_basemap(g::GeoGraph)::Bool = g.basemap !== nothing
basemap(g::GeoGraph)::Union{Basemap,Nothing} = g.basemap

# =============================================================================
# GeoGraph-specific accessors
# =============================================================================

"""Number of (undirected) edges currently in the active graph."""
num_edges(g::GeoGraph)::Int = sum(length, g.graph.adjacency_list) ÷ 2

"""Name of the settlement at node `id`."""
node_name(g::GeoGraph, id::Int)::String = g.node_names[id]

"""Population of the settlement at node `id` (`0` if unknown)."""
node_population(g::GeoGraph, id::Int)::Int = g.population[id]

"""All settlement names, indexed by node id."""
node_names(g::GeoGraph)::Vector{String} = g.node_names

"""Population of every settlement, indexed by node id."""
populations(g::GeoGraph)::Vector{Int} = g.population

"""The layers currently merged into the active graph, in declared order."""
active_layers(g::GeoGraph)::Vector{Symbol} = g.active_layers

"""Every layer available in the data (whether active or not), in declared order."""
available_layers(g::GeoGraph)::Vector{Symbol} = g.layer_order

"""Human-readable label for a layer symbol."""
layer_label(g::GeoGraph, layer::Symbol)::String = get(g.layer_labels, layer, string(layer))

"""
Find the node id of the settlement named `name` (case-insensitive), or `nothing`.

Handy for seeding a simulation from a named city, e.g.
`create_sir_process(g, β, γ; initial_infected = [find_node(g, "Oslo")])`.
"""
function find_node(g::GeoGraph, name::AbstractString)::Union{Int,Nothing}
    target = lowercase(strip(name))
    for (i, nm) in enumerate(g.node_names)
        lowercase(nm) == target && return i
    end
    return nothing
end

"""
The most populous settlement's node id — a sensible default "center" / seed for a
country graph (used by the explorer's *Center* initial condition).
"""
largest_settlement(g::GeoGraph)::Int = argmax(g.population)

# =============================================================================
# Layer selection
# =============================================================================

"""
Merge the chosen `layers` (a subset of `g.layer_order`) into one undirected
adjacency list over `n` nodes. Edges shared by several layers collapse to a single
edge (`create_graph_from_edges` de-duplicates), so the result is a plain simple
graph regardless of how many layers an edge appears in.
"""
function _merge_layers(layers_data::Dict{Symbol,Vector{Tuple{Int,Int}}},
                       selected::Vector{Symbol}, n::Int,
                       coords::Matrix{Float64})::AdjacencyGraph
    edges = Tuple{Int,Int}[]
    for layer in selected
        append!(edges, layers_data[layer])
    end
    return create_graph_from_edges(n, edges; coords = coords)
end

"""
Return a new `GeoGraph` with a different set of active edge layers.

Cheap: it re-merges from the in-memory layer data (no file I/O) and keeps the
nodes, coordinates, names, populations and basemap. Node states are reset to all
susceptible (a fresh graph). `layers` may be given as `Symbol`s or `String`s, or
`:all` for every available layer.

```julia
g  = load_geograph(:norway_mock)              # all layers
gr = with_layers(g, [:road, :rail])      # roads + railways only
```
"""
function with_layers(g::GeoGraph, layers)::GeoGraph
    selected = _normalize_layers(layers, g.layer_order)
    coords = node_positions(g.graph; dim = 2)
    inner = _merge_layers(g.layers, selected, num_nodes(g), coords)
    return GeoGraph(inner, g.name, g.display_name, g.node_names, g.population,
                    g.layers, g.layer_order, g.layer_labels, selected, g.basemap)
end

"""Resolve a layer selection (`:all`, or an iterable of Symbol/String) and validate it."""
function _normalize_layers(layers, available::Vector{Symbol})::Vector{Symbol}
    if layers === :all || layers == "all"
        return copy(available)
    end
    requested = Symbol[layers isa Union{Symbol,AbstractString} ? Symbol(layers) :
                       Symbol(x) for x in layers]
    for s in requested
        s in available || throw(ArgumentError(
            "Unknown edge layer :$s; available layers are $(available)"))
    end
    # Keep declared order, drop duplicates.
    return [s for s in available if s in requested]
end

# =============================================================================
# Loading from a bundle
# =============================================================================

"""Default directory holding the bundled country-graph data (`<repo>/data/countries`)."""
_countries_dir()::String = normpath(joinpath(@__DIR__, "..", "..", "data", "countries"))

"""Path to a country's bundle file, given its name and the containing directory."""
_bundle_path(name, dir::AbstractString)::String =
    joinpath(dir, string(name), "geograph.json")

"""
Load a geographic graph bundle by `name` (e.g. `:norway_mock`).

Reads the JSON bundle from `<dir>/<name>/geograph.json`, builds the node table and
all edge layers, and returns a [`GeoGraph`](@ref) whose active layers are `edges`.

# Arguments
- `name`: Bundle name — `Symbol` or `String` (the data subfolder name).
- `edges` (keyword): Which layers to activate — `:all` (default), or an iterable of
  layer symbols/strings (e.g. `[:road, :flight]`).
- `dir` (keyword): Directory containing the bundles (default: the package's
  `data/countries`).

# Returns
- `GeoGraph`

# Example
```julia
g  = load_geograph(:norway_mock)                       # all layers
gr = load_geograph(:norway_mock; edges = [:road, :rail])
sir = create_sir_process(g, 3.0, 1.0; initial_infected = [find_node(g, "Oslo")])
```
"""
function load_geograph(name::Union{Symbol,AbstractString};
                       edges = :all, dir::AbstractString = _countries_dir())::GeoGraph
    sym = Symbol(name)
    path = _bundle_path(sym, dir)
    isfile(path) || throw(ArgumentError(
        "No country-graph bundle for :$sym at $path. " *
        "Available: $(available_country_graphs(dir))"))
    bundle = parse_json(read(path, String))
    return _geograph_from_bundle(bundle, sym, dirname(path); edges = edges)
end

"""Build a `GeoGraph` from a parsed bundle dictionary (see the format spec)."""
function _geograph_from_bundle(bundle, sym::Symbol, bundle_dir::AbstractString; edges)::GeoGraph
    bundle isa Dict || throw(ArgumentError("Bundle root must be a JSON object"))

    display_name = String(get(bundle, "display_name", string(sym)))

    # --- layers (declared order + labels) ---
    raw_layers = get(bundle, "layers", nothing)
    raw_layers isa Vector || throw(ArgumentError(
        "Bundle 'layers' must be an array of [symbol, label] pairs"))
    layer_order = Symbol[]
    layer_labels = Dict{Symbol,String}()
    for entry in raw_layers
        (entry isa Vector && length(entry) == 2) || throw(ArgumentError(
            "Each 'layers' entry must be a [symbol, label] pair"))
        s = Symbol(String(entry[1]))
        push!(layer_order, s)
        layer_labels[s] = String(entry[2])
    end
    isempty(layer_order) && throw(ArgumentError("Bundle declares no edge layers"))

    # --- nodes ---
    raw_nodes = get(bundle, "nodes", nothing)
    raw_nodes isa Vector || throw(ArgumentError("Bundle 'nodes' must be an array"))
    n = length(raw_nodes)
    n >= 1 || throw(ArgumentError("Bundle must have at least one node"))

    names = Vector{String}(undef, n)
    pops = zeros(Int, n)
    coords = Matrix{Float64}(undef, 2, n)
    seen = falses(n)
    for node in raw_nodes
        node isa Dict || throw(ArgumentError("Each node must be a JSON object"))
        id = _as_int(node, "id")
        (1 <= id <= n) || throw(ArgumentError(
            "Node id $id out of range [1, $n]; ids must be contiguous 1-based"))
        seen[id] && throw(ArgumentError("Duplicate node id $id"))
        seen[id] = true
        names[id] = String(get(node, "name", "node $id"))
        pops[id] = haskey(node, "population") ? _as_int(node, "population") : 0
        coords[1, id] = _as_float(node, "lon")
        coords[2, id] = _as_float(node, "lat")
    end
    all(seen) || throw(ArgumentError("Node ids must cover 1:$n with no gaps"))

    # --- edges, grouped into layers ---
    raw_edges = get(bundle, "edges", nothing)
    raw_edges isa Vector || throw(ArgumentError("Bundle 'edges' must be an array"))
    layers = Dict{Symbol,Vector{Tuple{Int,Int}}}(s => Tuple{Int,Int}[] for s in layer_order)
    for edge in raw_edges
        edge isa Dict || throw(ArgumentError("Each edge must be a JSON object"))
        u = _as_int(edge, "u")
        v = _as_int(edge, "v")
        (1 <= u <= n && 1 <= v <= n) || throw(ArgumentError(
            "Edge ($u, $v) references a node id outside [1, $n]"))
        layer = Symbol(String(get(edge, "layer", "")))
        haskey(layers, layer) || throw(ArgumentError(
            "Edge ($u, $v) uses undeclared layer :$layer"))
        push!(layers[layer], (u, v))
    end

    # --- basemap (optional) ---
    bm = _basemap_from_bundle(bundle, bundle_dir)

    selected = _normalize_layers(edges, layer_order)
    inner = _merge_layers(layers, selected, n, coords)
    return GeoGraph(inner, sym, display_name, names, pops,
                    layers, layer_order, layer_labels, selected, bm)
end

"""Read the optional basemap handle from a bundle (a GeoJSON filename + bbox)."""
function _basemap_from_bundle(bundle::Dict, bundle_dir::AbstractString)::Union{Basemap,Nothing}
    file = get(bundle, "basemap", nothing)
    file === nothing && return nothing
    bbox_raw = get(bundle, "bbox", nothing)
    (bbox_raw isa Vector && length(bbox_raw) == 4) || throw(ArgumentError(
        "A bundle with a 'basemap' must also give a 'bbox' of [lon_min, lon_max, lat_min, lat_max]"))
    bbox = (Float64(bbox_raw[1]), Float64(bbox_raw[2]),
            Float64(bbox_raw[3]), Float64(bbox_raw[4]))
    return Basemap(joinpath(bundle_dir, String(file)), bbox)
end

"""Read a required numeric field as `Float64`, with a clear error if missing/wrong."""
function _as_float(d::Dict, key::String)::Float64
    haskey(d, key) || throw(ArgumentError("Missing required field '$key'"))
    v = d[key]
    v isa Number || throw(ArgumentError("Field '$key' must be a number, got $(typeof(v))"))
    return Float64(v)
end

"""Read a required field as `Int`, with a clear error if missing/non-integer."""
function _as_int(d::Dict, key::String)::Int
    haskey(d, key) || throw(ArgumentError("Missing required field '$key'"))
    v = d[key]
    if v isa Integer
        return Int(v)
    elseif v isa AbstractFloat && isinteger(v)
        return Int(v)
    else
        throw(ArgumentError("Field '$key' must be an integer, got $(repr(v))"))
    end
end

# =============================================================================
# Registry (discovery for the explorer UI)
# =============================================================================

"""
List the country-graph bundles available in `dir` (default: the package data dir).

Returns the bundle names (folder names containing a `geograph.json`) as sorted
`String`s — suitable for populating the explorer's country dropdown.
"""
function available_country_graphs(dir::AbstractString = _countries_dir())::Vector{String}
    isdir(dir) || return String[]
    names = String[]
    for entry in readdir(dir)
        isfile(joinpath(dir, entry, "geograph.json")) && push!(names, entry)
    end
    return sort!(names)
end

"""
The edge layers a country bundle offers, as `(symbol, label)` pairs in declared
order — what the explorer turns into one checkbox per layer.

Reads only the bundle's layer declaration (not its nodes/edges), so it is cheap to
call while building UI.
"""
function country_edge_sets(name::Union{Symbol,AbstractString};
                           dir::AbstractString = _countries_dir())::Vector{Tuple{Symbol,String}}
    path = _bundle_path(Symbol(name), dir)
    isfile(path) || throw(ArgumentError("No country-graph bundle for :$(Symbol(name)) at $path"))
    bundle = parse_json(read(path, String))
    raw_layers = get(bundle, "layers", nothing)
    raw_layers isa Vector || throw(ArgumentError("Bundle 'layers' must be an array"))
    return [(Symbol(String(e[1])), String(e[2])) for e in raw_layers]
end
