"""
Graph interface and general implementations for epidemic modeling.

This module defines the abstract graph interface that all graph types must implement,
along with efficient primitive state management and general graph implementations.
"""

using Random

# =============================================================================
# State Constants and Conversion (High Performance)
# =============================================================================

# Primitive state constants (Int8 for maximum performance)
const STATE_SUSCEPTIBLE = Int8(0)
const STATE_INFECTED = Int8(1)
const STATE_REMOVED = Int8(2)

# Public enum for type safety in external API
@enum NodeState::Int8 SUSCEPTIBLE=0 INFECTED=1 REMOVED=2

# Convenient aliases
const S = SUSCEPTIBLE
const I = INFECTED  
const R = REMOVED

# Zero-cost conversion functions (inlined by compiler)
@inline state_to_int(state::NodeState)::Int8 = Int8(state)
@inline int_to_state(val::Int8)::NodeState = NodeState(val)

# =============================================================================
# Abstract Graph Interface
# =============================================================================

"""
Abstract base type for all epidemic graph implementations.

All concrete graph types must implement the core interface methods.
This design allows epidemic processes to work with any graph structure
(lattices, networks, trees, etc.) through a common interface.
"""
abstract type AbstractEpidemicGraph end

"""
Abstract base type for *implicit* graphs, whose connectivity is a closed-form
function of the node index rather than stored data.

Implicit graphs compute a node's neighbors by arithmetic on demand and never
materialize an adjacency list, so their structure costs O(n) memory (just the
state vector) instead of O(n + m). This is the right category for any graph
family with a regular, parametrized topology — lattices (square, triangular,
hexagonal) and structured graphs like the complete graph.

This is a sibling of `AdjacencyGraph` (which stores explicit adjacency lists for
arbitrary topologies), not a supertype of it: implicit graphs deliberately avoid
materializing adjacency lists.
"""
abstract type AbstractImplicitGraph <: AbstractEpidemicGraph end

"""
Abstract base type for regular lattice graphs (square, triangular, hexagonal).

Lattices are [`AbstractImplicitGraph`](@ref)s whose implicit structure is a
*regular geometric arrangement*: neighbors are computed by O(1) coordinate
arithmetic, and they carry an intrinsic spatial layout (see the geometry
interface below) that the visualization layer draws as dual-tiling cells.
"""
abstract type AbstractLatticeGraph <: AbstractImplicitGraph end

"""
Pre-compute the perimeter node indices of a `width × height` rectangular array.

A node is on the perimeter if it sits in the first/last row or column. Shared by
lattice constructors for absorbing-boundary node lists; `coord_to_index(row,
col)` maps a 1-indexed coordinate to the lattice's linear index.
"""
# Half of √3, the vertical pitch shared by the triangular/hexagonal lattices.
const _SQRT3_2 = sqrt(3) / 2

function _compute_perimeter_nodes(width::Int, height::Int, coord_to_index)::Vector{Int}
    nodes = Int[]
    for row in 1:height, col in 1:width
        if row == 1 || row == height || col == 1 || col == width
            push!(nodes, coord_to_index(row, col))
        end
    end
    return nodes
end

# =============================================================================
# Core Interface Methods (REQUIRED - must be implemented by all graph types)
# =============================================================================

"""
Get the total number of nodes in the graph.

# Returns
- `Int`: Number of nodes
"""
function num_nodes(graph::AbstractEpidemicGraph)::Int
    error("num_nodes must be implemented by concrete graph type $(typeof(graph))")
end

"""
Get the neighbors of a specific node.

# Arguments
- `graph::AbstractEpidemicGraph`: The graph
- `node_id::Int`: Node identifier (1-indexed)

# Returns
- `Vector{Int}`: Vector of neighbor node IDs
"""
function get_neighbors(graph::AbstractEpidemicGraph, node_id::Int)::Vector{Int}
    error("get_neighbors must be implemented by concrete graph type $(typeof(graph))")
end

"""
Non-allocating variant of [`get_neighbors`](@ref).

Returns the neighbors of `node_id` as a vector that callers must treat as
**read-only**. Concrete graph types may either fill and return the supplied
`buffer` (so repeated calls reuse one allocation) or return an internally stored
neighbor vector directly. Either way the returned vector must not be mutated by
the caller.

This exists so performance-critical event loops (e.g. the ZIM step) can walk a
node's neighbors without allocating a fresh `Vector{Int}` on every step — a major
source of multithreaded GC pressure (issues #1 / #3).

The default implementation falls back to the allocating [`get_neighbors`](@ref),
so it is always correct; graph types override it where a cheaper path exists.

# Arguments
- `buffer::Vector{Int}`: Caller-owned scratch buffer that may be reused
- `graph::AbstractEpidemicGraph`: The graph
- `node_id::Int`: Node identifier (1-indexed)

# Returns
- `Vector{Int}`: Read-only neighbor list (possibly `buffer`, possibly internal)
"""
function get_neighbors!(buffer::Vector{Int}, graph::AbstractEpidemicGraph, node_id::Int)::Vector{Int}
    return get_neighbors(graph, node_id)
end

"""
Get the raw node states as primitive Int8 array (internal, performance-critical).

This is the internal representation used for all performance-critical operations.
Use node_states() for the external API with NodeState enums.

# Returns
- `Vector{Int8}`: Raw state array (STATE_SUSCEPTIBLE=0, STATE_INFECTED=1, STATE_REMOVED=2)
"""
function node_states_raw(graph::AbstractEpidemicGraph)::Vector{Int8}
    error("node_states_raw must be implemented by concrete graph type $(typeof(graph))")
end

"""
Set all node states using primitive Int8 array (internal, performance-critical).

# Arguments
- `states::Vector{Int8}`: Raw state array
"""
function set_node_states_raw!(graph::AbstractEpidemicGraph, states::Vector{Int8})
    error("set_node_states_raw! must be implemented by concrete graph type $(typeof(graph))")
end

# =============================================================================
# Optional Interface Methods (have default implementations)
# =============================================================================

"""
Get the degree (number of neighbors) of a node.

Default implementation uses get_neighbors(), but can be overridden for efficiency.

# Arguments
- `graph::AbstractEpidemicGraph`: The graph  
- `node_id::Int`: Node identifier

# Returns
- `Int`: Number of neighbors
"""
function get_node_degree(graph::AbstractEpidemicGraph, node_id::Int)::Int
    return length(get_neighbors(graph, node_id))
end

"""
Get boundary nodes for graphs with boundary concepts (e.g., lattices).

Default implementation returns empty vector (most graphs have no boundary).
Lattices and similar structures should override this method.

# Returns
- `Vector{Int}`: Vector of boundary node IDs
"""
function get_boundary_nodes(graph::AbstractEpidemicGraph)::Vector{Int}
    return Int[]
end

"""
Read-only, non-copying view of the boundary nodes for hot-path callers.

`get_boundary_nodes` copies defensively so callers may mutate the result, but
[`has_escaped`](@ref) only *reads* the list and runs on every Gillespie step
under `stop_on_escape` — the defensive copy was pure per-step allocation (a
boundary-sized `Vector{Int}` churned each step, the dominant GC cost of escape
analysis). This internal accessor returns the underlying vector directly and
must never be mutated. The default delegates to `get_boundary_nodes` so any type
is handled correctly; lattices override it to hand back their stored perimeter.
"""
_boundary_nodes_view(graph::AbstractEpidemicGraph)::Vector{Int} = get_boundary_nodes(graph)

"""
Check if the graph has a boundary concept.

# Returns
- `Bool`: true if graph has boundary nodes
"""
function has_boundary(graph::AbstractEpidemicGraph)::Bool
    return !isempty(get_boundary_nodes(graph))
end

# =============================================================================
# Shared AbstractImplicitGraph / AbstractLatticeGraph implementation
# =============================================================================
#
# Every implicit graph (the structured graphs and the lattices) follows one field
# convention: an `n_nodes::Int` node count and a `states::Vector{Int8}` state
# vector, with neighbors computed on demand by the type's own `get_neighbors!`.
# These shared methods supply the boilerplate that each such type would otherwise
# repeat verbatim, so a concrete type only implements what actually differs — its
# `get_neighbors!` rule and any optimized `count_neighbors_by_state` / geometry. A
# type that does not follow the convention just defines its own methods (as
# `AdjacencyGraph` does).

@inline num_nodes(graph::AbstractImplicitGraph)::Int = graph.n_nodes

node_states_raw(graph::AbstractImplicitGraph)::Vector{Int8} = graph.states

function set_node_states_raw!(graph::AbstractImplicitGraph, states::Vector{Int8})
    if length(states) != num_nodes(graph)
        throw(ArgumentError("Expected $(num_nodes(graph)) states, got $(length(states))"))
    end
    graph.states = states
end

# Allocating neighbor query: fill a fresh buffer via the type's `get_neighbors!`.
get_neighbors(graph::AbstractImplicitGraph, node_id::Int)::Vector{Int} =
    get_neighbors!(Int[], graph, node_id)

"""
Throw `BoundsError` unless `node_id` is a valid 1-indexed node of `graph`. Shared
guard for the on-demand neighbor / degree routines of implicit graphs.
"""
@inline function _check_node(graph::AbstractImplicitGraph, node_id::Int)
    n = num_nodes(graph)
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
    return nothing
end

# Lattices additionally precompute their perimeter (empty for periodic boundaries).
get_boundary_nodes(lattice::AbstractLatticeGraph)::Vector{Int} = copy(lattice.boundary_nodes)

# Hand back the stored perimeter without copying (read-only; see _boundary_nodes_view).
_boundary_nodes_view(lattice::AbstractLatticeGraph)::Vector{Int} = lattice.boundary_nodes

# =============================================================================
# Geometry Interface (Optional - consumed by the visualization layer)
# =============================================================================
#
# Topology (who connects to whom) is what the simulation engine needs and is
# covered by the interface above. Geometry (where each node sits in space) is
# needed only for plotting, so it lives in this separate, optional interface
# with safe defaults: a graph with no intrinsic layout (e.g. a bare
# AdjacencyGraph) needs to implement nothing, and the visualizer falls back to a
# computed layout.

"""
The intrinsic layout dimensions a graph type can produce, as a tuple of `Int`s.

This is the primary geometry-capability query: it lists every dimension `d` for
which the type knows a *closed-form* embedding, in preference order (so the first
entry is the "natural" one). Examples:

- `()`            — no intrinsic layout (e.g. a bare `AdjacencyGraph`); the
  visualizer falls back to a force-directed layout in whatever dimension it draws.
- `(2,)`          — a planar-only layout (lattices, cycle, path).
- `(2, 3)`        — both a 2D and a 3D closed-form layout (star, complete, tree).

A type advertises a dimension here *only* when it has a meaningful built-in
embedding for it. "Can it be drawn in 3D at all?" is a different (and almost
always `true`) question answered by the force-directed fallback, not by this.

Default: `()`. `has_layout` and `layout_dim` are derived from this.
"""
function supported_layout_dims(graph::AbstractEpidemicGraph)::Tuple{Vararg{Int}}
    return ()
end

"""
Whether the graph carries an intrinsic (or attached) spatial layout.

Derived from [`supported_layout_dims`](@ref): `true` iff the graph advertises at
least one intrinsic layout dimension. Lattices and the structured graphs return
`true`; an `AdjacencyGraph` returns `true` only when node coordinates have been
attached.
"""
has_layout(graph::AbstractEpidemicGraph)::Bool = !isempty(supported_layout_dims(graph))

"""
The *preferred* layout dimension (2 or 3); `0` when there is no layout.

Derived from [`supported_layout_dims`](@ref) as its first entry — the dimension
used when `node_positions` is called without an explicit `dim`. 2D graphs return
`2`; a graph whose natural embedding is 3D returns `3`.
"""
function layout_dim(graph::AbstractEpidemicGraph)::Int
    dims = supported_layout_dims(graph)
    return isempty(dims) ? 0 : first(dims)
end

"""
Validate that `dim` is one of the graph's [`supported_layout_dims`](@ref).

Concrete `node_positions` methods call this first so an unsupported `dim` gives a
clear error instead of silently wrong coordinates.
"""
@inline function _check_layout_dim(graph::AbstractEpidemicGraph, dim::Int)
    dim in supported_layout_dims(graph) && return nothing
    throw(ArgumentError(
        "$(typeof(graph)) supports layout dims $(supported_layout_dims(graph)); got dim=$dim"))
end

"""
Node coordinates as a `dim × N` matrix (column `i` is the position of node `i`).

`dim` selects which intrinsic layout to produce and must be one of the graph's
[`supported_layout_dims`](@ref); it defaults to the preferred [`layout_dim`](@ref).
Using `dim × N` (rather than `N × dim`) keeps each node's coordinates contiguous
in Julia's column-major storage, so a 2D vs 3D layout is just a different row
count (`2 × N` vs `3 × N`).

No default: graphs that report a non-empty `supported_layout_dims` must implement this.
"""
function node_positions(graph::AbstractEpidemicGraph;
                        dim::Int = layout_dim(graph))::Matrix{Float64}
    error("node_positions must be implemented by graph type $(typeof(graph)) " *
          "that reports a non-empty supported_layout_dims()")
end

"""
Evenly distributed points on the unit sphere via the golden-angle (Fibonacci)
spiral. Returns a `3 × n` matrix whose columns are unit vectors.

Shared by the closed-form 3D layouts (star sphere, complete-graph sphere, and the
per-level shells of a tree). The spiral spaces points near-uniformly for any `n`,
avoiding the clustering of naive lat/long grids.
"""
function _fibonacci_sphere(n::Int)::Matrix{Float64}
    pos = Matrix{Float64}(undef, 3, n)
    n == 0 && return pos
    golden = π * (3 - sqrt(5))            # golden angle ≈ 2.39996 rad
    @inbounds for i in 0:(n - 1)
        y = 1 - 2 * (i + 0.5) / n          # y from ~+1 down to ~-1
        r = sqrt(max(0.0, 1 - y * y))      # circle radius at height y
        θ = golden * i
        pos[1, i + 1] = r * cos(θ)
        pos[2, i + 1] = y
        pos[3, i + 1] = r * sin(θ)
    end
    return pos
end

"""
Evenly spaced points on the unit circle: returns a `2 × n` matrix whose column `i`
is `(cos θ_i, sin θ_i)` with `θ_i = 2π(i-1)/n`. The 2D analogue of
[`_fibonacci_sphere`](@ref), shared by the closed-form planar layouts (complete
graph, cycle, and a star's leaf ring).
"""
function _circle_layout(n::Int)::Matrix{Float64}
    pos = Matrix{Float64}(undef, 2, n)
    @inbounds for i in 1:n
        θ = 2π * (i - 1) / n
        pos[1, i] = cos(θ)
        pos[2, i] = sin(θ)
    end
    return pos
end

"""
Whether the graph fills space with one polygonal cell per node (a tiling).

Default: `false`. Regular lattices return `true` and supply `cell_polygons`;
general networks return `false` and are drawn as node-link diagrams instead.
"""
function has_cells(graph::AbstractEpidemicGraph)::Bool
    return false
end

"""
Per-node cell polygons for tiling visualization.

Returns one entry per node; each entry is a `2 × V` matrix whose columns are the
polygon vertices (in order) of that node's cell. The cell is the *dual* of the
lattice, so the number of vertices equals the node's degree and each cell edge
corresponds to one outgoing graph edge (square→square, triangular→hexagon,
hexagonal→triangle).

No default: graphs that report `has_cells() == true` must implement this.
"""
function cell_polygons(graph::AbstractEpidemicGraph)::Vector{Matrix{Float64}}
    error("cell_polygons must be implemented by graph type $(typeof(graph)) " *
          "that reports has_cells() == true")
end

"""
A geographic backdrop drawn behind a graph's node-link diagram.

A `Basemap` is a *handle*, not the image itself: it names a GeoJSON file (a
coastline / administrative outline, in `[lon, lat]` per the GeoJSON spec) and the
geographic `bbox` to frame. The actual coordinates are parsed and drawn lazily by
the visualization layer (the CairoMakie extension), so the simulation object stays
light and the heavy asset is touched only when something is actually rendered —
the same separation of concerns as `has_cells` / `cell_polygons` for lattices, and
as the optional Makie / CSV extensions.

# Fields
- `path::String`: Absolute path to the GeoJSON file.
- `bbox::NTuple{4,Float64}`: `(lon_min, lon_max, lat_min, lat_max)` to frame the map.
"""
struct Basemap
    path::String
    bbox::NTuple{4,Float64}
end

"""
Whether the graph carries a geographic [`Basemap`](@ref) to draw behind it.

Default: `false`. A [`GeoGraph`](@ref) returns `true` when a basemap was bundled
with its data. The visualizer consults this to decide whether to draw a map
backdrop (and to switch the axis to a geographic aspect) before the node-link
diagram.
"""
function has_basemap(graph::AbstractEpidemicGraph)::Bool
    return false
end

"""
The geographic [`Basemap`](@ref) for this graph, or `nothing` if it has none.

Default: `nothing`. Consumed by the visualization layer; see [`has_basemap`](@ref).
"""
function basemap(graph::AbstractEpidemicGraph)::Union{Basemap,Nothing}
    return nothing
end

# =============================================================================
# Public API (External Interface with Type Safety)
# =============================================================================

"""
Get node states as NodeState enums (external API).

This provides type safety for external code while using efficient 
primitive arrays internally.

# Returns
- `Vector{NodeState}`: Node states as enums
"""
function node_states(graph::AbstractEpidemicGraph)::Vector{NodeState}
    return NodeState.(node_states_raw(graph))
end

"""
Set node states using NodeState enums (external API).

# Arguments
- `states::Vector{NodeState}`: Node states as enums
"""
function set_node_states!(graph::AbstractEpidemicGraph, states::Vector{NodeState})
    if length(states) != num_nodes(graph)
        throw(ArgumentError("Expected $(num_nodes(graph)) states, got $(length(states))"))
    end
    set_node_states_raw!(graph, Int8.(states))
end

"""
Get the state of a specific node (external API).

# Arguments
- `node_id::Int`: Node identifier

# Returns
- `NodeState`: Node state as enum
"""
function get_node_state(graph::AbstractEpidemicGraph, node_id::Int)::NodeState
    states = node_states_raw(graph)
    if node_id < 1 || node_id > length(states)
        throw(BoundsError("Node ID $node_id out of range [1, $(length(states))]"))
    end
    return int_to_state(states[node_id])
end

"""
Set the state of a specific node (external API).

# Arguments  
- `node_id::Int`: Node identifier
- `state::NodeState`: New state
"""
function set_node_state!(graph::AbstractEpidemicGraph, node_id::Int, state::NodeState)
    states = node_states_raw(graph)
    if node_id < 1 || node_id > length(states)
        throw(BoundsError("Node ID $node_id out of range [1, $(length(states))]"))
    end
    states[node_id] = state_to_int(state)
end

# =============================================================================
# Derived Functions (High-Performance Implementations)
# =============================================================================

"""
Count nodes in each state using primitive arrays for maximum performance.

# Returns
- `Dict{NodeState, Int}`: Mapping from states to counts
"""
function count_states(graph::AbstractEpidemicGraph)::Dict{NodeState, Int}
    states = node_states_raw(graph)
    
    # Vectorized counting on primitive array (very fast)
    susceptible_count = count(==(STATE_SUSCEPTIBLE), states)
    infected_count = count(==(STATE_INFECTED), states)
    removed_count = count(==(STATE_REMOVED), states)
    
    return Dict{NodeState, Int}(
        SUSCEPTIBLE => susceptible_count,
        INFECTED => infected_count,
        REMOVED => removed_count
    )
end

"""
Get all nodes currently in a specific state using vectorized search.

# Arguments
- `state::NodeState`: Target state

# Returns
- `Vector{Int}`: Vector of node IDs in the specified state
"""
function get_nodes_in_state(graph::AbstractEpidemicGraph, state::NodeState)::Vector{Int}
    states = node_states_raw(graph)
    target_int = state_to_int(state)
    return findall(==(target_int), states)  # Vectorized search on Int8
end

"""
Count neighbors of a node in a specific state (performance-optimized).

# Arguments
- `node_id::Int`: Node to query
- `target_state::NodeState`: State to count

# Returns
- `Int`: Number of neighbors in target state
"""
function count_neighbors_by_state(graph::AbstractEpidemicGraph, node_id::Int, 
                                 target_state::NodeState)::Int
    neighbors = get_neighbors(graph, node_id)
    if isempty(neighbors)
        return 0
    end
    
    states = node_states_raw(graph)
    target_int = state_to_int(target_state)
    
    count = 0
    @inbounds for neighbor in neighbors
        if states[neighbor] == target_int
            count += 1
        end
    end
    return count
end

"""
Get active edges (connections between nodes that can interact).

For epidemic processes, this typically means infected-susceptible pairs.

# Arguments
- `from_state::NodeState`: Source state (default: INFECTED)
- `to_state::NodeState`: Target state (default: SUSCEPTIBLE)

# Returns
- `Vector{Tuple{Int, Int}}`: Vector of (from_node, to_node) pairs
"""
function get_active_edges(graph::AbstractEpidemicGraph, 
                         from_state::NodeState = INFECTED,
                         to_state::NodeState = SUSCEPTIBLE)::Vector{Tuple{Int, Int}}
    active_edges = Tuple{Int, Int}[]
    states = node_states_raw(graph)
    from_int = state_to_int(from_state)
    to_int = state_to_int(to_state)
    
    # Only check nodes in the from_state
    for from_node in 1:length(states)
        if states[from_node] == from_int
            neighbors = get_neighbors(graph, from_node)
            for neighbor in neighbors
                if states[neighbor] == to_int
                    push!(active_edges, (from_node, neighbor))
                end
            end
        end
    end
    
    return active_edges
end
