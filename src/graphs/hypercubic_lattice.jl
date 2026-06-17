"""
Hypercubic lattice in arbitrary dimension `D` (the nearest-neighbour graph on a
`d₁ × d₂ × … × d_D` box of ℤ^D).

A node has up to `2D` neighbours — `±1` along each axis — with the boundary
either *absorbing* (neighbours off the box are dropped) or *periodic* (the box
wraps into a torus). Connectivity is a closed-form function of the node index, so
this is an [`AbstractImplicitGraph`](@ref): neighbours are computed by coordinate
arithmetic on demand and never stored, costing O(n) memory (just the state
vector) instead of O(n·D).

`D` is a **type parameter**, not a runtime field, so the per-axis neighbour loop
has a statically known trip count and specializes/unrolls per dimension — the d=2
case compiles to the same shape as a hand-written N/S/E/W kernel.

# Dimension aliases
- [`SquareLattice`](@ref) `= HypercubicLattice{2}` — the 2D square lattice.
- [`CubeLattice`](@ref) `= HypercubicLattice{3}` — the 3D cubic lattice.

Only d=2 carries a space-filling cell tiling ([`has_cells`](@ref)); d=2 and d=3
carry a closed-form node layout for plotting (see the geometry section). Higher
dimensions have no intrinsic embedding and fall back to the interface defaults
(`supported_layout_dims() == ()`), which the visualization layer handles by
declining / using a computed layout.
"""

using Random

# =============================================================================
# Boundary Condition Types (shared by all lattice graphs)
# =============================================================================

@enum BoundaryCondition ABSORBING PERIODIC

# =============================================================================
# Index ↔ coordinate arithmetic (column-major, like Base's LinearIndices)
# =============================================================================
#
# Coordinates are 1-indexed `NTuple{D,Int}` in fast-to-slow axis order: axis 1
# varies fastest (stride 1). For the 2D alias this means dims = (width, height),
# coordinate = (col, row), and the linear index is `col + (row-1)*width` — which
# coincides exactly with the legacy `SquareLattice` index for square lattices.
#
# All three helpers recurse on the tuple (via `Base.tail`), so they are
# type-stable and unroll to straight-line code for the small `D` we ever use.

@inline _coord_tuple(::Int, ::Tuple{}) = ()
# Last axis: the remaining quotient *is* the coordinate, so no final divrem is
# needed (a D-dim decode costs D-1 divisions, matching the legacy 2D kernel's one).
@inline _coord_tuple(r::Int, ::Tuple{Int}) = (r + 1,)
@inline function _coord_tuple(r::Int, dims::NTuple{N,Int}) where {N}
    q, m = divrem(r, dims[1])
    return (m + 1, _coord_tuple(q, Base.tail(dims))...)
end

"""Linear index (1-based) → coordinate tuple (1-based), fast axis first."""
@inline _index_to_coord(index::Int, dims::NTuple{D,Int}) where {D} =
    _coord_tuple(index - 1, dims)

"""Cumulative strides for `dims`: `(1, d₁, d₁d₂, …)`."""
@inline _strides_from(::Int, ::Tuple{}) = ()
@inline _strides_from(acc::Int, dims::NTuple{N,Int}) where {N} =
    (acc, _strides_from(acc * dims[1], Base.tail(dims))...)
@inline _lattice_strides(dims::NTuple{D,Int}) where {D} = _strides_from(1, dims)

"""Coordinate tuple → linear index (1-based)."""
@inline function _coord_to_index(coord::NTuple{D,Int}, strides::NTuple{D,Int}) where {D}
    idx = 1
    @inbounds for k in 1:D
        idx += (coord[k] - 1) * strides[k]
    end
    return idx
end

# =============================================================================
# Hypercubic Lattice
# =============================================================================

"""
Hypercubic lattice on a `dims[1] × … × dims[D]` box of ℤ^D.

Stores only the box shape, the precomputed axis strides, the primitive `Int8`
state vector, and (for absorbing boundaries) the precomputed perimeter node list.
Neighbours are computed on demand.

# Fields
- `dims::NTuple{D,Int}`: side lengths, fast axis first (2D: `(width, height)`)
- `n_nodes::Int`: total nodes, `prod(dims)`
- `boundary::BoundaryCondition`: `ABSORBING` or `PERIODIC`
- `states::Vector{Int8}`: node states (primitive array)
- `strides::NTuple{D,Int}`: column-major strides, `(1, dims[1], …)`
- `boundary_nodes::Vector{Int}`: perimeter nodes (empty for periodic)
"""
mutable struct HypercubicLattice{D} <: AbstractLatticeGraph
    dims::NTuple{D,Int}
    n_nodes::Int
    boundary::BoundaryCondition
    states::Vector{Int8}
    strides::NTuple{D,Int}
    boundary_nodes::Vector{Int}

    function HypercubicLattice(dims::NTuple{D,Int},
                               boundary::BoundaryCondition = ABSORBING) where {D}
        if D < 1
            throw(ArgumentError("Lattice needs at least one dimension"))
        end
        if any(<(1), dims)
            throw(ArgumentError("Lattice side lengths must be positive, got $dims"))
        end
        if any(>(50_000), dims)
            throw(ArgumentError("Lattice side length too large (>50,000): $dims"))
        end

        n_nodes = prod(dims)
        strides = _lattice_strides(dims)
        states  = zeros(Int8, n_nodes)  # All start SUSCEPTIBLE
        boundary_nodes = boundary == ABSORBING ? _compute_boundary_nodes(dims) : Int[]

        new{D}(dims, n_nodes, boundary, states, strides, boundary_nodes)
    end
end

# =============================================================================
# Core Interface Implementation (Required Methods)
# =============================================================================

# `get_neighbors!` and `count_neighbors_by_state` are `@generated` so the per-axis
# work is emitted as `D` straight-line statements (`Base.Cartesian.@nexprs`) with
# literal tuple indices, rather than a runtime `for k in 1:D` loop. For d=2 this
# compiles to the same shape as the original hand-unrolled N/S/E/W kernel — a
# plain `for` loop over the dimension regressed the count hot path ~1.5×.
@generated function get_neighbors!(neighbors::Vector{Int}, lattice::HypercubicLattice{D},
                                   node_id::Int)::Vector{Int} where {D}
    quote
        if node_id < 1 || node_id > lattice.n_nodes
            throw(BoundsError("Node ID $node_id out of range [1, $(lattice.n_nodes)]"))
        end
        coord = _index_to_coord(node_id, lattice.dims)
        dims, s = lattice.dims, lattice.strides
        empty!(neighbors)
        @inbounds if lattice.boundary == ABSORBING
            Base.Cartesian.@nexprs $D k -> begin
                coord[k] > 1       && push!(neighbors, node_id - s[k])   # back along axis k
                coord[k] < dims[k] && push!(neighbors, node_id + s[k])   # forward along axis k
            end
        else  # PERIODIC: wrap each axis
            Base.Cartesian.@nexprs $D k -> begin
                wrap_k = (dims[k] - 1) * s[k]
                push!(neighbors, coord[k] > 1       ? node_id - s[k] : node_id + wrap_k)
                push!(neighbors, coord[k] < dims[k] ? node_id + s[k] : node_id - wrap_k)
            end
        end
        return neighbors
    end
end

# =============================================================================
# Performance-Optimized Neighbor Counting (simulation hot path)
# =============================================================================

"""
Count a node's neighbours in `target_state` without materializing a neighbour
list. Mirrors [`get_neighbors!`](@ref): walk the up-to-`2D` axis neighbours,
testing the contiguous `Int8` state array directly. This is the inner loop of the
SIR/ZIM steppers.
"""
@generated function count_neighbors_by_state(lattice::HypercubicLattice{D}, node_id::Int,
                                             target_state::NodeState)::Int where {D}
    quote
        if node_id < 1 || node_id > lattice.n_nodes
            throw(BoundsError("Node ID $node_id out of range [1, $(lattice.n_nodes)]"))
        end
        coord = _index_to_coord(node_id, lattice.dims)
        states = lattice.states
        dims, s = lattice.dims, lattice.strides
        target_int = state_to_int(target_state)
        cnt = 0
        @inbounds if lattice.boundary == ABSORBING
            Base.Cartesian.@nexprs $D k -> begin
                if coord[k] > 1 && states[node_id - s[k]] == target_int
                    cnt += 1
                end
                if coord[k] < dims[k] && states[node_id + s[k]] == target_int
                    cnt += 1
                end
            end
        else  # PERIODIC
            Base.Cartesian.@nexprs $D k -> begin
                wrap_k = (dims[k] - 1) * s[k]
                back_k = coord[k] > 1       ? node_id - s[k] : node_id + wrap_k
                fwd_k  = coord[k] < dims[k] ? node_id + s[k] : node_id - wrap_k
                cnt += (states[back_k] == target_int) + (states[fwd_k] == target_int)
            end
        end
        return cnt
    end
end

# =============================================================================
# Boundary Computation (Internal)
# =============================================================================

"""Perimeter node indices: any node with a coordinate at `1` or `dims[k]`."""
function _compute_boundary_nodes(dims::NTuple{D,Int}) where {D}
    nodes = Int[]
    n = prod(dims)
    for idx in 1:n
        coord = _index_to_coord(idx, dims)
        on_boundary = false
        @inbounds for k in 1:D
            if coord[k] == 1 || coord[k] == dims[k]
                on_boundary = true
                break
            end
        end
        on_boundary && push!(nodes, idx)
    end
    return nodes
end

# =============================================================================
# Coordinate Conversion (public)
# =============================================================================

"""
Convert a `D`-dimensional coordinate (fast axis first) to its linear index.

For the 2D alias, prefer the `(row, col)` convenience method below.
"""
function coord_to_index(lattice::HypercubicLattice{D}, coord::NTuple{D,Int})::Int where {D}
    @inbounds for k in 1:D
        if coord[k] < 1 || coord[k] > lattice.dims[k]
            throw(BoundsError("Coordinate $coord out of bounds for dims $(lattice.dims)"))
        end
    end
    return _coord_to_index(coord, lattice.strides)
end

"""Linear index → `D`-dimensional coordinate tuple (fast axis first)."""
function index_to_coord(lattice::HypercubicLattice{D}, index::Int)::NTuple{D,Int} where {D}
    if index < 1 || index > num_nodes(lattice)
        throw(BoundsError("Index $index out of bounds"))
    end
    return _index_to_coord(index, lattice.dims)
end

# 2D convenience overloads preserving the legacy `(row, col)` API. dims =
# (width, height), so the internal coordinate is (col, row): we just swap.
coord_to_index(lattice::HypercubicLattice{2}, row::Int, col::Int)::Int =
    coord_to_index(lattice, (col, row))

function index_to_coord(lattice::HypercubicLattice{2}, index::Int)::Tuple{Int,Int}
    x, y = _index_to_coord(index, lattice.dims)  # (col, row)
    return (y, x)                                 # (row, col), legacy order
end

# =============================================================================
# Lattice Utility Functions
# =============================================================================

"""Center node (rounded-down midpoint of each axis) — useful for seeding."""
function get_center_node(lattice::HypercubicLattice{D})::Int where {D}
    center = ntuple(k -> (lattice.dims[k] + 1) ÷ 2, Val(D))
    return _coord_to_index(center, lattice.strides)
end

"""Sample `count` random node indices (with replacement)."""
function get_random_nodes(lattice::HypercubicLattice, count::Int,
                          rng::AbstractRNG = Random.default_rng())::Vector{Int}
    if count > num_nodes(lattice)
        throw(ArgumentError("Cannot sample $count nodes from $(num_nodes(lattice)) total"))
    end
    return rand(rng, 1:num_nodes(lattice), count)
end

"""
Chebyshev-style distance from a node to the nearest absorbing boundary face
(`Inf` for periodic lattices). It is the minimum over axes of the distance to
either end of that axis.
"""
function distance_to_boundary(lattice::HypercubicLattice{D}, node_id::Int)::Float64 where {D}
    lattice.boundary == PERIODIC && return Inf
    coord = index_to_coord_internal(lattice, node_id)
    d = typemax(Int)
    @inbounds for k in 1:D
        d = min(d, coord[k] - 1, lattice.dims[k] - coord[k])
    end
    return Float64(d)
end

# Internal (un-swapped) coordinate accessor with bounds check, for D-generic use.
@inline function index_to_coord_internal(lattice::HypercubicLattice, node_id::Int)
    if node_id < 1 || node_id > num_nodes(lattice)
        throw(BoundsError("Index $node_id out of bounds"))
    end
    return _index_to_coord(node_id, lattice.dims)
end

"""Whether `node_id` lies on an absorbing boundary face (always false for periodic)."""
function is_boundary_node(lattice::HypercubicLattice, node_id::Int)::Bool
    lattice.boundary == PERIODIC && return false
    return node_id in lattice.boundary_nodes
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# Only low dimensions have a meaningful built-in embedding:
#   • d=2 — node at (x, y) = (col, row); dual cell is the unit square (the square
#     tiling is self-dual, so its 4 edges match the 4 neighbours).
#   • d=3 — node at (x, y, z); no cell tiling (cubes don't draw usefully as a 2D
#     dual), so positions only.
# d≥4 advertises no layout and inherits the interface defaults.

supported_layout_dims(::HypercubicLattice{2})::Tuple{Vararg{Int}} = (2,)
supported_layout_dims(::HypercubicLattice{3})::Tuple{Vararg{Int}} = (3,)

has_cells(::HypercubicLattice{2})::Bool = true

function node_positions(lattice::HypercubicLattice{2}; dim::Int = 2)::Matrix{Float64}
    _check_layout_dim(lattice, dim)
    n = lattice.n_nodes
    pos = Matrix{Float64}(undef, 2, n)
    @inbounds for idx in 1:n
        x, y = _index_to_coord(idx, lattice.dims)  # (col, row)
        pos[1, idx] = Float64(x)
        pos[2, idx] = Float64(y)
    end
    return pos
end

function node_positions(lattice::HypercubicLattice{3}; dim::Int = 3)::Matrix{Float64}
    _check_layout_dim(lattice, dim)
    n = lattice.n_nodes
    pos = Matrix{Float64}(undef, 3, n)
    @inbounds for idx in 1:n
        x, y, z = _index_to_coord(idx, lattice.dims)
        pos[1, idx] = Float64(x)
        pos[2, idx] = Float64(y)
        pos[3, idx] = Float64(z)
    end
    return pos
end

function cell_polygons(lattice::HypercubicLattice{2})::Vector{Matrix{Float64}}
    n = lattice.n_nodes
    cells = Vector{Matrix{Float64}}(undef, n)
    @inbounds for idx in 1:n
        x, y = _index_to_coord(idx, lattice.dims)
        xf = Float64(x); yf = Float64(y)
        # Unit square (counter-clockwise) centered on the node.
        cells[idx] = [xf-0.5 xf+0.5 xf+0.5 xf-0.5;
                      yf-0.5 yf-0.5 yf+0.5 yf+0.5]
    end
    return cells
end

# =============================================================================
# Dimension Aliases
# =============================================================================

"""2D square lattice — `HypercubicLattice{2}` with `dims = (width, height)`."""
const SquareLattice = HypercubicLattice{2}

"""3D cubic lattice — `HypercubicLattice{3}` with `dims = (width, height, depth)`."""
const CubeLattice = HypercubicLattice{3}

# =============================================================================
# Factory Functions
# =============================================================================

_boundary_from_symbol(boundary::Symbol)::BoundaryCondition =
    boundary == :absorbing ? ABSORBING :
    boundary == :periodic  ? PERIODIC  :
    throw(ArgumentError("Unknown boundary type: $boundary. Use :absorbing or :periodic"))

"""
Create a hypercubic lattice from a tuple of side lengths.

# Arguments
- `dims::NTuple{D,Int}`: side lengths (fast axis first)
- `boundary::Symbol`: `:absorbing` (default) or `:periodic`

# Example
```julia
julia> create_hypercubic_lattice((10, 10, 10, 10))   # 4D, 10⁴ nodes
julia> create_hypercubic_lattice((50, 50); boundary = :periodic)
```
"""
function create_hypercubic_lattice(dims::NTuple{D,Int};
                                   boundary::Symbol = :absorbing)::HypercubicLattice{D} where {D}
    return HypercubicLattice(dims, _boundary_from_symbol(boundary))
end

# Vararg convenience: create_hypercubic_lattice(10, 10, 10)
create_hypercubic_lattice(dims::Int...; boundary::Symbol = :absorbing) =
    create_hypercubic_lattice(dims; boundary = boundary)

"""
Create a 2D square lattice with the given boundary condition.

# Arguments
- `width::Int`, `height::Int`: lattice dimensions
- `boundary::Symbol`: `:absorbing` (default) or `:periodic`

# Example
```julia
julia> lattice = create_square_lattice(100, 100, :absorbing)
julia> center = get_center_node(lattice)
```
"""
function create_square_lattice(width::Int, height::Int,
                               boundary::Symbol = :absorbing)::SquareLattice
    return HypercubicLattice((width, height), _boundary_from_symbol(boundary))
end

"""
Create a 3D cubic lattice with the given boundary condition.

# Arguments
- `width::Int`, `height::Int`, `depth::Int`: lattice dimensions
- `boundary::Symbol`: `:absorbing` (default) or `:periodic`
"""
function create_cube_lattice(width::Int, height::Int, depth::Int,
                             boundary::Symbol = :absorbing)::CubeLattice
    return HypercubicLattice((width, height, depth), _boundary_from_symbol(boundary))
end

"""Create a square torus (2D periodic lattice) of side `size`."""
create_torus(size::Int)::SquareLattice = create_square_lattice(size, size, :periodic)
