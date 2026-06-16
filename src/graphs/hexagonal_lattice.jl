"""
Hexagonal (honeycomb) lattice implementation for epidemic modeling.

A *hexagonal lattice* is the honeycomb graph in which every interior node has **3**
neighbors (the lattice whose faces are hexagons). Neighbors are computed by O(1)
coordinate arithmetic via the "brick-wall" representation of the honeycomb.

Coordinate / indexing convention (row-major):

    index = (row - 1) * width + col,   row ∈ 1:height,  col ∈ 1:width

Each node has two horizontal neighbors `(row, col±1)` plus exactly one vertical
neighbor whose direction is set by the parity of `row + col`:
- `row + col` even → neighbor below, `(row + 1, col)`
- `row + col` odd  → neighbor above, `(row - 1, col)`

This is provably 3-regular and symmetric (if `A` points down to `B`, then `B`
points up to `A`). Embedded with the positions in `node_positions`, the three
edges sit 120° apart and the faces are regular hexagons.

Its dual tiling is the **triangular** cell (3 sides, one per neighbor), pointing
up or down by parity; see `cell_polygons`.

Only `ABSORBING` boundaries are supported for now; `PERIODIC` throws.
"""

using Random

# =============================================================================
# Hexagonal Lattice
# =============================================================================

"""
Hexagonal/honeycomb lattice (3-neighbor) with absorbing boundaries.

# Fields
- `width::Int`: Number of columns (nodes per row)
- `height::Int`: Number of rows
- `n_nodes::Int`: Total node count (`width * height`)
- `boundary::BoundaryCondition`: Boundary condition (only `ABSORBING` supported)
- `states::Vector{Int8}`: Node states (primitive array)
- `boundary_nodes::Vector{Int}`: Pre-computed perimeter node indices
"""
mutable struct HexagonalLattice <: AbstractLatticeGraph
    width::Int
    height::Int
    n_nodes::Int
    boundary::BoundaryCondition
    states::Vector{Int8}
    boundary_nodes::Vector{Int}

    function HexagonalLattice(width::Int, height::Int,
                              boundary::BoundaryCondition = ABSORBING)
        if width < 1 || height < 1
            throw(ArgumentError("Lattice dimensions must be positive"))
        end
        if width > 50_000 || height > 50_000
            throw(ArgumentError("Lattice dimensions too large (>50,000)"))
        end
        if boundary != ABSORBING
            throw(ArgumentError("HexagonalLattice currently supports only :absorbing " *
                                "boundaries (periodic tiling deferred)"))
        end

        n_nodes = width * height
        states = zeros(Int8, n_nodes)
        boundary_nodes = _compute_perimeter_nodes(width, height,
                                                  (r, c) -> _hex_coord_to_index(r, c, width))

        new(width, height, n_nodes, boundary, states, boundary_nodes)
    end
end

# =============================================================================
# Coordinate Conversion (row-major)
# =============================================================================

@inline _hex_coord_to_index(row::Int, col::Int, width::Int)::Int =
    (row - 1) * width + col

@inline function _hex_index_to_coord(index::Int, width::Int)::Tuple{Int, Int}
    row, col = divrem(index - 1, width)
    return (row + 1, col + 1)
end

# =============================================================================
# Core Interface Implementation
# =============================================================================

@inline num_nodes(lattice::HexagonalLattice)::Int = lattice.n_nodes

node_states_raw(lattice::HexagonalLattice)::Vector{Int8} = lattice.states

function set_node_states_raw!(lattice::HexagonalLattice, states::Vector{Int8})
    if length(states) != num_nodes(lattice)
        throw(ArgumentError("Expected $(num_nodes(lattice)) states, got $(length(states))"))
    end
    lattice.states = states
end

function get_neighbors(lattice::HexagonalLattice, node_id::Int)::Vector{Int}
    return get_neighbors!(Int[], lattice, node_id)
end

function get_neighbors!(neighbors::Vector{Int}, lattice::HexagonalLattice, node_id::Int)::Vector{Int}
    if node_id < 1 || node_id > num_nodes(lattice)
        throw(BoundsError("Node ID $node_id out of range [1, $(num_nodes(lattice))]"))
    end

    row, col = _hex_index_to_coord(node_id, lattice.width)
    empty!(neighbors)

    # Two horizontal neighbors (same row).
    _add_hex_neighbor!(neighbors, row, col - 1, lattice)
    _add_hex_neighbor!(neighbors, row, col + 1, lattice)

    # One vertical neighbor, direction set by parity of (row + col).
    if iseven(row + col)
        _add_hex_neighbor!(neighbors, row + 1, col, lattice)  # down
    else
        _add_hex_neighbor!(neighbors, row - 1, col, lattice)  # up
    end

    return neighbors
end

get_boundary_nodes(lattice::HexagonalLattice)::Vector{Int} = copy(lattice.boundary_nodes)

"""Add neighbor (row, col) to the list if within the rectangular array bounds."""
@inline function _add_hex_neighbor!(neighbors::Vector{Int}, row::Int, col::Int,
                                    lattice::HexagonalLattice)
    if 1 <= row <= lattice.height && 1 <= col <= lattice.width
        push!(neighbors, _hex_coord_to_index(row, col, lattice.width))
    end
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# Honeycomb embedding (pointy-top, unit edge length): with
#   x = (√3/2) * col
#   y = 1.5 * row + 0.25 * (-1)^(row + col)
# the two horizontal and one vertical edge sit exactly 120° apart and every face
# is a regular hexagon. The dual cell is an equilateral triangle of circumradius
# 1 centered on the node, pointing down when (row + col) is even and up when odd.

supported_layout_dims(::HexagonalLattice)::Tuple{Vararg{Int}} = (2,)
has_cells(::HexagonalLattice)::Bool = true

@inline function _hex_node_xy(row::Int, col::Int)::Tuple{Float64, Float64}
    x = _SQRT3_2 * Float64(col)
    y = 1.5 * Float64(row) + 0.25 * (iseven(row + col) ? 1.0 : -1.0)
    return (x, y)
end

function node_positions(lattice::HexagonalLattice; dim::Int = 2)::Matrix{Float64}
    _check_layout_dim(lattice, dim)
    n = lattice.n_nodes
    pos = Matrix{Float64}(undef, 2, n)
    @inbounds for idx in 1:n
        row, col = _hex_index_to_coord(idx, lattice.width)
        x, y = _hex_node_xy(row, col)
        pos[1, idx] = x
        pos[2, idx] = y
    end
    return pos
end

function cell_polygons(lattice::HexagonalLattice)::Vector{Matrix{Float64}}
    n = lattice.n_nodes
    # Equilateral triangle, circumradius 1 (inradius 0.5 = half the unit edge, so
    # neighboring cells share an edge). Down-pointing for even (row+col), up for odd.
    down_angles = [deg2rad(30 + 120k) for k in 0:2]   # vertices at 30,150,270 -> apex down
    up_angles   = [deg2rad(90 + 120k) for k in 0:2]   # vertices at 90,210,330 -> apex up
    down_dx = [cos(a) for a in down_angles]; down_dy = [sin(a) for a in down_angles]
    up_dx   = [cos(a) for a in up_angles];   up_dy   = [sin(a) for a in up_angles]
    cells = Vector{Matrix{Float64}}(undef, n)
    @inbounds for idx in 1:n
        row, col = _hex_index_to_coord(idx, lattice.width)
        cx, cy = _hex_node_xy(row, col)
        if iseven(row + col)
            cells[idx] = [cx .+ down_dx'; cy .+ down_dy']
        else
            cells[idx] = [cx .+ up_dx'; cy .+ up_dy']
        end
    end
    return cells
end

# =============================================================================
# Factory Function
# =============================================================================

"""
Create a hexagonal/honeycomb lattice (3-neighbor) with the given boundary.

# Arguments
- `width::Int`: Number of columns
- `height::Int`: Number of rows
- `boundary::Symbol`: `:absorbing` (only option for now)

# Example
```julia
julia> lat = create_hexagonal_lattice(20, 20)
```
"""
function create_hexagonal_lattice(width::Int, height::Int,
                                  boundary::Symbol = :absorbing)::HexagonalLattice
    # Reuse the shared symbol→enum converter (graphs/hypercubic_lattice.jl); the
    # constructor rejects the resulting PERIODIC with a HexagonalLattice-specific
    # "periodic tiling deferred" message.
    return HexagonalLattice(width, height, _boundary_from_symbol(boundary))
end
