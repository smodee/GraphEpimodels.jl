"""
Triangular lattice implementation for epidemic modeling.

A *triangular lattice* is the regular grid in which every interior node has **6**
neighbors (the lattice whose faces are triangles). Neighbors are computed by O(1)
coordinate arithmetic, never stored, mirroring `SquareLattice`.

Coordinate / indexing convention (row-major, distinct from `SquareLattice`'s
column-major scheme — each lattice documents its own and only `get_neighbors` /
states are index-based, so the convention is internal):

    index = (row - 1) * width + col,   row ∈ 1:height,  col ∈ 1:width

Rows are stacked vertically and *even rows are shifted right* by half a cell, so
each interior node has two horizontal neighbors plus four diagonal neighbors
whose columns depend on row parity.

Its dual tiling is the **hexagonal** cell (6 sides, one per neighbor); see
`cell_polygons`.

Only `ABSORBING` boundaries are supported for now; `PERIODIC` throws (a triangular
torus only tiles cleanly for matched parities — deferred).
"""

using Random

# =============================================================================
# Triangular Lattice
# =============================================================================

"""
Triangular lattice (6-neighbor) with absorbing boundaries.

# Fields
- `width::Int`: Number of columns (nodes per row)
- `height::Int`: Number of rows
- `n_nodes::Int`: Total node count (`width * height`)
- `boundary::BoundaryCondition`: Boundary condition (only `ABSORBING` supported)
- `states::Vector{Int8}`: Node states (primitive array)
- `boundary_nodes::Vector{Int}`: Pre-computed perimeter node indices
"""
mutable struct TriangularLattice <: AbstractLatticeGraph
    width::Int
    height::Int
    n_nodes::Int
    boundary::BoundaryCondition
    states::Vector{Int8}
    boundary_nodes::Vector{Int}

    function TriangularLattice(width::Int, height::Int,
                               boundary::BoundaryCondition = ABSORBING)
        if width < 1 || height < 1
            throw(ArgumentError("Lattice dimensions must be positive"))
        end
        if width > 50_000 || height > 50_000
            throw(ArgumentError("Lattice dimensions too large (>50,000)"))
        end
        if boundary != ABSORBING
            throw(ArgumentError("TriangularLattice currently supports only :absorbing " *
                                "boundaries (periodic tiling deferred)"))
        end

        n_nodes = width * height
        states = zeros(Int8, n_nodes)
        boundary_nodes = _compute_perimeter_nodes(width, height,
                                                  (r, c) -> _tri_coord_to_index(r, c, width))

        new(width, height, n_nodes, boundary, states, boundary_nodes)
    end
end

# =============================================================================
# Coordinate Conversion (row-major)
# =============================================================================

@inline _tri_coord_to_index(row::Int, col::Int, width::Int)::Int =
    (row - 1) * width + col

@inline function _tri_index_to_coord(index::Int, width::Int)::Tuple{Int, Int}
    row, col = divrem(index - 1, width)
    return (row + 1, col + 1)
end

# =============================================================================
# Core Interface Implementation
# =============================================================================

function get_neighbors!(neighbors::Vector{Int}, lattice::TriangularLattice, node_id::Int)::Vector{Int}
    _check_node(lattice, node_id)

    row, col = _tri_index_to_coord(node_id, lattice.width)
    empty!(neighbors)

    # Two horizontal neighbors (same row).
    _add_perimeter_neighbor!(neighbors, row, col - 1, lattice)
    _add_perimeter_neighbor!(neighbors, row, col + 1, lattice)

    # Four diagonal neighbors; column offsets depend on row parity.
    # Even rows are shifted right, so they reach (col, col+1) above/below;
    # odd rows reach (col-1, col).
    if iseven(row)
        _add_perimeter_neighbor!(neighbors, row - 1, col,     lattice)  # up-left
        _add_perimeter_neighbor!(neighbors, row - 1, col + 1, lattice)  # up-right
        _add_perimeter_neighbor!(neighbors, row + 1, col,     lattice)  # down-left
        _add_perimeter_neighbor!(neighbors, row + 1, col + 1, lattice)  # down-right
    else
        _add_perimeter_neighbor!(neighbors, row - 1, col - 1, lattice)  # up-left
        _add_perimeter_neighbor!(neighbors, row - 1, col,     lattice)  # up-right
        _add_perimeter_neighbor!(neighbors, row + 1, col - 1, lattice)  # down-left
        _add_perimeter_neighbor!(neighbors, row + 1, col,     lattice)  # down-right
    end

    return neighbors
end

"""Add neighbor (row, col) to the list if within the rectangular array bounds."""
@inline function _add_perimeter_neighbor!(neighbors::Vector{Int}, row::Int, col::Int,
                                          lattice::TriangularLattice)
    if 1 <= row <= lattice.height && 1 <= col <= lattice.width
        push!(neighbors, _tri_coord_to_index(row, col, lattice.width))
    end
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# Even rows shifted right by 0.5; vertical pitch √3/2 makes all 6 neighbors
# equidistant at unit length. The dual cell is a regular hexagon centered on the
# node, one edge crossing each of the 6 incident edges.

supported_layout_dims(::TriangularLattice)::Tuple{Vararg{Int}} = (2,)
has_cells(::TriangularLattice)::Bool = true

@inline function _tri_node_xy(row::Int, col::Int)::Tuple{Float64, Float64}
    x = Float64(col) + (iseven(row) ? 0.5 : 0.0)
    y = Float64(row) * _SQRT3_2
    return (x, y)
end

function node_positions(lattice::TriangularLattice; dim::Int = 2)::Matrix{Float64}
    _check_layout_dim(lattice, dim)
    n = lattice.n_nodes
    pos = Matrix{Float64}(undef, 2, n)
    @inbounds for idx in 1:n
        row, col = _tri_index_to_coord(idx, lattice.width)
        x, y = _tri_node_xy(row, col)
        pos[1, idx] = x
        pos[2, idx] = y
    end
    return pos
end

function cell_polygons(lattice::TriangularLattice)::Vector{Matrix{Float64}}
    n = lattice.n_nodes
    # Regular hexagon (flat-top) circumradius so cells tile: vertex distance
    # 1/√3 from center places hexagon edges midway between unit-spaced nodes.
    r = 1 / sqrt(3)
    angles = [deg2rad(30 + 60k) for k in 0:5]
    dx = [r * cos(a) for a in angles]
    dy = [r * sin(a) for a in angles]
    cells = Vector{Matrix{Float64}}(undef, n)
    @inbounds for idx in 1:n
        row, col = _tri_index_to_coord(idx, lattice.width)
        cx, cy = _tri_node_xy(row, col)
        cells[idx] = [cx .+ dx'; cy .+ dy']
    end
    return cells
end

# =============================================================================
# Factory Function
# =============================================================================

"""
Create a triangular lattice (6-neighbor) with the given boundary.

# Arguments
- `width::Int`: Number of columns
- `height::Int`: Number of rows
- `boundary::Symbol`: `:absorbing` (only option for now)

# Example
```julia
julia> lat = create_triangular_lattice(20, 20)
```
"""
function create_triangular_lattice(width::Int, height::Int,
                                   boundary::Symbol = :absorbing)::TriangularLattice
    # Reuse the shared symbol→enum converter (graphs/hypercubic_lattice.jl); the
    # constructor rejects the resulting PERIODIC with a TriangularLattice-specific
    # "periodic tiling deferred" message.
    return TriangularLattice(width, height, _boundary_from_symbol(boundary))
end
