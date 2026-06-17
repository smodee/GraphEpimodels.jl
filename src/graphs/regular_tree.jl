"""
Regular rooted tree implementations for epidemic modeling.

This file provides a single [`RegularTree`](@ref) type that covers two common
rooted-tree conventions, distinguished only by how many children non-root
internal nodes have:

- **Graph-theory regular tree** (Cayley tree / Bethe lattice), via
  [`create_regular_tree`](@ref): *every* internal vertex has degree `d`, so the
  root has `d` children and each non-root internal node has `d − 1` children (one
  edge goes to its parent). This is the canonical regular tree in
  percolation/branching-process analysis — the branching ratio `d − 1` is exactly
  what governs the epidemic threshold — and is the default meaning of "regular
  tree".
- **Balanced d-ary tree** (the computer-science convention), via
  [`create_dary_tree`](@ref): *every* internal node, root included, has `k`
  children (so non-root internal nodes have degree `k + 1`).

Node numbering is 1-indexed BFS (level) order in both cases. Because every level
is complete and every internal node at a given "tier" has the same number of
children, parent/child indices reduce to O(1) arithmetic (see the helpers below).

Like [`StarGraph`](@ref) and [`PathGraph`](@ref), this is an
[`AbstractImplicitGraph`](@ref): only the state vector is stored; no adjacency
lists are materialized.
"""

# =============================================================================
# Regular Rooted Tree
# =============================================================================

"""
Regular rooted tree with `height` levels, parametrized by two child counts.

The root (node 1) has `root_children` children; every *non-root* internal node
has `branching` children. The two public constructors set these:
[`create_regular_tree`](@ref) (Cayley: `root_children = d`, `branching = d − 1`)
and [`create_dary_tree`](@ref) (d-ary: `root_children = branching = k`).

Node numbering is BFS level-order (1-indexed): root = 1, then level 1, etc.

# Fields
- `root_children::Int`: Number of children of the root
- `branching::Int`: Number of children of each non-root internal node
- `height::Int`: Number of levels (`h ≥ 1`; `h = 1` = root only)
- `n_nodes::Int`: Total number of nodes
- `states::Vector{Int8}`: Node states (primitive array)
"""
mutable struct RegularTree <: AbstractImplicitGraph
    root_children::Int
    branching::Int
    height::Int
    n_nodes::Int
    states::Vector{Int8}

    function RegularTree(root_children::Int, branching::Int, height::Int)
        root_children >= 1 || throw(ArgumentError("root_children must be ≥ 1, got $root_children"))
        branching >= 1 || throw(ArgumentError("branching must be ≥ 1, got $branching"))
        height >= 1 || throw(ArgumentError("height must be ≥ 1, got $height"))
        n = _tree_n_nodes(root_children, branching, height)
        new(root_children, branching, height, n, zeros(Int8, n))
    end
end

"""
Total node count of a tree whose root has `R` children, whose other internal
nodes have `b` children, and which has `h` complete levels.

`n = 1 + R·(b^(h-1) − 1)/(b − 1)`, with the `b == 1` case (a path-like tree)
handled separately to avoid dividing by zero.
"""
function _tree_n_nodes(R::Int, b::Int, h::Int)::Int
    h == 1 && return 1
    b == 1 && return 1 + R * (h - 1)
    return 1 + R * (b^(h - 1) - 1) ÷ (b - 1)
end

# =============================================================================
# Child / parent index arithmetic (O(1), shared by topology and layout)
# =============================================================================
#
# BFS order fills the root's `R` children into indices 2..R+1, then each
# subsequent node contributes `b` children. So for node i ≥ 2 the first child is
# at R + 2 + (i-2)·b, and the parent inverts that.

@inline _num_children(tree::RegularTree, i::Int)::Int =
    i == 1 ? tree.root_children : tree.branching

@inline function _first_child(tree::RegularTree, i::Int)::Int
    i == 1 && return 2
    return tree.root_children + 2 + (i - 2) * tree.branching
end

@inline function _parent(tree::RegularTree, i::Int)::Int
    # Caller guarantees i ≥ 2.
    R = tree.root_children
    i <= R + 1 && return 1
    return 2 + (i - R - 2) ÷ tree.branching
end

# =============================================================================
# Core Interface Implementation (Required Methods)
# =============================================================================

function get_neighbors!(neighbors::Vector{Int}, tree::RegularTree, node_id::Int)::Vector{Int}
    _check_node(tree, node_id)
    n = tree.n_nodes
    empty!(neighbors)

    # Parent: every node except the root has one.
    if node_id > 1
        push!(neighbors, _parent(tree, node_id))
    end

    # Children: a contiguous block in BFS order.
    first_child = _first_child(tree, node_id)
    if first_child <= n
        last_child = min(first_child + _num_children(tree, node_id) - 1, n)
        sizehint!(neighbors, length(neighbors) + last_child - first_child + 1)
        @inbounds for c in first_child:last_child
            push!(neighbors, c)
        end
    end

    return neighbors
end

# Degree: root has `root_children` (or 0 for a single-node tree); a node with no
# children is a leaf (degree 1, just its parent); other non-root nodes have
# `branching` children + 1 parent.
@inline function get_node_degree(tree::RegularTree, node_id::Int)::Int
    _check_node(tree, node_id)
    n = tree.n_nodes
    node_id == 1 && return tree.height == 1 ? 0 : tree.root_children
    _first_child(tree, node_id) > n && return 1
    return tree.branching + 1
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# Root sits at the origin; level-l nodes live on the shell of radius l — a circle
# in 2D, a sphere in 3D — so radius encodes depth and the picture grows outward as
# infection spreads from the root.
#
# Both layouts are *subtree-coherent*: a node's children are placed near it, so
# edges fan outward instead of crossing the figure.
# - 2D: nodes are equidistant on each circle, and each node sits at the CENTER of
#   its angular arc (the `+0.5`). Because BFS numbering keeps a subtree's nodes
#   contiguous, that arc is exactly the union of its children's arcs, so children
#   straddle their parent — equidistant *and* coherent.
# - 3D: a globally-equidistant sphere has no hierarchical nesting, so we start from
#   a recursive cone-tree (each node fans its children into a shrinking cap around
#   its own direction) and then run a short shell-constrained relaxation that
#   spreads nodes evenly while pinning each to its shell — see `_relax_shells!`.

supported_layout_dims(::RegularTree)::Tuple{Vararg{Int}} = (2, 3)

function node_positions(tree::RegularTree; dim::Int = 2)::Matrix{Float64}
    _check_layout_dim(tree, dim)
    return dim == 3 ? _tree_positions_3d(tree) : _tree_positions_2d(tree)
end

# Number of nodes at level `l` (0-indexed): 1 at the root, then R·b^(l-1).
@inline _level_count(tree::RegularTree, l::Int)::Int =
    l == 0 ? 1 : tree.root_children * tree.branching^(l - 1)

# Level (depth, 0-indexed) of every node, in BFS order.
function _node_levels(tree::RegularTree)::Vector{Int}
    levels = Vector{Int}(undef, tree.n_nodes)
    idx = 1
    @inbounds for l in 0:(tree.height - 1), _ in 1:_level_count(tree, l)
        levels[idx] = l
        idx += 1
    end
    return levels
end

function _tree_positions_2d(tree::RegularTree)::Matrix{Float64}
    n = tree.n_nodes
    pos = Matrix{Float64}(undef, 2, n)
    node_idx = 1
    @inbounds for level in 0:(tree.height - 1)
        count = _level_count(tree, level)
        r = Float64(level)
        for j in 0:(count - 1)
            θ = 2π * (j + 0.5) / count   # center of this node's angular arc
            pos[1, node_idx] = r * cos(θ)
            pos[2, node_idx] = r * sin(θ)
            node_idx += 1
        end
    end
    return pos
end

# --- 3D cone-tree initial layout ----------------------------------------------

# An orthonormal pair perpendicular to the unit vector `u`, used as the plane in
# which a node spreads its children's azimuths.
@inline function _perp_basis(u::NTuple{3,Float64})::NTuple{2,NTuple{3,Float64}}
    # Cross `u` with whichever axis it is least aligned to (avoids degeneracy).
    ax, ay, az = abs(u[1]), abs(u[2]), abs(u[3])
    a = (ax <= ay && ax <= az) ? (1.0, 0.0, 0.0) :
        (ay <= az ? (0.0, 1.0, 0.0) : (0.0, 0.0, 1.0))
    e1 = _normalize3((u[2]*a[3] - u[3]*a[2], u[3]*a[1] - u[1]*a[3], u[1]*a[2] - u[2]*a[1]))
    e2 = (u[2]*e1[3] - u[3]*e1[2], u[3]*e1[1] - u[1]*e1[3], u[1]*e1[2] - u[2]*e1[1])
    return (e1, e2)
end

@inline function _normalize3(v::NTuple{3,Float64})::NTuple{3,Float64}
    m = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    return m == 0 ? v : (v[1]/m, v[2]/m, v[3]/m)
end

const _TREE_CONE_SPREAD = 0.6   # child polar offset as a fraction of the parent cap
const _TREE_CONE_DECAY  = 0.55  # how much the cap narrows per level

# Place node `g` at radius `level` along unit direction `u`, then recurse into its
# children, fanning them around `u` within the cap `cap`.
function _place_tree_node!(pos::Matrix{Float64}, tree::RegularTree,
                           g::Int, u::NTuple{3,Float64}, level::Int, cap::Float64)
    r = Float64(level)
    @inbounds begin
        pos[1, g] = r * u[1]
        pos[2, g] = r * u[2]
        pos[3, g] = r * u[3]
    end
    n = tree.n_nodes
    first_child = _first_child(tree, g)
    first_child > n && return
    last_child = min(first_child + _num_children(tree, g) - 1, n)
    nch = last_child - first_child + 1

    e1, e2 = _perp_basis(u)
    β = cap * _TREE_CONE_SPREAD
    child_cap = cap * _TREE_CONE_DECAY
    cβ, sβ = cos(β), sin(β)
    for (idx, c) in enumerate(first_child:last_child)
        az = 2π * (idx - 1) / nch
        ca, sa = cos(az), sin(az)
        t = (ca*e1[1] + sa*e2[1], ca*e1[2] + sa*e2[2], ca*e1[3] + sa*e2[3])
        cu = (cβ*u[1] + sβ*t[1], cβ*u[2] + sβ*t[2], cβ*u[3] + sβ*t[3])
        _place_tree_node!(pos, tree, c, _normalize3(cu), level + 1, child_cap)
    end
    return
end

# --- 3D shell-constrained relaxation ------------------------------------------

const _TREE_RELAX_ITERS  = 300
const _TREE_RELAX_REP     = 1.0    # repulsion strength
const _TREE_RELAX_SPRING  = 0.4    # parent-child attraction strength
const _TREE_RELAX_STEP0   = 0.08   # initial step (cooled over iterations)

"""
Even out the 3D layout while keeping its structure. Starting from the cone-tree
positions, repeatedly apply inverse-square repulsion between all nodes and linear
parent-child springs, then project every node back onto its shell (radius = its
level). Repulsion spreads same-shell cousins apart; the springs and the radial
projection keep subtrees coherent and the shell structure intact.

Deterministic (no randomness) — the cone-tree seed makes the result reproducible.
"""
function _relax_shells!(pos::Matrix{Float64}, levels::Vector{Int}, tree::RegularTree)
    n = size(pos, 2)
    n <= 1 && return pos
    force = zeros(Float64, 3, n)
    for it in 1:_TREE_RELAX_ITERS
        fill!(force, 0.0)
        # Pairwise inverse-square repulsion.
        @inbounds for i in 1:n
            levels[i] == 0 && continue
            for j in 1:n
                i == j && continue
                dx = pos[1, i] - pos[1, j]
                dy = pos[2, i] - pos[2, j]
                dz = pos[3, i] - pos[3, j]
                d2 = dx*dx + dy*dy + dz*dz + 1e-6
                inv = _TREE_RELAX_REP / (d2 * sqrt(d2))   # → force magnitude ∝ 1/d²
                force[1, i] += dx * inv
                force[2, i] += dy * inv
                force[3, i] += dz * inv
            end
        end
        # Parent-child springs (pull both endpoints together).
        @inbounds for i in 2:n
            p = _parent(tree, i)
            dx = pos[1, p] - pos[1, i]
            dy = pos[2, p] - pos[2, i]
            dz = pos[3, p] - pos[3, i]
            force[1, i] += _TREE_RELAX_SPRING * dx
            force[2, i] += _TREE_RELAX_SPRING * dy
            force[3, i] += _TREE_RELAX_SPRING * dz
            force[1, p] -= _TREE_RELAX_SPRING * dx
            force[2, p] -= _TREE_RELAX_SPRING * dy
            force[3, p] -= _TREE_RELAX_SPRING * dz
        end
        # Cooling step, then project each node back onto its shell.
        step = _TREE_RELAX_STEP0 * (1 - (it - 1) / _TREE_RELAX_ITERS)
        @inbounds for i in 1:n
            li = levels[i]
            li == 0 && continue
            x = pos[1, i] + step * force[1, i]
            y = pos[2, i] + step * force[2, i]
            z = pos[3, i] + step * force[3, i]
            m = sqrt(x*x + y*y + z*z)
            if m == 0
                x, y, z, m = 1.0, 0.0, 0.0, 1.0
            end
            s = li / m
            pos[1, i] = x * s
            pos[2, i] = y * s
            pos[3, i] = z * s
        end
    end
    return pos
end

function _tree_positions_3d(tree::RegularTree)::Matrix{Float64}
    R, n = tree.root_children, tree.n_nodes
    pos = zeros(Float64, 3, n)            # root (node 1) stays at the origin
    if tree.height >= 2
        # Root's children seed the whole sphere (no parent direction to fan from),
        # then each seeds a coherent cone for its own subtree.
        dirs = _fibonacci_sphere(R)
        cap0 = π / 2
        @inbounds for m in 1:R
            child = m + 1                 # root is node 1; its children are 2..R+1
            u = _normalize3((dirs[1, m], dirs[2, m], dirs[3, m]))
            _place_tree_node!(pos, tree, child, u, 1, cap0)
        end
    end
    _relax_shells!(pos, _node_levels(tree), tree)
    return pos
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
Create a graph-theory **regular tree** (Cayley tree / Bethe lattice) of degree
`degree` and `height` levels.

Every internal vertex has degree `degree`: the root has `degree` children and
every other internal node has `degree − 1` children (plus one parent). This is the
canonical regular tree for percolation/branching-process analysis — the branching
ratio is `degree − 1`.

Note `degree = 2` is degenerate: the root has two children and every other
internal node has a single child, so the tree is a **path** (two arms from the
root). Use `degree ≥ 3` for a genuinely branching regular tree, or
[`create_dary_tree`](@ref) for a balanced binary tree.

# Arguments
- `degree::Int`: Degree of every internal vertex (`≥ 2`)
- `height::Int`: Number of levels (`h ≥ 1`; `h = 1` = root only)

# Returns
- `RegularTree`: regular tree with root degree `degree`, internal branching `degree − 1`

# Examples
```julia
julia> create_regular_tree(3, 4)   # degree-3 Cayley tree: root→3, then →2 each
julia> create_regular_tree(4, 3)   # degree-4 Cayley tree
```
"""
function create_regular_tree(degree::Int, height::Int)::RegularTree
    degree >= 2 || throw(ArgumentError("degree must be ≥ 2, got $degree"))
    return RegularTree(degree, degree - 1, height)
end

"""
Create a balanced **d-ary tree** (computer-science convention) with branching
factor `branching` and `height` levels.

Every internal node — the root included — has exactly `branching` children, so a
non-root internal node has degree `branching + 1`. The tree has
`(branchingʰ − 1) ÷ (branching − 1)` nodes.

# Arguments
- `branching::Int`: Children per internal node (`≥ 2`)
- `height::Int`: Number of levels (`h ≥ 1`; `h = 1` = root only, `h = 2` = root + `branching` leaves)

# Returns
- `RegularTree`: balanced d-ary tree

# Examples
```julia
julia> create_dary_tree(2, 4)   # binary tree: 15 nodes, 4 levels
julia> create_dary_tree(3, 3)   # ternary tree: 13 nodes, 3 levels
```
"""
function create_dary_tree(branching::Int, height::Int)::RegularTree
    branching >= 2 || throw(ArgumentError("branching must be ≥ 2, got $branching"))
    return RegularTree(branching, branching, height)
end
