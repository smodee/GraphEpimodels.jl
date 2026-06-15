"""
Regular rooted k-ary tree implementation for epidemic modeling.

A *regular rooted tree* of branching factor `k` and height `h` has
`n = (kʰ − 1) ÷ (k − 1)` nodes in `h` levels. The root (node 1) is at level 1
and has `k` children; every internal node has `k` children and one parent; every
leaf has one parent. Node numbering follows 1-indexed BFS (heap) order, so
parent/child indices reduce to O(1) arithmetic:

- Parent of node `i > 1`:  `(i − 2) ÷ k + 1`
- Children of node `i`:    `k(i − 1) + 2` through `ki + 1` (if ≤ n)

Like [`StarGraph`](@ref) and [`PathGraph`](@ref), this is an
[`AbstractImplicitGraph`](@ref): only the state vector is stored; no adjacency
lists are materialized.
"""

# =============================================================================
# Regular Rooted Tree
# =============================================================================

"""
Regular rooted k-ary tree with branching factor `k` and `h` levels.

Node numbering is BFS level-order (1-indexed heap): root = 1,
level-1 nodes = 2..k+1, level-2 nodes = k+2..k²+k+1, and so on.

# Fields
- `branching_factor::Int`: Children per internal node (`k ≥ 2`)
- `height::Int`: Number of levels (`h ≥ 1`; `h = 1` = root only)
- `n_nodes::Int`: Total nodes `= (kʰ − 1) ÷ (k − 1)`
- `states::Vector{Int8}`: Node states (primitive array)
"""
mutable struct RegularTree <: AbstractImplicitGraph
    branching_factor::Int
    height::Int
    n_nodes::Int
    states::Vector{Int8}

    function RegularTree(branching_factor::Int, height::Int)
        k, h = branching_factor, height
        k >= 2 || throw(ArgumentError("branching_factor must be ≥ 2, got $k"))
        h >= 1 || throw(ArgumentError("height must be ≥ 1, got $h"))
        n = (k^h - 1) ÷ (k - 1)
        new(k, h, n, zeros(Int8, n))
    end
end

# =============================================================================
# Core Interface Implementation (Required Methods)
# =============================================================================

@inline function num_nodes(tree::RegularTree)::Int
    return tree.n_nodes
end

function node_states_raw(tree::RegularTree)::Vector{Int8}
    return tree.states
end

function set_node_states_raw!(tree::RegularTree, states::Vector{Int8})
    if length(states) != num_nodes(tree)
        throw(ArgumentError("Expected $(num_nodes(tree)) states, got $(length(states))"))
    end
    tree.states = states
end

function get_neighbors(tree::RegularTree, node_id::Int)::Vector{Int}
    return get_neighbors!(Int[], tree, node_id)
end

function get_neighbors!(neighbors::Vector{Int}, tree::RegularTree, node_id::Int)::Vector{Int}
    k, n = tree.branching_factor, tree.n_nodes
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
    empty!(neighbors)

    # Parent: every node except the root has one
    if node_id > 1
        push!(neighbors, (node_id - 2) ÷ k + 1)
    end

    # Children: k nodes at heap positions k(i-1)+2 .. ki+1
    first_child = k * (node_id - 1) + 2
    if first_child <= n
        last_child = min(k * node_id + 1, n)
        sizehint!(neighbors, length(neighbors) + last_child - first_child + 1)
        @inbounds for c in first_child:last_child
            push!(neighbors, c)
        end
    end

    return neighbors
end

# Degree: root has k children (or 0 if the tree is a single node), leaves have
# degree 1 (one parent), internal nodes have k children + 1 parent.
@inline function get_node_degree(tree::RegularTree, node_id::Int)::Int
    k, n = tree.branching_factor, tree.n_nodes
    if node_id < 1 || node_id > n
        throw(BoundsError("Node ID $node_id out of range [1, $n]"))
    end
    node_id == 1 && return tree.height == 1 ? 0 : k
    k * (node_id - 1) + 2 > n && return 1
    return k + 1
end

# =============================================================================
# Geometry Interface (for visualization)
# =============================================================================
#
# Root sits at the origin; level-l nodes are placed on a circle of radius l,
# equally spaced in angle. This reflects the radial symmetry of the tree and
# mirrors how infection spreads outward from the root.

has_layout(::RegularTree)::Bool = true
layout_dim(::RegularTree)::Int = 2

function node_positions(tree::RegularTree)::Matrix{Float64}
    k, h, n = tree.branching_factor, tree.height, tree.n_nodes
    pos = Matrix{Float64}(undef, 2, n)
    node_idx = 1
    @inbounds for level in 0:(h - 1)
        level_count = k^level
        r = Float64(level)
        for j in 0:(level_count - 1)
            θ = 2π * j / level_count
            pos[1, node_idx] = r * cos(θ)
            pos[2, node_idx] = r * sin(θ)
            node_idx += 1
        end
    end
    return pos
end

# =============================================================================
# Factory Function
# =============================================================================

"""
Create a regular rooted k-ary tree.

# Arguments
- `branching_factor::Int`: Children per internal node (`k ≥ 2`)
- `height::Int`: Number of levels (`h ≥ 1`; `h = 1` = root only, `h = 2` = root + k leaves)

# Returns
- `RegularTree`: k-ary tree with `(kʰ − 1) ÷ (k − 1)` nodes

# Examples
```julia
julia> create_regular_tree(2, 4)   # binary tree: 15 nodes, 4 levels
julia> create_regular_tree(3, 3)   # ternary tree: 13 nodes, 3 levels
```
"""
function create_regular_tree(branching_factor::Int, height::Int)::RegularTree
    return RegularTree(branching_factor, height)
end
