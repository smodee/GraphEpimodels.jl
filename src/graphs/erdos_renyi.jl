"""
Erdős–Rényi random graph type.

`ErdosRenyiGraph` is the package's dedicated random-graph type. It supports both
classic Erdős–Rényi models:

- **G(n, p)** — each of the `n(n-1)/2` possible edges is present independently with
  probability `p`.
- **G(n, m)** — exactly `m` edges are chosen uniformly at random.

Connectivity is stored as adjacency lists (the representation the simulation hot
path is optimized for): the type wraps an [`AdjacencyGraph`](@ref) and adds the
generating parameters as metadata. G(n, p) is sampled with the O(n + m)
Batagelj–Brandes geometric-skip algorithm rather than testing all O(n²) pairs, so
large sparse graphs generate quickly.
"""

using Random

# =============================================================================
# Type
# =============================================================================

"""
Erdős–Rényi random graph (adjacency-list backed).

# Fields
- `graph::AdjacencyGraph`: Wrapped storage; provides the full graph interface.
- `model::Symbol`: Which model generated it — `:gnp` or `:gnm`.
- `p::Float64`: Edge probability (G(n, p) parameter; realized density for G(n, m)).
- `m::Int`: Edge count (G(n, m) parameter; realized edge count for G(n, p)).

Construct via [`create_erdos_renyi`](@ref), or the [`create_gnp`](@ref) /
[`create_gnm`](@ref) shorthands.
"""
struct ErdosRenyiGraph <: AbstractEpidemicGraph
    graph::AdjacencyGraph
    model::Symbol
    p::Float64
    m::Int
end

function Base.show(io::IO, g::ErdosRenyiGraph)
    print(io, "ErdosRenyiGraph(n=$(num_nodes(g)), m=$(g.m), model=:$(g.model), ",
          "p=$(round(g.p; digits = 5)))")
end

# =============================================================================
# Interface — forwarded to the wrapped AdjacencyGraph
# =============================================================================
#
# The wrapper is immutable; mutation flows through the (mutable) inner graph. All
# hot-path methods delegate, so they reuse AdjacencyGraph's optimized,
# zero-allocation implementations and pre-computed degrees.

@inline num_nodes(g::ErdosRenyiGraph)::Int = num_nodes(g.graph)
get_neighbors(g::ErdosRenyiGraph, node_id::Int)::Vector{Int} = get_neighbors(g.graph, node_id)
get_neighbors!(buffer::Vector{Int}, g::ErdosRenyiGraph, node_id::Int)::Vector{Int} =
    get_neighbors!(buffer, g.graph, node_id)
@inline get_node_degree(g::ErdosRenyiGraph, node_id::Int)::Int = get_node_degree(g.graph, node_id)
node_states_raw(g::ErdosRenyiGraph)::Vector{Int8} = node_states_raw(g.graph)
set_node_states_raw!(g::ErdosRenyiGraph, states::Vector{Int8}) = set_node_states_raw!(g.graph, states)
get_boundary_nodes(g::ErdosRenyiGraph)::Vector{Int} = get_boundary_nodes(g.graph)
count_neighbors_by_state(g::ErdosRenyiGraph, node_id::Int, target_state::NodeState)::Int =
    count_neighbors_by_state(g.graph, node_id, target_state)

# Geometry interface (delegated; an ER graph has a layout only if coordinates were
# attached to the inner graph — see set_coords! below).
has_layout(g::ErdosRenyiGraph)::Bool = has_layout(g.graph)
layout_dim(g::ErdosRenyiGraph)::Int = layout_dim(g.graph)
node_positions(g::ErdosRenyiGraph)::Matrix{Float64} = node_positions(g.graph)

"""Attach (or replace) node coordinates for plotting (`dim × n`)."""
set_coords!(g::ErdosRenyiGraph, coords::AbstractMatrix{<:Real}) = (set_coords!(g.graph, coords); g)

# =============================================================================
# Generators (return raw adjacency lists)
# =============================================================================

"""Adjacency lists for the complete graph on `n` nodes."""
_complete_adjacency(n::Int)::Vector{Vector{Int}} =
    [Int[j for j in 1:n if j != i] for i in 1:n]

"""
Sample G(n, p) adjacency lists with the Batagelj–Brandes algorithm (O(n + m)).

Instead of drawing a Bernoulli for each of the `n(n-1)/2` candidate pairs, the
number of consecutive non-edges to skip is drawn from a geometric distribution,
so the work is proportional to the number of edges actually produced. Pairs are
enumerated in lexicographic order, which leaves both endpoints' neighbor lists
sorted with no explicit sort.
"""
function _erdos_renyi_gnp(n::Int, p::Float64, rng::AbstractRNG)::Vector{Vector{Int}}
    adjacency_list = [Int[] for _ in 1:n]
    (n < 2 || p <= 0.0) && return adjacency_list
    p >= 1.0 && return _complete_adjacency(n)

    # Reserve roughly the expected degree to cut push! reallocations.
    expected_degree = p * (n - 1)
    if expected_degree > 1
        hint = ceil(Int, expected_degree)
        for nb in adjacency_list
            sizehint!(nb, hint)
        end
    end

    log_1mp = log1p(-p)   # log(1 - p), accurate for small p
    # Enumerate lower-triangle pairs (v, w) with 0 ≤ w < v < n (0-indexed).
    v = 1
    w = -1
    while v < n
        r = rand(rng)                                  # ∈ [0, 1); never 1, so finite skip
        w += 1 + floor(Int, log1p(-r) / log_1mp)
        while w >= v && v < n
            w -= v
            v += 1
        end
        if v < n
            a = v + 1                                  # to 1-indexed; a > b
            b = w + 1
            push!(adjacency_list[a], b)
            push!(adjacency_list[b], a)
        end
    end
    return adjacency_list
end

"""
Map a 0-indexed strict-upper-triangle pair index to its `(a, b)` node pair (`a < b`,
1-indexed). Inverts `k = b₀(b₀-1)/2 + a₀` via the triangular-number formula, with a
small correction loop to absorb floating-point error in `sqrt`.
"""
function _pair_from_index(k::Int)::Tuple{Int, Int}
    b0 = floor(Int, (1 + sqrt(1 + 8 * k)) / 2)
    while b0 * (b0 - 1) ÷ 2 > k
        b0 -= 1
    end
    while (b0 + 1) * b0 ÷ 2 <= k
        b0 += 1
    end
    a0 = k - b0 * (b0 - 1) ÷ 2
    return (a0 + 1, b0 + 1)
end

"""
Sample G(n, m) adjacency lists: `m` distinct edges chosen uniformly at random.

Uses Floyd's algorithm to pick `m` distinct pair indices from the `n(n-1)/2`
possible in O(m) without rejection, then maps each index back to a node pair.
"""
function _erdos_renyi_gnm(n::Int, m::Int, rng::AbstractRNG)::Vector{Vector{Int}}
    total = n * (n - 1) ÷ 2
    if m < 0 || m > total
        throw(ArgumentError("Edge count m=$m out of range [0, $total] for n=$n nodes"))
    end
    adjacency_list = [Int[] for _ in 1:n]
    m == 0 && return adjacency_list
    m == total && return _complete_adjacency(n)

    # Floyd's algorithm: m distinct indices in 1:total.
    chosen = Set{Int}()
    sizehint!(chosen, m)
    for j in (total - m + 1):total
        t = rand(rng, 1:j)
        push!(chosen, (t in chosen) ? j : t)
    end

    expected_degree = 2m / n
    if expected_degree > 1
        hint = ceil(Int, expected_degree)
        for nb in adjacency_list
            sizehint!(nb, hint)
        end
    end

    for k in chosen
        a, b = _pair_from_index(k - 1)
        push!(adjacency_list[a], b)
        push!(adjacency_list[b], a)
    end
    # `chosen` iterates in hash order, so canonicalize each list.
    for nb in adjacency_list
        sort!(nb)
    end
    return adjacency_list
end

# =============================================================================
# Public constructors
# =============================================================================

"""
Create an Erdős–Rényi random graph.

Provide **exactly one** of `p` (the G(n, p) model) or `m` (the G(n, m) model).

# Arguments
- `n::Int`: Number of nodes (≥ 1).
- `p::Real` (keyword): Edge probability in `[0, 1]` for G(n, p).
- `m::Integer` (keyword): Edge count in `[0, n(n-1)/2]` for G(n, m).
- `rng::AbstractRNG` (keyword): Random number generator.

# Returns
- `ErdosRenyiGraph`

# Examples
```julia
julia> g = create_erdos_renyi(100; p = 0.1)        # G(n, p)
julia> h = create_erdos_renyi(100; m = 250)        # G(n, m)
```
"""
function create_erdos_renyi(n::Int;
                            p::Union{Real, Nothing} = nothing,
                            m::Union{Integer, Nothing} = nothing,
                            rng::AbstractRNG = Random.default_rng())::ErdosRenyiGraph
    n >= 1 || throw(ArgumentError("Number of nodes must be positive (got $n)"))
    if (p === nothing) == (m === nothing)
        throw(ArgumentError("Provide exactly one of `p` (G(n,p)) or `m` (G(n,m))"))
    end

    total = n * (n - 1) ÷ 2
    if p !== nothing
        (0.0 <= p <= 1.0) ||
            throw(ArgumentError("Edge probability p must be in [0, 1] (got $p)"))
        graph = AdjacencyGraph(_erdos_renyi_gnp(n, Float64(p), rng))
        realized_m = sum(graph.node_degrees) ÷ 2
        return ErdosRenyiGraph(graph, :gnp, Float64(p), realized_m)
    else
        m >= 0 || throw(ArgumentError("Edge count m must be non-negative (got $m)"))
        graph = AdjacencyGraph(_erdos_renyi_gnm(n, Int(m), rng))   # validates m ≤ total
        density = total == 0 ? 0.0 : m / total
        return ErdosRenyiGraph(graph, :gnm, density, Int(m))
    end
end

"""Shorthand for the G(n, p) Erdős–Rényi model. See [`create_erdos_renyi`](@ref)."""
create_gnp(n::Int, p::Real; rng::AbstractRNG = Random.default_rng())::ErdosRenyiGraph =
    create_erdos_renyi(n; p = p, rng = rng)

"""Shorthand for the G(n, m) Erdős–Rényi model. See [`create_erdos_renyi`](@ref)."""
create_gnm(n::Int, m::Integer; rng::AbstractRNG = Random.default_rng())::ErdosRenyiGraph =
    create_erdos_renyi(n; m = m, rng = rng)
