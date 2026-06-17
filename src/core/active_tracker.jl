"""
Active-node tracking — the data structure at the heart of every SIR-like process.

An epidemic process should only iterate over the nodes that can actually cause an
event (e.g. infected nodes with at least one susceptible neighbor), not over the
whole graph. `DictActiveTracker` maintains that active set together with each
node's event weight (its susceptible-neighbor count), and provides the
allocation-free, deterministic weighted sampler the Gillespie steppers rely on.

This subsystem is independent of the process/graph interfaces: it only needs an
`AbstractRNG`. It is `include`d before `models/epiprocess.jl`, whose shared
substrate (and every concrete process) stores one or more `DictActiveTracker`s.
"""

using Random

# =============================================================================
# Active Node Tracking (Critical for Performance)
# =============================================================================

"""
Abstract interface for tracking active nodes efficiently.

This is critical for performance - epidemic processes should only iterate
over nodes that can actually cause events, not all infected nodes.
"""
abstract type ActiveNodeTracker end

"""
Simple active node tracker using a dictionary.

Maps node_id → number of susceptible neighbors.
This is the pattern from your efficient old implementation.
"""
mutable struct DictActiveTracker <: ActiveNodeTracker
    active_nodes::Dict{Int, Int}  # node_id → susceptible_neighbor_count

    # Scratch buffers reused by _weighted_sample_active_fast so that weighted
    # sampling allocates nothing on the hot path. A tracker is owned by a single
    # process (and thus a single thread during threaded runs), so reusing these
    # across calls is safe without any locking. See issues #1 / #3.
    nodes_buf::Vector{Int}        # idx → node_id
    cumw_buf::Vector{Int}         # idx → cumulative weight up to and including idx

    DictActiveTracker() = new(Dict{Int, Int}(), Int[], Int[])
end

"""
Add a node to active tracking.

# Arguments
- `tracker::DictActiveTracker`: The tracker
- `node_id::Int`: Node to add
- `neighbor_count::Int`: Number of susceptible neighbors
"""
function add_active_node!(tracker::DictActiveTracker, node_id::Int, neighbor_count::Int)
    if neighbor_count > 0
        tracker.active_nodes[node_id] = neighbor_count
    end
end

"""
Remove a node from active tracking.
"""
function remove_active_node!(tracker::DictActiveTracker, node_id::Int)
    delete!(tracker.active_nodes, node_id)
end

"""
Update neighbor count for an active node.
"""
function update_active_node!(tracker::DictActiveTracker, node_id::Int, new_count::Int)
    if new_count > 0
        tracker.active_nodes[node_id] = new_count
    else
        delete!(tracker.active_nodes, node_id)
    end
end

"""
Get all active nodes.
"""
function get_active_nodes(tracker::DictActiveTracker)::Vector{Int}
    return collect(keys(tracker.active_nodes))
end

"""
Get total boundary (sum of all neighbor counts).
"""
function get_total_boundary(tracker::DictActiveTracker)::Int
    return sum(values(tracker.active_nodes))
end

"""
Check if any nodes are active.
"""
function has_active_nodes(tracker::DictActiveTracker)::Bool
    return !isempty(tracker.active_nodes)
end

"""
Clear all active nodes.
"""
function clear_active_nodes!(tracker::DictActiveTracker)
    empty!(tracker.active_nodes)
end

# =============================================================================
# Internal Sampling Helpers (Not Exported)
# =============================================================================

"""
Weighted sampling from active nodes based on susceptible neighbor counts.

Implements efficient weighted sampling where the probability of selecting a node
is proportional to its number of susceptible neighbors. This is critical for
processes like ZIM where event rates depend on neighbor counts.

Uses the standard weighted sampling algorithm:
1. Calculate cumulative weights
2. Sample uniform random value in [0, total_weight]
3. Binary search to find selected node

# Arguments
- `tracker::DictActiveTracker`: Active node tracker with neighbor counts
- `rng::AbstractRNG`: Random number generator

# Returns
- `Int`: Selected node ID (weighted by neighbor count)

# Throws
- `ArgumentError`: If no active nodes available
"""
function _weighted_sample_active(tracker::DictActiveTracker, rng::AbstractRNG)::Int
    n_active = length(tracker.active_nodes)
    if n_active == 0
        throw(ArgumentError("No active nodes to sample from"))
    end

    # Delegate to the single canonical (deterministic) sampler so that every
    # active-set size goes through the same order-independent selection. The
    # previous implementation walked the Dict directly, which made the chosen
    # node depend on Dict iteration order — see _weighted_sample_active_fast.
    return _weighted_sample_active_fast(tracker, n_active, rng)
end

"""
Canonical weighted sampler over the active set, using the tracker's buffers.

Builds a cumulative-weight array and binary-searches it (`O(n)` to fill plus
`O(log n)` to sample). Two properties matter:

1. **Allocation-free.** The node/cumulative-weight scratch arrays live on the
   tracker and are resized in place, so this allocates nothing on the hot path.
   It runs on every Gillespie step of every simulation; per-step heap allocation
   here previously caused severe multithreaded GC pressure that serialized worker
   threads (issues #1 and #3).

2. **Deterministic.** Nodes are sampled in a canonical order (ascending node id),
   NOT in `Dict` iteration order. A `Dict`'s iteration order depends on its
   insertion/deletion/rehash history, so two trackers with identical contents but
   different histories — a freshly constructed process vs. one reused across many
   simulations, or per-thread processes that each ran a different block of work —
   would otherwise map the same random draw onto different nodes. That made
   results depend on execution mode (serial vs. threaded) for a fixed seed.
   Sorting by node id removes the dependence; QuickSort is in-place so this stays
   allocation-free.

The tracker is owned by a single process, so buffer reuse needs no locking.

# Arguments
- `tracker::DictActiveTracker`: Active node tracker with neighbor counts
- `n_active::Int`: Number of active nodes (length of `tracker.active_nodes`)
- `rng::AbstractRNG`: Random number generator

# Returns
- `Int`: Selected node ID (weighted by neighbor count)
"""
function _weighted_sample_active_fast(tracker::DictActiveTracker, n_active::Int, rng::AbstractRNG)::Int
    if n_active == 0
        throw(ArgumentError("No active nodes to sample from"))
    end

    if n_active == 1
        # first(d::Dict) returns a Pair{K,V}; .first extracts the key without
        # allocating a KeySet wrapper (unlike first(keys(d))).
        return first(tracker.active_nodes).first
    end

    # Reuse the tracker's scratch buffers instead of allocating fresh vectors.
    nodes = tracker.nodes_buf
    cumw = tracker.cumw_buf
    resize!(nodes, n_active)
    resize!(cumw, n_active)

    # Collect node ids, then sort into a canonical order so sampling does not
    # depend on Dict iteration order (see the docstring). QuickSort is in-place.
    # Iterate the Dict directly (not via keys()) to avoid allocating a KeySet
    # wrapper object on every call — the actual 16 B/step found in issue #11.
    i = 1
    for (node_id, _) in tracker.active_nodes
        nodes[i] = node_id
        i += 1
    end
    sort!(nodes; alg = QuickSort)

    # Build cumulative weights in canonical order.
    running = 0
    @inbounds for j in 1:n_active
        running += tracker.active_nodes[nodes[j]]
        cumw[j] = running
    end

    total_weight = running
    if total_weight <= 0
        throw(ArgumentError("Total weight is zero - no valid nodes to sample"))
    end

    # Binary search for efficiency (one rand() call, searchsortedfirst over the
    # cumulative weights).
    random_value = rand(rng) * total_weight
    idx = searchsortedfirst(cumw, random_value)

    # Handle edge case where random_value == total_weight
    if idx > n_active
        idx = n_active
    end

    return nodes[idx]
end
