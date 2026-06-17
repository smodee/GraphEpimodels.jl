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

    # Running sum of all weights in `active_nodes`, maintained incrementally by the
    # add/remove/update/clear mutators below so that `get_total_boundary` is O(1)
    # instead of an O(active) `sum(values(...))`. It runs on every Gillespie step
    # (often more than once per step for the two-event-class models), so the linear
    # sum was pure per-step overhead. Invariant: total_weight == sum(values(active_nodes)).
    total_weight::Int

    # Scratch buffers reused by _weighted_sample_active so that weighted sampling
    # allocates nothing on the hot path. A tracker is owned by a single process
    # (and thus a single thread during threaded runs), so reusing these across
    # calls is safe without any locking. See issues #1 / #3.
    nodes_buf::Vector{Int}        # idx → node_id
    cumw_buf::Vector{Int}         # idx → cumulative weight up to and including idx

    DictActiveTracker() = new(Dict{Int, Int}(), 0, Int[], Int[])
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
        # Adjust the running total by the delta vs. any existing weight, so a
        # repeated add (overwrite) stays consistent with the dict.
        old = get(tracker.active_nodes, node_id, 0)
        tracker.active_nodes[node_id] = neighbor_count
        tracker.total_weight += neighbor_count - old
    end
end

"""
Remove a node from active tracking.
"""
function remove_active_node!(tracker::DictActiveTracker, node_id::Int)
    # pop! returns the removed weight (0 if absent), so the total stays in sync
    # whether or not the node was present.
    tracker.total_weight -= pop!(tracker.active_nodes, node_id, 0)
end

"""
Update neighbor count for an active node.
"""
function update_active_node!(tracker::DictActiveTracker, node_id::Int, new_count::Int)
    if new_count > 0
        old = get(tracker.active_nodes, node_id, 0)
        tracker.active_nodes[node_id] = new_count
        tracker.total_weight += new_count - old
    else
        tracker.total_weight -= pop!(tracker.active_nodes, node_id, 0)
    end
end

"""
Get all active nodes.
"""
function get_active_nodes(tracker::DictActiveTracker)::Vector{Int}
    return collect(keys(tracker.active_nodes))
end

"""
Get total boundary (sum of all neighbor counts). O(1): returns the incrementally
maintained running sum (see `total_weight`).
"""
function get_total_boundary(tracker::DictActiveTracker)::Int
    return tracker.total_weight
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
    tracker.total_weight = 0
end

# =============================================================================
# Internal Sampling Helpers (Not Exported)
# =============================================================================

"""
Weighted sampler over the active set: selects a node with probability proportional
to its susceptible-neighbor count. Critical for processes like ZIM where event
rates depend on neighbor counts.

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
