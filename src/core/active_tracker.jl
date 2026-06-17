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

    # Active node ids held in ascending (canonical) order, maintained incrementally
    # by the add/remove/update/clear mutators below. The weighted sampler reads this
    # directly instead of collecting the Dict keys and `sort!`-ing them on every
    # call, dropping the per-step sampling cost from O(active log active) to
    # O(active) (the cumulative-weight pass) while keeping the same deterministic,
    # order-independent selection. Invariant: `sorted_nodes` is exactly
    # `sort(collect(keys(active_nodes)))`.
    #
    # `cumw_buf` is reused scratch for the cumulative-weight array so sampling still
    # allocates nothing on the hot path. A tracker is owned by a single process (and
    # thus a single thread during threaded runs), so reusing it across calls — and
    # mutating `sorted_nodes` in place — is safe without any locking. See issues #1 / #3.
    sorted_nodes::Vector{Int}     # active node ids, ascending (canonical order)
    cumw_buf::Vector{Int}         # idx → cumulative weight up to and including idx

    DictActiveTracker() = new(Dict{Int, Int}(), 0, Int[], Int[])
end

# Keep `sorted_nodes` canonical under membership changes. Both are O(active) in the
# worst case (element shift), matching the cumulative-weight pass the sampler already
# pays — so the sort's log factor is removed without adding a higher-order cost.
@inline function _sorted_insert!(v::Vector{Int}, node_id::Int)
    insert!(v, searchsortedfirst(v, node_id), node_id)
end

@inline function _sorted_delete!(v::Vector{Int}, node_id::Int)
    # Caller guarantees node_id is present, so searchsortedfirst lands on it.
    deleteat!(v, searchsortedfirst(v, node_id))
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
        old == 0 && _sorted_insert!(tracker.sorted_nodes, node_id)  # newly active
    end
end

"""
Remove a node from active tracking.
"""
function remove_active_node!(tracker::DictActiveTracker, node_id::Int)
    # pop! returns the removed weight (0 if absent), so the total stays in sync
    # whether or not the node was present.
    removed = pop!(tracker.active_nodes, node_id, 0)
    tracker.total_weight -= removed
    removed > 0 && _sorted_delete!(tracker.sorted_nodes, node_id)  # was active
end

"""
Update neighbor count for an active node.
"""
function update_active_node!(tracker::DictActiveTracker, node_id::Int, new_count::Int)
    if new_count > 0
        old = get(tracker.active_nodes, node_id, 0)
        tracker.active_nodes[node_id] = new_count
        tracker.total_weight += new_count - old
        old == 0 && _sorted_insert!(tracker.sorted_nodes, node_id)  # newly active
    else
        removed = pop!(tracker.active_nodes, node_id, 0)
        tracker.total_weight -= removed
        removed > 0 && _sorted_delete!(tracker.sorted_nodes, node_id)  # dropped to inactive
    end
end

"""
Get all active nodes (a fresh copy, in ascending/canonical order).
"""
function get_active_nodes(tracker::DictActiveTracker)::Vector{Int}
    return copy(tracker.sorted_nodes)
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
    empty!(tracker.sorted_nodes)
    tracker.total_weight = 0
end

# =============================================================================
# Internal Sampling Helpers (Not Exported)
# =============================================================================

"""
Weighted sampler over the active set: selects a node with probability proportional
to its susceptible-neighbor count. Critical for processes like ZIM where event
rates depend on neighbor counts.

Builds a cumulative-weight array over the active set and binary-searches it
(`O(active)` to fill plus `O(log active)` to sample). Two properties matter:

1. **Allocation-free.** The canonical node array (`sorted_nodes`) and the
   cumulative-weight scratch (`cumw_buf`) live on the tracker and are resized in
   place, so this allocates nothing on the hot path. It runs on every Gillespie
   step of every simulation; per-step heap allocation here previously caused severe
   multithreaded GC pressure that serialized worker threads (issues #1 and #3).

2. **Deterministic.** Nodes are sampled in a canonical order (ascending node id),
   NOT in `Dict` iteration order. A `Dict`'s iteration order depends on its
   insertion/deletion/rehash history, so two trackers with identical contents but
   different histories — a freshly constructed process vs. one reused across many
   simulations, or per-thread processes that each ran a different block of work —
   would otherwise map the same random draw onto different nodes. That made
   results depend on execution mode (serial vs. threaded) for a fixed seed.

`sorted_nodes` is kept in ascending order incrementally by the tracker mutators,
so this sampler no longer collects the Dict keys and `sort!`s them on every call —
the per-step cost drops from O(active log active) to O(active) (just the cumulative
pass) while the selection stays bit-identical. The tracker is owned by a single
process, so reading/reusing its buffers needs no locking.

# Arguments
- `tracker::DictActiveTracker`: Active node tracker with neighbor counts
- `rng::AbstractRNG`: Random number generator

# Returns
- `Int`: Selected node ID (weighted by neighbor count)

# Throws
- `ArgumentError`: If no active nodes available
"""
function _weighted_sample_active(tracker::DictActiveTracker, rng::AbstractRNG)::Int
    nodes = tracker.sorted_nodes
    n_active = length(nodes)
    if n_active == 0
        throw(ArgumentError("No active nodes to sample from"))
    end

    if n_active == 1
        return @inbounds nodes[1]
    end

    # `sorted_nodes` is already in canonical (ascending) order, so we only need to
    # build cumulative weights over it — no per-call collect or sort.
    cumw = tracker.cumw_buf
    resize!(cumw, n_active)
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
