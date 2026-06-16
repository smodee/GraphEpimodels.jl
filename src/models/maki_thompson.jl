"""
Maki-Thompson rumor spreading model implementation.

A rumor-spreading process on graphs with three compartments:
  S = Ignorants (unaware of the rumor)
  I = Spreaders (actively spreading the rumor)
  R = Stiflers  (know the rumor but no longer spread it)

Transition rules:
1. S→I: A spreader contacts an ignorant at rate α per I–S edge;
         the ignorant becomes a spreader (one-sided, only the ignorant changes).
2. I→R: A spreader contacts a catalyst neighbor at rate β per I–catalyst edge;
         ONLY the initiating spreader becomes a stifler.

The two variants differ in which neighbors count as catalysts for I→R:
  stifler_contact = true  (original Maki & Thompson 1973):
      catalysts = I neighbors ∪ R neighbors
  stifler_contact = false (spreader-only variant):
      catalysts = I neighbors only

Both variants use the same optimised two-tracker Gillespie algorithm.
"""

using Random

# =============================================================================
# MakiThompson Process Implementation
# =============================================================================

"""
High-performance Maki-Thompson rumor spreading process.

Uses two `DictActiveTracker` instances for the two event classes:
- `spreading_tracker`: I nodes weighted by S-neighbor count (for S→I events)
- `stifling_tracker`:  I nodes weighted by catalyst-neighbor count (for I→R events)

Both event types use weighted sampling from their respective trackers,
identical to the approach in ZIM and SIR.

# Fields
- `graph::AbstractEpidemicGraph`: The underlying graph
- `α::Float64`: Spreading rate per I–S contact
- `β::Float64`: Stifling rate per I–catalyst contact
- `stifler_contact::Bool`: If `true`, R neighbors also catalyse stifling (original MT).
                           If `false`, only I neighbors catalyse stifling.
- `spreading_tracker::DictActiveTracker`: I nodes with susceptible neighbours
- `stifling_tracker::DictActiveTracker`: I nodes with catalyst neighbours
- `spreaders::Set{Int}`: All currently infected (spreading) nodes
- `time::Float64`: Current simulation time
- `steps::Int`: Number of steps executed
- `rng::AbstractRNG`: Random number generator
"""
mutable struct MakiThompsonProcess{G<:AbstractEpidemicGraph, R<:AbstractRNG} <: SIRLikeProcess
    # Concrete `graph`/`rng` type parameters keep the neighbor queries and
    # rand()/randexp() calls in step! statically dispatched and box-free (#1/#3).
    graph::G
    α::Float64
    β::Float64
    stifler_contact::Bool
    spreading_tracker::DictActiveTracker
    stifling_tracker::DictActiveTracker
    spreaders::Set{Int}
    time::Float64
    steps::Int
    rng::R

    function MakiThompsonProcess(graph::G,
                                  α::Float64, β::Float64,
                                  stifler_contact::Bool;
                                  rng::R = Random.default_rng()) where {G<:AbstractEpidemicGraph, R<:AbstractRNG}
        _validate_rates("Spreading rate α" => α, "Stifling rate β" => β)
        new{G,R}(graph, α, β, stifler_contact,
            DictActiveTracker(), DictActiveTracker(),
            Set{Int}(), 0.0, 0, rng)
    end
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

function is_active(process::MakiThompsonProcess)::Bool
    return !isempty(process.spreaders)
end

function get_total_rate(process::MakiThompsonProcess)::Float64
    spreading_rate = process.α * get_total_boundary(process.spreading_tracker)
    stifling_rate  = process.β * get_total_boundary(process.stifling_tracker)
    return spreading_rate + stifling_rate
end

# Two event classes: spreading S→I (rate α·spreading-boundary) vs. stifling I→R
# (rate β·stifling-boundary), each weighted-sampled from its own tracker. (The
# shared step! draws the waiting time and advances the clock.)
@inline function _fire_event!(process::MakiThompsonProcess, total_rate::Float64)
    spreading_rate = process.α * get_total_boundary(process.spreading_tracker)
    if rand(process.rng) < spreading_rate / total_rate
        _mt_spread!(process, _weighted_sample_active(process.spreading_tracker, process.rng))
    else
        _mt_stifle!(process, _weighted_sample_active(process.stifling_tracker, process.rng))
    end
end

function _clear_trackers!(process::MakiThompsonProcess)
    clear_active_nodes!(process.spreading_tracker)
    clear_active_nodes!(process.stifling_tracker)
    empty!(process.spreaders)
end

function reset!(process::MakiThompsonProcess,
                initial_infected::Vector{Int};
                rng_seed::Union{Int, Nothing} = nothing)
    validate_initial_infected(initial_infected, process.graph)
    states = _reset_prologue!(process; rng_seed = rng_seed)

    infected_state = state_to_int(INFECTED)
    for node_id in initial_infected
        states[node_id] = infected_state
        push!(process.spreaders, node_id)
    end

    # Compute tracker weights now that all initial spreaders are marked
    for node_id in initial_infected
        s_count = count_neighbors_by_state(process.graph, node_id, SUSCEPTIBLE)
        add_active_node!(process.spreading_tracker, node_id, s_count)

        i_count = count_neighbors_by_state(process.graph, node_id, INFECTED)
        r_count = count_neighbors_by_state(process.graph, node_id, REMOVED)
        stifle_count = process.stifler_contact ? (i_count + r_count) : i_count
        add_active_node!(process.stifling_tracker, node_id, stifle_count)
    end
end

# =============================================================================
# Event Handlers (Internal)
# =============================================================================

function _mt_spread!(process::MakiThompsonProcess, acting_node::Int)
    neighbors = get_neighbors(process.graph, acting_node)
    states    = node_states_raw(process.graph)
    target = _random_susceptible_neighbor(neighbors, states, Int[], process.rng)

    if target == 0
        @warn "Spreader $acting_node has no susceptible neighbours but is in spreading tracker"
        remove_active_node!(process.spreading_tracker, acting_node)
        return
    end

    states[target] = state_to_int(INFECTED)
    push!(process.spreaders, target)
    _update_tracking_after_mt_spread!(process, acting_node, target)
end

"""
Update both trackers after `new_infected` (previously S) becomes I.

For every I-neighbour of `new_infected`:
- spreading_tracker weight decreases by 1 (lost a S-neighbour)
- stifling_tracker weight increases by 1 (gained a catalyst neighbour: new_infected is now I)

`new_infected` itself is added to both trackers with its initial weights.
"""
function _update_tracking_after_mt_spread!(process::MakiThompsonProcess,
                                            attacker::Int, new_infected::Int)
    states         = node_states_raw(process.graph)
    infected_state = state_to_int(INFECTED)

    # Add new_infected to spreading tracker
    s_count = count_neighbors_by_state(process.graph, new_infected, SUSCEPTIBLE)
    add_active_node!(process.spreading_tracker, new_infected, s_count)

    # Add new_infected to stifling tracker
    i_count = count_neighbors_by_state(process.graph, new_infected, INFECTED)
    r_count = count_neighbors_by_state(process.graph, new_infected, REMOVED)
    stifle_count = process.stifler_contact ? (i_count + r_count) : i_count
    add_active_node!(process.stifling_tracker, new_infected, stifle_count)

    # Update all I-neighbours of new_infected
    for neighbor in get_neighbors(process.graph, new_infected)
        if states[neighbor] == infected_state
            # Spreading: each I-neighbour lost new_infected as a S-neighbour
            spread_w = get(process.spreading_tracker.active_nodes, neighbor, 0)
            update_active_node!(process.spreading_tracker, neighbor, spread_w - 1)

            # Stifling: each I-neighbour gained new_infected as a catalyst
            stifle_w = get(process.stifling_tracker.active_nodes, neighbor, 0)
            update_active_node!(process.stifling_tracker, neighbor, stifle_w + 1)
        end
    end
end

function _mt_stifle!(process::MakiThompsonProcess, stifling_node::Int)
    states = node_states_raw(process.graph)
    states[stifling_node] = state_to_int(REMOVED)

    delete!(process.spreaders, stifling_node)
    remove_active_node!(process.spreading_tracker, stifling_node)
    remove_active_node!(process.stifling_tracker, stifling_node)

    # stifler_contact = true (original MT):
    #   stifling_node transitions I→R; both I and R are catalysts, so I-neighbours'
    #   stifling weights are unchanged (lost one I-catalyst, gained one R-catalyst).
    #
    # stifler_contact = false (spreader-only variant):
    #   stifling_node was an I-catalyst; now R (not a catalyst).
    #   Each I-neighbour loses one catalyst → decrement stifling tracker.
    if !process.stifler_contact
        infected_state = state_to_int(INFECTED)
        for neighbor in get_neighbors(process.graph, stifling_node)
            if states[neighbor] == infected_state
                w = get(process.stifling_tracker.active_nodes, neighbor, 0)
                if w > 0
                    update_active_node!(process.stifling_tracker, neighbor, w - 1)
                end
            end
        end
    end
end

# =============================================================================
# Maki-Thompson–Specific Statistics
# =============================================================================

function get_maki_thompson_statistics(process::MakiThompsonProcess)::Dict{Symbol, Any}
    return _augment_statistics(process;
        α = process.α,
        β = process.β,
        stifler_contact = process.stifler_contact,
        active_spreaders = length(process.stifling_tracker.active_nodes),
        spreading_boundary = get_total_boundary(process.spreading_tracker),
        stifling_boundary = get_total_boundary(process.stifling_tracker))
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
Create a complete Maki-Thompson simulation setup.

# Arguments
- `graph::AbstractEpidemicGraph`: The graph to simulate on
- `α::Float64`: Spreading rate per I–S contact
- `β::Float64`: Stifling rate per I–catalyst contact (default: 1.0)
- `stifler_contact::Bool`: If `true` (default), use the original 1973 model where
  both I and R neighbours catalyse I→R. If `false`, only I neighbours catalyse.
- `initial_infected::Union{Symbol, Vector{Int}}`: `:center`, `:random`, or node indices
- `rng_seed::Union{Int, Nothing}`: Random seed for reproducibility

# Returns
- `MakiThompsonProcess`: Configured process ready to run

# Example
```julia
julia> lattice = create_square_lattice(100, 100, :absorbing)
julia> mt = create_maki_thompson_process(lattice, 1.0)
julia> results = run_simulation(mt)
```
"""
function create_maki_thompson_process(graph::AbstractEpidemicGraph,
                                      α::Float64, β::Float64 = 1.0;
                                      stifler_contact::Bool = true,
                                      initial_infected::Union{Symbol, Vector{Int}} = :center,
                                      rng_seed::Union{Int, Nothing} = nothing)
    rng     = create_rng(rng_seed)
    process = MakiThompsonProcess(graph, α, β, stifler_contact; rng = rng)
    reset!(process, resolve_initial_nodes(graph, initial_infected, rng))
    return process
end

"""
Convenience overload for creating a Maki-Thompson process on a square lattice.
Keyword arguments (`stifler_contact`, `boundary`, `initial_infected`, `rng_seed`)
flow through to the graph-based method.
"""
create_maki_thompson_process(width::Int, height::Int, α::Float64, β::Float64 = 1.0; kwargs...) =
    create_on_square_lattice(create_maki_thompson_process, width, height, α, β; kwargs...)
