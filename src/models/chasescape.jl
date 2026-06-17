"""
Chase-escape model implementation.

A predator–prey competitive growth process on graphs (a.k.a. the escape model)
with three compartments mapped onto the package's S/I/R encoding:
  S = White  (empty / unoccupied)
  I = Red    (prey;     the spreading front)
  R = Blue   (predator; absorbing once reached)

Transition rules:
1. W→R: A red node spreads onto a white neighbour at rate λ per red–white edge;
         the white node becomes red (one-sided, only the white node changes).
2. R→B: A red node is caught by a blue neighbour at rate μ per red–blue edge;
         the caught red node becomes blue. Blue chases red; red escapes onto white.

Every vertex advances W → R → B monotonically and never moves backward, so the
process is `SIRLikeProcess`. The two event classes use the same optimised
weighted Gillespie machinery as ZIM / SIR / Maki-Thompson, with two
`DictActiveTracker`s:
  spread_tracker: red nodes weighted by WHITE-neighbour count (drives W→R)
  catch_tracker:  red nodes weighted by BLUE-neighbour count  (drives R→B)

Ghost node (default initial condition):
The canonical setup pairs a single red seed with one blue "ghost" node — an
extra node not part of the graph — so blue reaches the seed in exponential time
and then proceeds into the graph. A permanent blue node adjacent only to a red
seed contributes exactly +1 to that seed's blue-neighbour (catch) weight until
the seed turns blue, so the ghost is modelled as a +1 bump to the seed's
catch_tracker weight at reset time. It is never placed in the state array, so it
is never plotted, never counted by `count_states`, and not part of `num_nodes`.
Without ghost mode, the user supplies explicit `initial_blue` seeds instead.
"""

using Random

# =============================================================================
# ChaseEscape Process Implementation
# =============================================================================

"""
High-performance Chase-escape (predator–prey) process.

Uses two `DictActiveTracker` instances for the two event classes:
- `spread_tracker`: red (I) nodes weighted by white (S) neighbour count (W→R events)
- `catch_tracker`:  red (I) nodes weighted by blue (R) neighbour count (R→B events)

Both event types use weighted sampling from their respective trackers, identical
to the approach in ZIM, SIR and Maki-Thompson.

# Fields
- `graph::AbstractEpidemicGraph`: The underlying graph
- `λ::Float64`: Red spread rate per red–white edge (W→R)
- `μ::Float64`: Blue catch rate per red–blue edge (R→B)
- `ghost::Bool`: If `true`, attach a virtual blue ghost neighbour to each red seed
- `spread_tracker::DictActiveTracker`: red nodes with white neighbours
- `catch_tracker::DictActiveTracker`: red nodes with blue neighbours (incl. ghost)
- `time::Float64`: Current simulation time
- `steps::Int`: Number of steps executed
- `rng::AbstractRNG`: Random number generator
"""
mutable struct ChaseEscapeProcess{G<:AbstractEpidemicGraph, R<:AbstractRNG} <: SIRLikeProcess
    # Concrete `graph`/`rng` type parameters keep the neighbor queries and
    # rand()/randexp() calls in step! statically dispatched and box-free (#1/#3).
    graph::G
    λ::Float64
    μ::Float64
    ghost::Bool
    spread_tracker::DictActiveTracker
    catch_tracker::DictActiveTracker
    time::Float64
    steps::Int
    rng::R

    function ChaseEscapeProcess(graph::G, λ::Float64, μ::Float64;
                                ghost::Bool = true,
                                rng::R = Random.default_rng()) where {G<:AbstractEpidemicGraph, R<:AbstractRNG}
        _validate_rates("Red spread rate λ" => λ, "Blue catch rate μ" => μ)
        new{G,R}(graph, λ, μ, ghost,
            DictActiveTracker(), DictActiveTracker(),
            0.0, 0, rng)
    end
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

function is_active(process::ChaseEscapeProcess)::Bool
    return get_total_boundary(process.spread_tracker) > 0 ||
           get_total_boundary(process.catch_tracker) > 0
end

function get_total_rate(process::ChaseEscapeProcess)::Float64
    spread_rate = process.λ * get_total_boundary(process.spread_tracker)
    catch_rate  = process.μ * get_total_boundary(process.catch_tracker)
    return spread_rate + catch_rate
end

# Two event classes: red spread W→R (rate λ·spread-boundary) vs. blue catch R→B
# (rate μ·catch-boundary), each weighted-sampled from its own tracker. (The shared
# step! draws the waiting time and advances the clock.)
@inline function _fire_event!(process::ChaseEscapeProcess, total_rate::Float64)
    spread_rate = process.λ * get_total_boundary(process.spread_tracker)
    if rand(process.rng) < spread_rate / total_rate
        _ce_spread!(process, _weighted_sample_active(process.spread_tracker, process.rng))
    else
        _ce_catch!(process, _weighted_sample_active(process.catch_tracker, process.rng))
    end
end

function _clear_trackers!(process::ChaseEscapeProcess)
    clear_active_nodes!(process.spread_tracker)
    clear_active_nodes!(process.catch_tracker)
end

function reset!(process::ChaseEscapeProcess,
                initial_red::Vector{Int};
                initial_blue::Vector{Int} = Int[],
                rng_seed::Union{Int, Nothing} = nothing)
    validate_initial_infected(initial_red, process.graph)
    _validate_blue_seeds(initial_blue, initial_red, process.graph)
    states = _reset_prologue!(process; rng_seed = rng_seed)

    # Place blue seeds first so reds see their real blue neighbours.
    removed_state  = state_to_int(REMOVED)
    for node_id in initial_blue
        states[node_id] = removed_state
    end

    infected_state = state_to_int(INFECTED)
    for node_id in initial_red
        states[node_id] = infected_state
    end

    # Compute tracker weights now that all seeds are marked.
    for node_id in initial_red
        white_count = count_neighbors_by_state(process.graph, node_id, SUSCEPTIBLE)
        add_active_node!(process.spread_tracker, node_id, white_count)

        blue_count  = count_neighbors_by_state(process.graph, node_id, REMOVED)
        catch_count = process.ghost ? (blue_count + 1) : blue_count
        add_active_node!(process.catch_tracker, node_id, catch_count)
    end

    if !process.ghost && isempty(initial_blue)
        @warn "Chase-escape started with no blue (predator) seeds and ghost=false; " *
              "no red node can ever be caught (degenerate pure red-growth regime)."
    end
end

# =============================================================================
# Event Handlers (Internal)
# =============================================================================

function _ce_spread!(process::ChaseEscapeProcess, acting_node::Int)
    neighbors = get_neighbors(process.graph, acting_node)
    states    = node_states_raw(process.graph)
    # White nodes are SUSCEPTIBLE in the S/I/R encoding, so a "white neighbour" is a
    # susceptible neighbour — the shared picker applies directly.
    target = _random_susceptible_neighbor(neighbors, states, Int[], process.rng)

    if target == 0
        @warn "Red node $acting_node has no white neighbours but is in spread tracker"
        remove_active_node!(process.spread_tracker, acting_node)
        return
    end

    states[target] = state_to_int(INFECTED)
    _update_tracking_after_ce_spread!(process, target)
end

"""
Update both trackers after `new_red` (previously white) becomes red.

`new_red` joins both trackers with its own white/blue neighbour counts. Every
red neighbour of `new_red` lost one white neighbour, so its `spread_tracker`
weight decreases by 1. No catch weights change: `new_red` is red, not blue, so
it is not a catalyst for any neighbour's R→B transition.
"""
function _update_tracking_after_ce_spread!(process::ChaseEscapeProcess, new_red::Int)
    states         = node_states_raw(process.graph)
    infected_state = state_to_int(INFECTED)

    white_count = count_neighbors_by_state(process.graph, new_red, SUSCEPTIBLE)
    add_active_node!(process.spread_tracker, new_red, white_count)

    blue_count = count_neighbors_by_state(process.graph, new_red, REMOVED)
    add_active_node!(process.catch_tracker, new_red, blue_count)

    for neighbor in get_neighbors(process.graph, new_red)
        if states[neighbor] == infected_state
            spread_w = get(process.spread_tracker.active_nodes, neighbor, 0)
            update_active_node!(process.spread_tracker, neighbor, spread_w - 1)
        end
    end
end

function _ce_catch!(process::ChaseEscapeProcess, caught_node::Int)
    states         = node_states_raw(process.graph)
    infected_state = state_to_int(INFECTED)
    states[caught_node] = state_to_int(REMOVED)

    remove_active_node!(process.spread_tracker, caught_node)
    remove_active_node!(process.catch_tracker, caught_node)

    # `caught_node` went red→blue, so every red neighbour gained one blue
    # neighbour: increment its catch weight. Spread weights are unchanged
    # (`caught_node` was red, not white).
    for neighbor in get_neighbors(process.graph, caught_node)
        if states[neighbor] == infected_state
            catch_w = get(process.catch_tracker.active_nodes, neighbor, 0)
            update_active_node!(process.catch_tracker, neighbor, catch_w + 1)
        end
    end
end

# =============================================================================
# Chase-escape–Specific Statistics
# =============================================================================

function get_chase_escape_statistics(process::ChaseEscapeProcess)::Dict{Symbol, Any}
    return _augment_statistics(process;
        λ = process.λ,
        μ = process.μ,
        ghost = process.ghost,
        active_red = length(process.spread_tracker.active_nodes),
        spread_boundary = get_total_boundary(process.spread_tracker),
        catch_boundary = get_total_boundary(process.catch_tracker))
end

# =============================================================================
# Seed Validation
# =============================================================================

function _validate_blue_seeds(initial_blue::Vector{Int},
                              initial_red::Vector{Int},
                              graph::AbstractEpidemicGraph)
    isempty(initial_blue) && return

    n_nodes = num_nodes(graph)
    for node in initial_blue
        if node < 1 || node > n_nodes
            throw(ArgumentError("Blue seed index $node out of range [1, $n_nodes]"))
        end
    end
    if length(unique(initial_blue)) != length(initial_blue)
        throw(ArgumentError("Duplicate nodes in initial blue list"))
    end
    if !isempty(intersect(Set(initial_blue), Set(initial_red)))
        throw(ArgumentError("Initial red and blue seed sets must be disjoint"))
    end
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
Create a complete Chase-escape simulation setup.

# Arguments
- `graph::AbstractEpidemicGraph`: The graph to simulate on
- `λ::Float64`: Red spread rate per red–white edge (W→R)
- `μ::Float64`: Blue catch rate per red–blue edge (R→B) (default: 1.0)
- `ghost::Bool`: If `true` (default), attach a virtual blue ghost neighbour to each
  red seed so it is caught in exponential time. If `false`, use `initial_blue`.
- `initial_red::Union{Symbol, Vector{Int}}`: `:center`, `:random`, or node indices
- `initial_blue::Vector{Int}`: Explicit blue (predator) seeds; used when `ghost=false`
- `rng_seed::Union{Int, Nothing}`: Random seed for reproducibility

# Returns
- `ChaseEscapeProcess`: Configured process ready to run

# Example
```julia
julia> lattice = create_square_lattice(100, 100, :absorbing)
julia> ce = create_chase_escape_process(lattice, 2.0)
julia> results = run_simulation(ce)
```
"""
function create_chase_escape_process(graph::AbstractEpidemicGraph,
                                     λ::Float64, μ::Float64 = 1.0;
                                     ghost::Bool = true,
                                     initial_red::Union{Symbol, Vector{Int}} = :center,
                                     initial_blue::Vector{Int} = Int[],
                                     rng_seed::Union{Int, Nothing} = nothing)
    rng     = create_rng(rng_seed)
    process = ChaseEscapeProcess(graph, λ, μ; ghost = ghost, rng = rng)
    reset!(process, resolve_initial_nodes(graph, initial_red, rng); initial_blue = initial_blue)
    return process
end

"""
Convenience overload for creating a Chase-Escape process on a square lattice.
Keyword arguments (`ghost`, `boundary`, `initial_red`, `initial_blue`, `rng_seed`)
flow through to the graph-based method.
"""
create_chase_escape_process(width::Int, height::Int, λ::Float64, μ::Float64 = 1.0; kwargs...) =
    create_on_square_lattice(create_chase_escape_process, width, height, λ, μ; kwargs...)
