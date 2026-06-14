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
mutable struct ChaseEscapeProcess{R<:AbstractRNG} <: SIRLikeProcess
    graph::AbstractEpidemicGraph
    λ::Float64
    μ::Float64
    ghost::Bool
    spread_tracker::DictActiveTracker
    catch_tracker::DictActiveTracker
    time::Float64
    steps::Int
    # Parametric on the concrete RNG type to keep rand()/randexp() statically
    # dispatched and allocation-free in the hot path (see issues #1/#3).
    rng::R

    function ChaseEscapeProcess(graph::AbstractEpidemicGraph, λ::Float64, μ::Float64;
                                ghost::Bool = true,
                                rng::R = Random.default_rng()) where {R<:AbstractRNG}
        _validate_chase_escape_parameters(λ, μ)
        new{R}(graph, λ, μ, ghost,
            DictActiveTracker(), DictActiveTracker(),
            0.0, 0, rng)
    end
end

# =============================================================================
# Required Interface Implementation
# =============================================================================

@inline function get_graph(process::ChaseEscapeProcess)::AbstractEpidemicGraph
    return process.graph
end

@inline function current_time(process::ChaseEscapeProcess)::Float64
    return process.time
end

@inline function step_count(process::ChaseEscapeProcess)::Int
    return process.steps
end

function is_active(process::ChaseEscapeProcess)::Bool
    return get_total_boundary(process.spread_tracker) > 0 ||
           get_total_boundary(process.catch_tracker) > 0
end

function get_total_rate(process::ChaseEscapeProcess)::Float64
    spread_rate = process.λ * get_total_boundary(process.spread_tracker)
    catch_rate  = process.μ * get_total_boundary(process.catch_tracker)
    return spread_rate + catch_rate
end

function sample_active_node(process::ChaseEscapeProcess, rng::AbstractRNG)::Int
    n_active = length(process.spread_tracker.active_nodes)
    if n_active < 1024
        return _weighted_sample_active(process.spread_tracker, rng)
    else
        return _weighted_sample_active_fast(process.spread_tracker, n_active, rng)
    end
end

function step!(process::ChaseEscapeProcess)::Float64
    if !is_active(process)
        return Inf
    end

    total_rate = get_total_rate(process)
    if total_rate <= 0.0
        return Inf
    end

    dt = randexp(process.rng) / total_rate

    spread_rate = process.λ * get_total_boundary(process.spread_tracker)
    if rand(process.rng) < spread_rate / total_rate
        acting_node = sample_active_node(process, process.rng)
        _ce_spread!(process, acting_node)
    else
        caught_node = _sample_catch_node(process)
        _ce_catch!(process, caught_node)
    end

    process.time  += dt
    process.steps += 1
    return dt
end

function reset!(process::ChaseEscapeProcess,
                initial_red::Vector{Int};
                initial_blue::Vector{Int} = Int[],
                rng_seed::Union{Int, Nothing} = nothing)
    validate_initial_infected(initial_red, process.graph)
    _validate_blue_seeds(initial_blue, initial_red, process.graph)

    process.time  = 0.0
    process.steps = 0

    if rng_seed !== nothing
        Random.seed!(process.rng, rng_seed)
    end

    clear_active_nodes!(process.spread_tracker)
    clear_active_nodes!(process.catch_tracker)

    states = node_states_raw(process.graph)
    fill!(states, state_to_int(SUSCEPTIBLE))

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
    neighbors         = get_neighbors(process.graph, acting_node)
    states            = node_states_raw(process.graph)
    susceptible_state = state_to_int(SUSCEPTIBLE)
    infected_state    = state_to_int(INFECTED)

    white_neighbors = Int[]
    for neighbor in neighbors
        if states[neighbor] == susceptible_state
            push!(white_neighbors, neighbor)
        end
    end

    if isempty(white_neighbors)
        @warn "Red node $acting_node has no white neighbours but is in spread tracker"
        remove_active_node!(process.spread_tracker, acting_node)
        return
    end

    target = rand(process.rng, white_neighbors)
    states[target] = infected_state

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
# Internal Sampling Helper
# =============================================================================

function _sample_catch_node(process::ChaseEscapeProcess)::Int
    n_active = length(process.catch_tracker.active_nodes)
    if n_active < 1024
        return _weighted_sample_active(process.catch_tracker, process.rng)
    else
        return _weighted_sample_active_fast(process.catch_tracker, n_active, process.rng)
    end
end

# =============================================================================
# Chase-escape–Specific Statistics
# =============================================================================

function get_chase_escape_statistics(process::ChaseEscapeProcess)::Dict{Symbol, Any}
    base_stats = get_statistics(process)
    base_stats[:λ]               = process.λ
    base_stats[:μ]               = process.μ
    base_stats[:ghost]           = process.ghost
    base_stats[:escaped]         = has_escaped(process)
    base_stats[:active_red]      = length(process.spread_tracker.active_nodes)
    base_stats[:spread_boundary] = get_total_boundary(process.spread_tracker)
    base_stats[:catch_boundary]  = get_total_boundary(process.catch_tracker)
    return base_stats
end

# =============================================================================
# Parameter / Seed Validation
# =============================================================================

function _validate_chase_escape_parameters(λ::Float64, μ::Float64)
    if λ <= 0.0
        throw(ArgumentError("Red spread rate λ must be positive, got $λ"))
    end
    if μ <= 0.0
        throw(ArgumentError("Blue catch rate μ must be positive, got $μ"))
    end
    if λ > 1000.0 || μ > 1000.0
        @warn "Very large rates (λ=$λ, μ=$μ) may cause numerical issues"
    end
end

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
julia> ce = create_chase_escape_simulation(lattice, 2.0)
julia> results = run_simulation(ce)
```
"""
function create_chase_escape_simulation(graph::AbstractEpidemicGraph,
                                        λ::Float64, μ::Float64 = 1.0;
                                        ghost::Bool = true,
                                        initial_red::Union{Symbol, Vector{Int}} = :center,
                                        initial_blue::Vector{Int} = Int[],
                                        rng_seed::Union{Int, Nothing} = nothing)
    rng     = create_rng(rng_seed)
    process = ChaseEscapeProcess(graph, λ, μ; ghost = ghost, rng = rng)

    red_nodes = if initial_red == :center
        if hasmethod(get_center_node, (typeof(graph),))
            [get_center_node(graph)]
        else
            [num_nodes(graph) ÷ 2]
        end
    elseif initial_red == :random
        [rand(rng, 1:num_nodes(graph))]
    else
        initial_red
    end

    reset!(process, red_nodes; initial_blue = initial_blue)
    return process
end

"""
Convenience function for creating a Chase-escape simulation on a square lattice.
"""
function create_chase_escape_simulation(width::Int, height::Int,
                                        λ::Float64, μ::Float64 = 1.0;
                                        ghost::Bool = true,
                                        boundary::Symbol = :absorbing,
                                        initial_red::Union{Symbol, Vector{Int}} = :center,
                                        initial_blue::Vector{Int} = Int[],
                                        rng_seed::Union{Int, Nothing} = nothing)
    lattice = create_square_lattice(width, height, boundary)
    return create_chase_escape_simulation(lattice, λ, μ;
                                          ghost        = ghost,
                                          initial_red  = initial_red,
                                          initial_blue = initial_blue,
                                          rng_seed     = rng_seed)
end
