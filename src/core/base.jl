"""
Base types and interfaces for epidemic processes on graphs.

This module defines the abstract types and core interfaces that all
epidemic models and graph types must implement for the GraphEpimodels.jl package.

Supports multiple interacting particle systems including:
- ZIM (Zombie Infection Model)
- SIR (Susceptible-Infected-Recovered) 
- Contact Process (SIS model)
- Biased Voter Model
"""

using Random

# =============================================================================
# Node States for Different Models
# =============================================================================

"""
Standard node states for epidemic processes.

Note: Different models may not use all states:
- SIR, ZIM: Use all three states (S → I → R)
- Contact Process: Uses only S, I (S ⇄ I)
- Biased Voter: Uses I, S as competing opinions
"""
@enum NodeState begin
    SUSCEPTIBLE = 0
    INFECTED = 1  
    REMOVED = 2
end

# Convenient aliases for readability
const S = SUSCEPTIBLE
const I = INFECTED  
const R = REMOVED

# =============================================================================
# Boundary Conditions
# =============================================================================

"""
Boundary condition types for lattice graphs.

- ABSORBING: Process stops when reaching boundary (escape condition)
- PERIODIC: Toroidal topology (no boundary)
- REFLECTING: Particles bounce off boundary (rare in epidemic models)
"""
@enum BoundaryCondition begin
    ABSORBING
    PERIODIC
    REFLECTING
end

# =============================================================================
# Abstract Graph Interface
# =============================================================================

"""
Abstract base type for graph representations.

All graph implementations must provide methods for:
- Neighbor access and counting
- Boundary detection  
- State management
- Efficient iteration over active edges

Concrete implementations include:
- SquareLattice: 2D square lattices with various boundary conditions
- TreeGraph: Regular trees and general trees
- CompleteGraph: Complete graphs K_n
"""
abstract type EpidemicGraph end

"""
Get all neighbors of a specific node.

# Arguments
- `graph::EpidemicGraph`: The graph
- `node_id::Int`: Node to find neighbors for (1-indexed)

# Returns  
- `Vector{Int}`: Array of neighbor node IDs
"""
function get_neighbors(graph::EpidemicGraph, node_id::Int)::Vector{Int}
    error("get_neighbors must be implemented by concrete graph types")
end

"""
Get nodes on the graph boundary.

For lattices, these are edge nodes. For general graphs, 
these might be predefined "escape" nodes.

# Returns
- `Vector{Int}`: Array of boundary node IDs  
"""
function get_boundary_nodes(graph::EpidemicGraph)::Vector{Int}
    error("get_boundary_nodes must be implemented by concrete graph types")
end

"""
Get the degree (number of neighbors) of a node.

# Arguments
- `graph::EpidemicGraph`: The graph
- `node_id::Int`: Node to query

# Returns
- `Int`: Number of neighbors
"""
function get_node_degree(graph::EpidemicGraph, node_id::Int)::Int
    length(get_neighbors(graph, node_id))
end

"""
Get total number of nodes in the graph.

# Returns
- `Int`: Total number of nodes
"""
function num_nodes(graph::EpidemicGraph)::Int
    error("num_nodes must be implemented by concrete graph types")
end

"""
Get current state of all nodes.

# Returns
- `Vector{NodeState}`: Current state of each node
"""
function node_states(graph::EpidemicGraph)::Vector{NodeState}
    error("node_states must be implemented by concrete graph types")
end

"""
Set the state of all nodes.

# Arguments
- `graph::EpidemicGraph`: The graph
- `states::Vector{NodeState}`: New states for all nodes
"""
function set_node_states!(graph::EpidemicGraph, states::Vector{NodeState})
    error("set_node_states! must be implemented by concrete graph types")
end

"""
Get the state of a specific node.

# Arguments  
- `graph::EpidemicGraph`: The graph
- `node_id::Int`: Node to query

# Returns
- `NodeState`: Current state of the node
"""
function get_node_state(graph::EpidemicGraph, node_id::Int)::NodeState
    error("get_node_state must be implemented by concrete graph types")
end

"""
Set the state of a specific node.

# Arguments
- `graph::EpidemicGraph`: The graph  
- `node_id::Int`: Node to update
- `state::NodeState`: New state
"""
function set_node_state!(graph::EpidemicGraph, node_id::Int, state::NodeState)
    error("set_node_state! must be implemented by concrete graph types")
end

# =============================================================================
# Graph Utility Functions (Default Implementations)
# =============================================================================

"""
Count nodes in each state.

# Returns
- `Dict{NodeState, Int}`: Mapping from states to counts
"""
function count_states(graph::EpidemicGraph)::Dict{NodeState, Int}
    states = node_states(graph)
    counts = Dict{NodeState, Int}()
    
    # Initialize all states to 0
    for state in instances(NodeState)
        counts[state] = 0
    end
    
    # Count occurrences
    for state in states
        counts[state] += 1
    end
    
    return counts
end

"""
Get all nodes currently in a specific state.

# Arguments
- `graph::EpidemicGraph`: The graph
- `state::NodeState`: State to query

# Returns  
- `Vector{Int}`: Array of node IDs in the specified state
"""
function get_nodes_in_state(graph::EpidemicGraph, state::NodeState)::Vector{Int}
    states = node_states(graph)
    return findall(s -> s == state, states)
end

"""
Count neighbors of a node in a specific state.

# Arguments
- `graph::EpidemicGraph`: The graph
- `node_id::Int`: Node to query
- `target_state::NodeState`: State to count

# Returns
- `Int`: Number of neighbors in target state
"""
function count_neighbors_by_state(graph::EpidemicGraph, node_id::Int, 
                                 target_state::NodeState)::Int
    neighbors = get_neighbors(graph, node_id)
    states = node_states(graph)
    return count(i -> states[i] == target_state, neighbors)
end

"""
Get all active edges for epidemic processes.

An active edge connects nodes in states that can interact.
For most epidemic models, this means infected-susceptible pairs.

# Arguments
- `graph::EpidemicGraph`: The graph
- `from_state::NodeState`: Source state (default: INFECTED)
- `to_state::NodeState`: Target state (default: SUSCEPTIBLE)

# Returns
- `Vector{Tuple{Int, Int}}`: Array of (from_node, to_node) pairs
"""
function get_active_edges(graph::EpidemicGraph, 
                         from_state::NodeState = INFECTED,
                         to_state::NodeState = SUSCEPTIBLE)::Vector{Tuple{Int, Int}}
    active_edges = Tuple{Int, Int}[]
    states = node_states(graph)
    
    from_nodes = get_nodes_in_state(graph, from_state)
    
    for from_node in from_nodes
        neighbors = get_neighbors(graph, from_node)
        for neighbor in neighbors
            if states[neighbor] == to_state
                push!(active_edges, (from_node, neighbor))
            end
        end
    end
    
    return active_edges
end

# =============================================================================
# Abstract Process Interface  
# =============================================================================

"""
Abstract base type for epidemic processes.

All epidemic process implementations must provide:
- Single-step evolution
- Activity detection (when process should stop)
- Rate computation for stochastic simulation
- Statistics and state information

Concrete implementations include:
- ZIMProcess: Zombie Infection Model
- SIRProcess: Standard SIR model  
- ContactProcess: Contact process (SIS model)
- BiasedVoterProcess: Biased voter model
"""
abstract type EpidemicProcess end

"""
Execute one simulation step.

Updates the process state and returns the time increment.
Uses Gillespie algorithm internally for exact stochastic simulation.

# Arguments
- `process::EpidemicProcess`: The process to step

# Returns  
- `Float64`: Time increment for this step (Inf if no events possible)
"""
function step!(process::EpidemicProcess)::Float64
    error("step! must be implemented by concrete process types")
end

"""
Check if the process should continue running.

# Arguments
- `process::EpidemicProcess`: The process to check

# Returns
- `Bool`: true if process should continue, false if stopped
"""
function is_active(process::EpidemicProcess)::Bool  
    error("is_active must be implemented by concrete process types")
end

"""
Get the total rate of all possible events.

Used by Gillespie algorithm to determine time increments.

# Arguments  
- `process::EpidemicProcess`: The process to query

# Returns
- `Float64`: Total event rate
"""
function get_total_rate(process::EpidemicProcess)::Float64
    error("get_total_rate must be implemented by concrete process types") 
end

"""
Get current simulation time.

# Arguments
- `process::EpidemicProcess`: The process to query

# Returns
- `Float64`: Current simulation time
"""
function current_time(process::EpidemicProcess)::Float64
    error("current_time must be implemented by concrete process types")
end

"""
Get current step count.

# Arguments  
- `process::EpidemicProcess`: The process to query

# Returns
- `Int`: Number of steps executed
"""
function step_count(process::EpidemicProcess)::Int
    error("step_count must be implemented by concrete process types")
end

"""
Get the underlying graph.

# Arguments
- `process::EpidemicProcess`: The process to query

# Returns  
- `EpidemicGraph`: The graph on which the process runs
"""
function get_graph(process::EpidemicProcess)::EpidemicGraph
    error("get_graph must be implemented by concrete process types")
end

# =============================================================================
# Process Utility Functions (Default Implementations)
# =============================================================================

"""
Get current simulation statistics.

# Arguments
- `process::EpidemicProcess`: The process to analyze

# Returns
- `Dict{Symbol, Any}`: Dictionary with current state information
"""
function get_statistics(process::EpidemicProcess)::Dict{Symbol, Any}
    graph = get_graph(process)
    state_counts = count_states(graph)
    
    return Dict{Symbol, Any}(
        :time => current_time(process),
        :step_count => step_count(process),
        :susceptible => state_counts[SUSCEPTIBLE],
        :infected => state_counts[INFECTED], 
        :removed => state_counts[REMOVED],
        :total_ever_infected => state_counts[INFECTED] + state_counts[REMOVED],
        :is_active => is_active(process),
    )
end

"""
Reset process to initial conditions.

# Arguments
- `process::EpidemicProcess`: The process to reset
- `initial_infected::Vector{Int}`: Nodes to start as infected
"""
function reset!(process::EpidemicProcess, initial_infected::Vector{Int})
    error("reset! must be implemented by concrete process types")
end

"""
Check if infection has reached the graph boundary.

Useful for detecting "outbreak" conditions on lattices.

# Arguments  
- `process::EpidemicProcess`: The process to check

# Returns
- `Bool`: true if infection has reached boundary
"""
function has_reached_boundary(process::EpidemicProcess)::Bool
    graph = get_graph(process)
    boundary_nodes = get_boundary_nodes(graph)
    states = node_states(graph)
    
    return any(i -> states[i] == INFECTED, boundary_nodes)
end

"""
Get total number of nodes ever infected.

# Arguments
- `process::EpidemicProcess`: The process to analyze

# Returns
- `Int`: Number of nodes in INFECTED or REMOVED state
"""
function get_cluster_size(process::EpidemicProcess)::Int
    stats = get_statistics(process)
    return stats[:total_ever_infected]
end

# =============================================================================
# Simulation Runner
# =============================================================================

"""
Run complete simulation until stopping condition.

# Arguments  
- `process::EpidemicProcess`: The process to run
- `max_time::Float64`: Maximum simulation time (default: Inf)
- `max_steps::Int`: Maximum number of steps (default: 1_000_000)
- `save_history::Bool`: Whether to save state snapshots (default: false)
- `history_interval::Int`: Steps between history saves (default: 100)

# Returns
- `Dict{Symbol, Any}`: Final simulation statistics and optional history
"""
function run_simulation(process::EpidemicProcess;
                       max_time::Float64 = Inf,
                       max_steps::Int = 1_000_000,
                       save_history::Bool = false,
                       history_interval::Int = 100)::Dict{Symbol, Any}
    
    history = save_history ? Dict{Symbol, Any}[] : nothing
    
    while (current_time(process) < max_time && 
           step_count(process) < max_steps && 
           is_active(process))
        
        # Save history if requested
        if save_history && step_count(process) % history_interval == 0
            snapshot = get_statistics(process)
            snapshot[:node_states] = copy(node_states(get_graph(process)))
            push!(history, snapshot)
        end
        
        # Execute one step
        dt = step!(process)
        
        # Break if no more events possible
        if dt == Inf
            break
        end
    end
    
    # Get final statistics
    final_stats = get_statistics(process)
    
    if save_history
        final_stats[:history] = history
    end
    
    return final_stats
end

# =============================================================================
# Model-Specific Process Types (for type hierarchy)
# =============================================================================

"""
Abstract type for SIR-like processes.

Includes models where nodes transition S → I → R:
- ZIM: Infection by neighbors, removal by fighting back
- SIR: Infection by neighbors, spontaneous recovery
"""
abstract type SIRLikeProcess <: EpidemicProcess end

"""
Abstract type for contact-like processes.  

Includes models where nodes transition S ⇄ I:
- Contact Process: Birth-death process on graphs
- SIS: SIR with reinfection possible
"""
abstract type ContactLikeProcess <: EpidemicProcess end

"""
Abstract type for voter-like processes.

Includes models with competing states/opinions:
- Biased Voter: Asymmetric influence between states
- Voter Model: Symmetric opinion dynamics
"""
abstract type VoterLikeProcess <: EpidemicProcess end