"""
analysis/survival_analysis.jl

High-performance Monte Carlo survival probability estimation for epidemic processes.
Uses threading for parallelization with minimal memory overhead and clean interface.

Primary use case: Estimating survival probabilities for the Zombie Infection Model (ZIM)
and other epidemic processes on lattices.
"""

using ProgressMeter

# =============================================================================
# Analysis Modes
# =============================================================================

@enum AnalysisMode begin
    MINIMAL     # Only survival probability and count (fastest)
    DETAILED    # Include survival times and final sizes (comprehensive)
end

# =============================================================================
# Survival Criteria
# =============================================================================

"""Abstract type for survival criteria"""
abstract type SurvivalCriterion end

struct EscapeCriterion <: SurvivalCriterion end
struct PersistenceCriterion <: SurvivalCriterion end  
struct ThresholdCriterion <: SurvivalCriterion
    threshold::Int
end

# =============================================================================
# High-Performance Survival Evaluation
# =============================================================================

"""Evaluate survival - inlined for performance"""
@inline function evaluate_survival(
    process::AbstractEpidemicProcess, 
    ::EscapeCriterion, 
    results::Dict{Symbol, Any}
)::Bool
    return has_escaped(process)
end

@inline function evaluate_survival(
    process::AbstractEpidemicProcess,
    ::PersistenceCriterion,
    results::Dict{Symbol, Any}  
)::Bool
    return is_active(process)
end

@inline function evaluate_survival(
    process::AbstractEpidemicProcess,
    criterion::ThresholdCriterion,
    results::Dict{Symbol, Any}
)::Bool
    return get_cluster_size(process) >= criterion.threshold
end

# =============================================================================
# Core Survival Analysis Functions
# =============================================================================

"""
Estimate survival probability using threading for maximum performance with minimal memory.

# Arguments
- `process_factory::Function`: Function that creates a fresh process (no arguments)
- `initial_infected::Vector{Int}`: Initial infected nodes
- `criterion::SurvivalCriterion`: How to define survival
- `num_simulations::Int`: Number of Monte Carlo runs
- `max_time::Float64`: Maximum time per simulation (default: Inf)
- `max_steps::Int`: Maximum steps per simulation (default: typemax(Int))
- `mode::AnalysisMode`: Data collection mode (MINIMAL or DETAILED)
- `use_threading::Bool`: Use threading for parallelization (default: true)

# Returns
- `Dict{Symbol, Any}`: Survival statistics (minimal or detailed based on mode)

# Examples
```julia
# Simple ZIM analysis
factory = () -> create_zim_simulation(100, 100, 2.0)
result = estimate_survival_probability(factory, [center_node])

# With detailed data collection
result = estimate_survival_probability(factory, [center_node]; mode=DETAILED)

# Custom factory with parameters
λ, width, height = 2.5, 200, 200
factory = () -> create_zim_simulation(width, height, λ; rng_seed=rand(UInt))
result = estimate_survival_probability(factory, [center_node]; num_simulations=10000)
```
"""
function estimate_survival_probability(
    process_factory::Function,
    initial_infected::Vector{Int},
    criterion::SurvivalCriterion = EscapeCriterion();
    num_simulations::Int = 1000,
    max_time::Float64 = Inf,
    max_steps::Int = typemax(Int),
    mode::AnalysisMode = MINIMAL,
    use_threading::Bool = Threads.nthreads() > 1
)::Dict{Symbol, Any}

    # Validation for PersistenceCriterion
    if criterion isa PersistenceCriterion && !isfinite(max_time)
        throw(ArgumentError("PersistenceCriterion requires finite max_time"))
    end

    try
        if use_threading && Threads.nthreads() > 1
            return _estimate_survival_threaded(
                process_factory, initial_infected, criterion,
                num_simulations, max_time, max_steps, mode
            )
        else
            return _estimate_survival_serial(
                process_factory, initial_infected, criterion,
                num_simulations, max_time, max_steps, mode
            )
        end
    finally
        # Always cleanup thread-local processes after analysis
        clear_thread_local_processes!()
    end
end

"""Threading implementation with thread-local process reuse for maximum efficiency"""
function _estimate_survival_threaded(
    process_factory::Function,
    initial_infected::Vector{Int},
    criterion::SurvivalCriterion,
    num_simulations::Int,
    max_time::Float64,
    max_steps::Int,
    mode::AnalysisMode
)::Dict{Symbol, Any}

    if mode == MINIMAL
        # Minimal mode: just count survivals
        survival_count = Threads.Atomic{Int}(0)
        
        @showprogress Threads.@threads for i in 1:num_simulations
            # Thread-local process - create once per thread, reuse across simulations
            thread_process = get_thread_local_process(process_factory)
            
            # Reset and run simulation (much faster than creating new process)
            reset!(thread_process, initial_infected)
            sim_results = run_simulation(thread_process; max_time=max_time, max_steps=max_steps)
            
            # Thread-safe increment if survived
            if evaluate_survival(thread_process, criterion, sim_results)
                Threads.atomic_add!(survival_count, 1)
            end
        end
        
        survival_prob = survival_count[] / num_simulations
        survival_se = sqrt(survival_prob * (1 - survival_prob) / num_simulations)
        
        return Dict{Symbol, Any}(
            :survival_probability => survival_prob,
            :survival_std_error => survival_se,
            :num_survivals => survival_count[],
            :num_extinctions => num_simulations - survival_count[]
        )
        
    else  # DETAILED mode
        # Pre-allocate results arrays
        results = Vector{NamedTuple{(:survived, :time, :size), Tuple{Bool, Float64, Int}}}(undef, num_simulations)
        
        @showprogress Threads.@threads for i in 1:num_simulations
            # Thread-local process - create once per thread, reuse across simulations
            thread_process = get_thread_local_process(process_factory)
            
            # Reset and run simulation
            reset!(thread_process, initial_infected)
            sim_results = run_simulation(thread_process; max_time=max_time, max_steps=max_steps)
            
            # Collect detailed data
            survived = evaluate_survival(thread_process, criterion, sim_results)
            survival_time = survived ? sim_results[:time] : NaN
            final_size = get_cluster_size(thread_process)
            
            results[i] = (survived=survived, time=survival_time, size=final_size)
        end
        
        # Process collected results
        survival_count = count(r -> r.survived, results)
        survival_times = [r.time for r in results if r.survived && !isnan(r.time)]
        final_sizes = [r.size for r in results]
        
        survival_prob = survival_count / num_simulations
        survival_se = sqrt(survival_prob * (1 - survival_prob) / num_simulations)
        
        return Dict{Symbol, Any}(
            :survival_probability => survival_prob,
            :survival_std_error => survival_se,
            :num_survivals => survival_count,
            :num_extinctions => num_simulations - survival_count,
            :mean_survival_time => length(survival_times) > 0 ? mean(survival_times) : NaN,
            :survival_times => survival_times,
            :mean_final_size => mean(final_sizes),
            :final_sizes => final_sizes
        )
    end
end

# =============================================================================
# Thread-Local Process Management
# =============================================================================

"""
Thread-local storage for epidemic processes.
Each thread gets its own process that's reused across simulations within a single analysis.
Key: (thread_id, factory_hash)
"""
const THREAD_LOCAL_PROCESSES = Dict{Tuple{Int, UInt64}, AbstractEpidemicProcess}()
const THREAD_PROCESS_LOCK = Threads.SpinLock()

"""
Get or create a thread-local process for maximum efficiency.

Each thread creates one process and reuses it across all simulations within
a single estimate_survival_probability call. Simple hashing ensures correctness.
"""
function get_thread_local_process(process_factory::Function)::AbstractEpidemicProcess
    thread_id = Threads.threadid()
    factory_hash = hash(process_factory)  # Simple, reliable hashing
    key = (thread_id, factory_hash)
    
    # Try to get existing process (lockless read for performance)
    existing_process = get(THREAD_LOCAL_PROCESSES, key, nothing)
    if existing_process !== nothing
        return existing_process
    end
    
    # Need to create new process - use lock for thread safety
    Threads.lock(THREAD_PROCESS_LOCK) do
        # Double-check pattern - another thread might have created it
        existing_process = get(THREAD_LOCAL_PROCESSES, key, nothing)
        if existing_process !== nothing
            return existing_process
        end
        
        # Create new process for this thread with error handling
        try
            new_process = process_factory()
            THREAD_LOCAL_PROCESSES[key] = new_process
            return new_process
        catch e
            @error "Failed to create process in thread $thread_id" exception=e
            rethrow()
        end
    end
end

"""
Clear all thread-local processes.
Called automatically at the end of each estimate_survival_probability call.
"""
function clear_thread_local_processes!()
    Threads.lock(THREAD_PROCESS_LOCK) do
        empty!(THREAD_LOCAL_PROCESSES)
        GC.gc()  # Force garbage collection to free memory
    end
end

"""Serial implementation - no parallelization"""
function _estimate_survival_serial(
    process_factory::Function,
    initial_infected::Vector{Int},
    criterion::SurvivalCriterion,
    num_simulations::Int,
    max_time::Float64,
    max_steps::Int,
    mode::AnalysisMode
)::Dict{Symbol, Any}

    # Create process once for serial execution
    process = process_factory()

    # Data collection based on mode
    survival_count = 0
    survival_times = mode == DETAILED ? Float64[] : nothing
    final_sizes = mode == DETAILED ? Int[] : nothing
    
    # Pre-allocate for detailed mode
    if mode == DETAILED
        sizehint!(survival_times, num_simulations ÷ 2)  # Rough estimate
        sizehint!(final_sizes, num_simulations)
    end
    
    @showprogress for i in 1:num_simulations
        # Reset to initial state
        reset!(process, initial_infected)
        
        # Run simulation
        sim_results = run_simulation(process; max_time=max_time, max_steps=max_steps)
        
        # Evaluate survival
        survived = evaluate_survival(process, criterion, sim_results)
        
        if survived
            survival_count += 1
            if mode == DETAILED
                push!(survival_times, sim_results[:time])
            end
        end
        
        if mode == DETAILED
            push!(final_sizes, get_cluster_size(process))
        end
    end
    
    # Compute core statistics
    survival_prob = survival_count / num_simulations
    survival_se = sqrt(survival_prob * (1 - survival_prob) / num_simulations)
    
    # Build results dict
    result = Dict{Symbol, Any}(
        :survival_probability => survival_prob,
        :survival_std_error => survival_se,
        :num_survivals => survival_count,
        :num_extinctions => num_simulations - survival_count
    )
    
    # Add detailed data if requested
    if mode == DETAILED
        result[:mean_survival_time] = length(survival_times) > 0 ? mean(survival_times) : NaN
        result[:survival_times] = survival_times
        result[:mean_final_size] = mean(final_sizes)
        result[:final_sizes] = final_sizes
    end
    
    return result
end

# =============================================================================
# Parameter Sweep Functions
# =============================================================================

"""
High-performance parameter sweep using threading.

# Arguments  
- `parameter_values::Vector{Float64}`: Parameter values to test
- `factory_generator::Function`: Function that takes parameter and returns factory function
- `initial_infected::Vector{Int}`: Initial infected nodes
- `criterion::SurvivalCriterion`: Survival criterion
- `kwargs...`: Additional arguments passed to estimate_survival_probability

# Returns
- Parameter sweep results with survival curves

# Examples
```julia
# ZIM lambda sweep
sweep = run_parameter_sweep(
    [1.5, 2.0, 2.5],
    λ -> (() -> create_zim_simulation(100, 100, λ)),
    [center_node]
)

# More complex factory with multiple parameters
sweep = run_parameter_sweep(
    [1.0:0.1:3.0...],
    λ -> (() -> create_zim_simulation(200, 200, λ, 1.5; rng_seed=rand(UInt))),
    [center_node];
    mode=DETAILED
)
```
"""
function run_parameter_sweep(
    parameter_values::Vector{Float64},
    factory_generator::Function,
    initial_infected::Vector{Int},
    criterion::SurvivalCriterion = EscapeCriterion();
    kwargs...
)::Dict{Symbol, Any}
    
    n_params = length(parameter_values)
    survival_probs = Vector{Float64}(undef, n_params)
    std_errors = Vector{Float64}(undef, n_params)
    
    @showprogress for (i, param) in enumerate(parameter_values)
        factory = factory_generator(param)
        results = estimate_survival_probability(
            factory, initial_infected, criterion; kwargs...
        )
        
        survival_probs[i] = results[:survival_probability]
        std_errors[i] = results[:survival_std_error]
    end
    
    return Dict{Symbol, Any}(
        :parameter_values => parameter_values,
        :survival_probabilities => survival_probs,
        :std_errors => std_errors
    )
end

"""
Convenience function for ZIM parameter sweeps with standard setup.

# Arguments
- `lambda_values::Vector{Float64}`: Lambda values to test
- `width::Int`: Lattice width  
- `height::Int`: Lattice height
- `mu::Float64`: Kill rate (default: 1.0)
- `initial_location::Symbol`: Where to start infection (:center, :corner, :edge)
- `kwargs...`: Additional arguments passed to estimate_survival_probability

# Example
```julia
sweep = run_zim_survival_analysis([1.5, 2.0, 2.5], 100, 100; mode=DETAILED)
```
"""
function run_zim_survival_analysis(
    lambda_values::Vector{Float64},
    width::Int,
    height::Int;
    mu::Float64 = 1.0,
    initial_location::Symbol = :center,
    kwargs...
)::Dict{Symbol, Any}
    
    # Create factory generator for ZIM processes
    factory_generator = λ -> (() -> create_zim_simulation(width, height, λ, mu))
    
    # Determine initial infected nodes
    if initial_location == :center
        # Create dummy process to get center node
        dummy_process = create_zim_simulation(width, height, lambda_values[1], mu)
        initial_infected = [get_center_node(get_graph(dummy_process))]
    elseif initial_location == :corner
        initial_infected = [1]  # Corner node
    elseif initial_location == :edge
        initial_infected = [width ÷ 2]  # Edge node
    else
        throw(ArgumentError("Unsupported initial_location: $initial_location. Use :center, :corner, or :edge"))
    end
    
    return run_parameter_sweep(
        lambda_values, factory_generator, initial_infected, EscapeCriterion(); 
        kwargs...
    )
end

# =============================================================================
# Threading Information and Setup Helpers
# =============================================================================

"""
Check current threading setup and provide guidance.
"""
function check_threading_setup()
    nthreads = Threads.nthreads()
    println("Julia threading setup:")
    println("  Current threads: $nthreads")
    println("  System CPU threads: $(Sys.CPU_THREADS)")
    
    if nthreads == 1
        println("\n⚠️  Threading disabled (only 1 thread)")
        println("To enable threading:")
        println("  1. Start Julia with: julia --threads=4")
        println("  2. Or set environment: JULIA_NUM_THREADS=4")
        println("  3. Or set threads in startup.jl")
    else
        println("✅ Threading enabled - survival analysis will be parallelized")
    end
    
    return nthreads
end

"""
Get recommended number of threads for survival analysis.
"""
function get_recommended_threads()::Int
    total_threads = Sys.CPU_THREADS
    if total_threads <= 2
        return total_threads
    elseif total_threads <= 8
        return total_threads - 1  # Reserve 1 for system
    else
        return min(8, total_threads - 2)  # Reserve 2, cap at 8
    end
end

# =============================================================================
# Exports
# =============================================================================

export AnalysisMode, MINIMAL, DETAILED
export SurvivalCriterion, EscapeCriterion, PersistenceCriterion, ThresholdCriterion
export estimate_survival_probability, run_parameter_sweep, run_zim_survival_analysis
export check_threading_setup, get_recommended_threads