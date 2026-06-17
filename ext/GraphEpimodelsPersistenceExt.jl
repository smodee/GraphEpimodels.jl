"""
GraphEpimodelsPersistenceExt — CSV-backed survival-result storage.

Loads automatically when the user runs `using CSV, DataFrames` alongside
GraphEpimodels. Implements `get_next_start_seed` and
`update_or_append_survival_result` (declared in src/core/persistence.jl). Without
CSV/DataFrames loaded, those functions raise a friendly error.

The JSON/config helpers they rely on (`process_info_to_config_string`,
`process_info_to_json`, `parse_process_info_json`) are dependency-free and live
in the package.
"""
module GraphEpimodelsPersistenceExt

using GraphEpimodels
using CSV, DataFrames

# =============================================================================
# Batched in-memory survival-result store (read once / write once)
# =============================================================================
#
# `run_parameter_sweep` previously called the single-row `get_next_start_seed` and
# `update_or_append_survival_result` once per parameter; each re-read (and the
# latter re-wrote) the entire CSV and re-parsed every row's JSON config. The store
# below collapses that to one read up front and one write at the end: rows live in
# a DataFrame, and each existing row's reproducibility config string is parsed once
# on load and cached, so per-parameter seed lookups and result records are pure
# in-memory operations.

"""
In-memory survival-result store: the result `DataFrame` plus, parallel to its
rows, the cached config string of each row (so matching is one string compare, not
a JSON re-parse).
"""
mutable struct SurvivalStore
    df::DataFrame
    config_strings::Vector{String}
end

const _SURVIVAL_COLUMNS = (:parameter, :num_simulations, :num_survivals,
                           :survival_probability, :std_error, :start_seed,
                           :end_seed, :process_config)

"""Empty result DataFrame with the canonical column names and types."""
_empty_survival_df()::DataFrame = DataFrame(
    parameter            = Float64[],
    num_simulations      = Int[],
    num_survivals        = Int[],
    survival_probability = Float64[],
    std_error            = Float64[],
    start_seed           = Int[],
    end_seed             = Int[],
    process_config       = String[],
)

"""Reproducibility config string for an existing CSV row (parses its JSON once)."""
_row_config_string(row)::String =
    process_info_to_config_string(parse_process_info_json(String(row.process_config)))

"""
Load the survival-result store from `filename`, reading and parsing the file once.
A missing file yields an empty store; a corrupt file raises before any expensive
computation runs.
"""
function GraphEpimodels.load_survival_results(filename::String)::SurvivalStore
    if !isfile(filename)
        return SurvivalStore(_empty_survival_df(), String[])
    end
    local df
    try
        df = CSV.read(filename, DataFrame)
    catch e
        error("Failed to read CSV file '$filename'. Please verify the file is not corrupted before running expensive computations. Error: $e")
    end
    config_strings = [_row_config_string(row) for row in eachrow(df)]
    return SurvivalStore(df, config_strings)
end

"""Index of the row matching `(process_info, parameter_value)`, or `0` if none."""
function _find_match(store::SurvivalStore, parameter_value::Float64,
                     process_info::Dict{String, Any})::Int
    target = process_info_to_config_string(process_info)
    @inbounds for i in eachindex(store.config_strings)
        if store.config_strings[i] == target && store.df[i, :parameter] == parameter_value
            return i
        end
    end
    return 0
end

"""
Next starting seed for `(process_info, parameter_value)` against the in-memory
store: `end_seed + 1` of the matching row, or `1` for a fresh start. No I/O.
"""
function GraphEpimodels.next_start_seed(store::SurvivalStore, parameter_value::Float64,
                                        process_info::Dict{String, Any})::Int
    idx = _find_match(store, parameter_value, process_info)
    return idx == 0 ? 1 : store.df[idx, :end_seed] + 1
end

"""
Record a batch of new results into the in-memory store. If a row for
`(process_info, parameter_value)` exists, fold in the cumulative statistics;
otherwise append a new row. Returns `true` if an existing row was updated. No I/O —
call [`save_survival_results`](@ref) once afterwards to persist.
"""
function GraphEpimodels.record_survival_result!(
    store::SurvivalStore,
    parameter_value::Float64,
    num_simulations::Int,
    num_survivals::Int,
    survival_probability::Float64,
    std_error::Float64,
    start_seed::Int,
    end_seed::Int,
    process_info::Dict{String, Any}
)::Bool
    idx = _find_match(store, parameter_value, process_info)

    if idx != 0
        # Fold the new batch into the existing row's cumulative statistics.
        cumulative_num_sims      = store.df[idx, :num_simulations] + num_simulations
        cumulative_num_survivals = store.df[idx, :num_survivals] + num_survivals
        cumulative_survival_prob = cumulative_num_survivals / cumulative_num_sims
        cumulative_std_error     = sqrt(cumulative_survival_prob *
                                        (1 - cumulative_survival_prob) / cumulative_num_sims)

        store.df[idx, :num_simulations]      = cumulative_num_sims
        store.df[idx, :num_survivals]        = cumulative_num_survivals
        store.df[idx, :survival_probability] = cumulative_survival_prob
        store.df[idx, :std_error]            = cumulative_std_error
        store.df[idx, :end_seed]             = end_seed
        # start_seed and process_config stay as originally recorded.
        return true
    end

    push!(store.df, (
        parameter            = parameter_value,
        num_simulations      = num_simulations,
        num_survivals        = num_survivals,
        survival_probability = survival_probability,
        std_error            = std_error,
        start_seed           = start_seed,
        end_seed             = end_seed,
        process_config       = process_info_to_json(process_info),
    ))
    push!(store.config_strings, process_info_to_config_string(process_info))
    return false
end

"""Write the accumulated store to `filename` in a single pass."""
function GraphEpimodels.save_survival_results(store::SurvivalStore, filename::String)
    CSV.write(filename, store.df)
    return nothing
end

"""
Get the next starting seed for a parameter sweep continuation.

Checks the CSV file for an existing entry with matching configuration and parameter.
Returns 1 if no entry exists (fresh start), or end_seed + 1 if continuing from previous run.

# Arguments
- `filename::String`: Path to CSV file
- `parameter_value::Float64`: The parameter value (e.g., λ) being tested
- `process_info::Dict{String, Any}`: Process configuration from extract_process_info

# Returns
- `Int`: The seed to start from (1 for fresh start, or continuation seed)
"""
function GraphEpimodels.get_next_start_seed(
    filename::String,
    parameter_value::Float64,
    process_info::Dict{String, Any}
)::Int

    # If file doesn't exist, start from seed 1
    if !isfile(filename)
        return 1
    end

    # Read existing data with error handling
    local existing_df
    try
        existing_df = CSV.read(filename, DataFrame)
    catch e
        error("Failed to read CSV file '$filename'. Please verify the file is not corrupted before running expensive computations. Error: $e")
    end

    # Search for matching entry
    config_string = process_info_to_config_string(process_info)

    for row in eachrow(existing_df)
        # Parse existing config
        existing_config = parse_process_info_json(row.process_config)

        # Check if configurations and parameter match
        if process_info_to_config_string(existing_config) == config_string &&
           row.parameter == parameter_value
            # Found matching entry - continue from where it left off
            return row.end_seed + 1
        end
    end

    # No matching entry found - start fresh
    return 1
end

"""
Update existing survival result or append new one to CSV file.

If an entry with matching (config, parameter) exists, updates it with cumulative statistics.
Otherwise, appends a new row. Creates the CSV file if it doesn't exist.

# Arguments
- `filename::String`: Path to CSV file
- `parameter_value::Float64`: The parameter value (e.g., λ) that was tested
- `num_simulations::Int`: Number of NEW simulations run (not cumulative)
- `num_survivals::Int`: Number of NEW survivals observed (not cumulative)
- `survival_probability::Float64`: Survival probability from NEW simulations
- `std_error::Float64`: Standard error from NEW simulations
- `start_seed::Int`: First random seed used in NEW simulations
- `end_seed::Int`: Last random seed used in NEW simulations
- `process_info::Dict{String, Any}`: Process configuration from extract_process_info

# Returns
- `Bool`: true if existing row was updated, false if new row was appended
"""
function GraphEpimodels.update_or_append_survival_result(
    filename::String,
    parameter_value::Float64,
    num_simulations::Int,
    num_survivals::Int,
    survival_probability::Float64,
    std_error::Float64,
    start_seed::Int,
    end_seed::Int,
    process_info::Dict{String, Any}
)::Bool

    # Convert process info to JSON for storage
    process_config_json = process_info_to_json(process_info)

    # If file doesn't exist, create new file
    if !isfile(filename)
        new_row = DataFrame(
            parameter = [parameter_value],
            num_simulations = [num_simulations],
            num_survivals = [num_survivals],
            survival_probability = [survival_probability],
            std_error = [std_error],
            start_seed = [start_seed],
            end_seed = [end_seed],
            process_config = [process_config_json]
        )
        CSV.write(filename, new_row)
        return false  # New row appended
    end

    # Read existing data with error handling
    local existing_df
    try
        existing_df = CSV.read(filename, DataFrame)
    catch e
        error("Failed to read CSV file '$filename'. Please verify the file is not corrupted before running expensive computations. Error: $e")
    end

    # Search for matching entry to update
    config_string = process_info_to_config_string(process_info)

    for (idx, row) in enumerate(eachrow(existing_df))
        # Parse existing config
        existing_config = parse_process_info_json(row.process_config)

        # Check if configurations and parameter match
        if process_info_to_config_string(existing_config) == config_string &&
           row.parameter == parameter_value

            # Found matching entry - update with cumulative statistics
            cumulative_num_sims = row.num_simulations + num_simulations
            cumulative_num_survivals = row.num_survivals + num_survivals
            cumulative_survival_prob = cumulative_num_survivals / cumulative_num_sims
            cumulative_std_error = sqrt(cumulative_survival_prob * (1 - cumulative_survival_prob) / cumulative_num_sims)

            # Update the row in place
            existing_df[idx, :num_simulations] = cumulative_num_sims
            existing_df[idx, :num_survivals] = cumulative_num_survivals
            existing_df[idx, :survival_probability] = cumulative_survival_prob
            existing_df[idx, :std_error] = cumulative_std_error
            existing_df[idx, :end_seed] = end_seed
            # Note: start_seed stays as the original, process_config stays the same

            # Write updated dataframe back to file
            CSV.write(filename, existing_df)
            return true  # Existing row updated
        end
    end

    # No matching entry found - append new row
    new_row = DataFrame(
        parameter = [parameter_value],
        num_simulations = [num_simulations],
        num_survivals = [num_survivals],
        survival_probability = [survival_probability],
        std_error = [std_error],
        start_seed = [start_seed],
        end_seed = [end_seed],
        process_config = [process_config_json]
    )

    CSV.write(filename, vcat(existing_df, new_row))
    return false  # New row appended
end

end  # module GraphEpimodelsPersistenceExt
