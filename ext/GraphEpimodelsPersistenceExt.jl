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
