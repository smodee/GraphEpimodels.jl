"""
core/persistence.jl

Functions for serializing epidemic process configurations and managing
CSV-based result storage for survival probability analyses.

Provides process introspection, JSON serialization, and utilities for
reading/writing analysis results to CSV files.
"""

using CSV, DataFrames

# =============================================================================
# Process Information Extraction
# =============================================================================

"""
Extract comprehensive process and graph information for serialization.

Introspects an epidemic process to capture all relevant parameters for
reproducibility and identification. Returns a dictionary suitable for
JSON serialization.

# Arguments
- `process::AbstractEpidemicProcess`: The process to extract information from

# Returns
- `Dict{String, Any}`: Dictionary containing process type, parameters, and graph info

# Example
```julia
zim = create_zim_simulation(100, 100, 2.0, 1.0)
info = extract_process_info(zim)
# Returns: Dict("process_type" => "ZIMProcess", "lambda" => 2.0, ...)
```
"""
function extract_process_info(process::AbstractEpidemicProcess)::Dict{String, Any}
    info = Dict{String, Any}()
    
    # Basic process information
    info["process_type"] = string(nameof(typeof(process)))
    
    # Extract process-specific parameters
    # Check for common epidemic process parameters
    if hasfield(typeof(process), :λ)
        info["lambda"] = process.λ
    end
    
    if hasfield(typeof(process), :μ)
        info["mu"] = process.μ
    end
    
    # Extract graph information
    graph = get_graph(process)
    info["graph_type"] = string(nameof(typeof(graph)))
    
    # Graph-specific parameters
    if hasfield(typeof(graph), :width)
        info["width"] = graph.width
    end
    
    if hasfield(typeof(graph), :height)
        info["height"] = graph.height
    end
    
    if hasfield(typeof(graph), :boundary)
        info["boundary"] = string(graph.boundary)
    end
    
    # General graph properties
    info["num_nodes"] = num_nodes(graph)
    
    # Additional useful metadata
    info["has_boundary"] = has_boundary(graph)
    
    return info
end

"""
Create a compact configuration string from process info for comparison.

Used to check if two processes have the same configuration.
Excludes num_nodes and has_boundary as these are derived properties.

# Arguments
- `info::Dict{String, Any}`: Process info from extract_process_info

# Returns
- `String`: Compact string representation for comparison

# Example
```julia
config_str = process_info_to_config_string(info)
# Returns: "ZIMProcess_lambda=2.0_mu=1.0_SquareLattice_width=100_height=100_boundary=ABSORBING"
```
"""
function process_info_to_config_string(info::Dict{String, Any})::String
    # Sort keys for consistent ordering (exclude derived properties)
    exclude_keys = Set(["num_nodes", "has_boundary"])
    relevant_keys = sort([k for k in keys(info) if !(k in exclude_keys)])
    
    parts = String[]
    for key in relevant_keys
        value = info[key]
        # Format the value appropriately - normalize all numbers to Float64
        value_str = if value isa Number
            string(round(Float64(value), digits=10))
        else
            string(value)
        end
        push!(parts, "$(key)=$(value_str)")
    end
    
    return join(parts, "_")
end

# =============================================================================
# JSON Serialization (Lightweight Implementation)
# =============================================================================

"""
Convert process info dictionary to JSON string for CSV storage.

# Arguments
- `info::Dict{String, Any}`: Process info from extract_process_info

# Returns
- `String`: JSON string representation

# Example
```julia
json_str = process_info_to_json(info)
# Returns: "{\"process_type\":\"ZIMProcess\",\"lambda\":2.0,...}"
```
"""
function process_info_to_json(info::Dict{String, Any})::String
    # Simple JSON serialization without external dependencies
    # Sort by keys only (not by pairs) to avoid comparing mixed-type values
    pairs = String[]
    for key in sort(collect(keys(info)))
        value = info[key]
        json_value = if value isa String
            "\"$(value)\""
        elseif value isa Number
            string(value)
        elseif value isa Bool
            value ? "true" : "false"
        else
            "\"$(string(value))\""
        end
        push!(pairs, "\"$(key)\":$(json_value)")
    end
    return "{" * join(pairs, ",") * "}"
end

"""
Parse JSON string back to dictionary.

# Arguments
- `json_str::String`: JSON string from CSV

# Returns
- `Dict{String, Any}`: Parsed dictionary

# Example
```julia
info = parse_process_info_json(json_str)
```
"""
function parse_process_info_json(json_str::String)::Dict{String, Any}
    # Simple JSON parsing for the specific format we generate
    # Remove outer braces and convert to String (strip returns SubString)
    content = String(strip(json_str, ['{', '}']))
    
    info = Dict{String, Any}()
    
    # Split by commas (not inside quotes)
    pairs = split_json_pairs(content)
    
    for pair in pairs
        # Split key:value
        key_val = split(pair, ":", limit=2)
        if length(key_val) != 2
            continue
        end
        
        key = strip(key_val[1], ['"', ' '])
        value_str = strip(key_val[2], [' '])
        
        # Parse value
        value = if startswith(value_str, "\"")
            strip(value_str, ['"'])
        elseif value_str == "true"
            true
        elseif value_str == "false"
            false
        else
            # Try to parse as number
            try
                parse(Float64, value_str)
            catch
                value_str
            end
        end
        
        info[key] = value
    end
    
    return info
end

"""
Helper function to split JSON pairs handling nested quotes.
"""
function split_json_pairs(content::String)::Vector{String}
    pairs = String[]
    current = ""
    in_quotes = false
    
    for c in content
        if c == '"'
            in_quotes = !in_quotes
            current *= c
        elseif c == ',' && !in_quotes
            push!(pairs, current)
            current = ""
        else
            current *= c
        end
    end
    
    if !isempty(current)
        push!(pairs, current)
    end
    
    return pairs
end

# =============================================================================
# CSV Persistence Functions
# =============================================================================

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

# Example
```julia
zim = create_zim_simulation(100, 100, 2.0)
info = extract_process_info(zim)
start_seed = get_next_start_seed("study.csv", 2.0, info)
# Returns: 1 if no previous runs, or 1001 if previous run ended at seed 1000
```
"""
function get_next_start_seed(
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

# Example
```julia
zim = create_zim_simulation(100, 100, 2.0)
info = extract_process_info(zim)
was_updated = update_or_append_survival_result("study.csv", 2.0, 1000, 450, 0.450, 0.016, 1, 1000, info)
# If was_updated == true: extended existing entry
# If was_updated == false: created new entry
```
"""
function update_or_append_survival_result(
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

# =============================================================================
# Exports
# =============================================================================

export extract_process_info, process_info_to_config_string
export process_info_to_json, parse_process_info_json
export get_next_start_seed, update_or_append_survival_result