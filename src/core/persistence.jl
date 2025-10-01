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
        # Format the value appropriately
        value_str = if value isa Float64
            string(round(value, digits=10))  # Avoid floating point comparison issues
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
Append survival analysis result to CSV file.

Checks if an entry with the same configuration and seed range already exists.
If not, appends a new row. Creates the CSV file if it doesn't exist.

# Arguments
- `filename::String`: Path to CSV file
- `parameter_value::Float64`: The parameter value (e.g., λ) that was tested
- `num_simulations::Int`: Number of simulations run
- `num_survivals::Int`: Number of survivals observed
- `survival_probability::Float64`: Computed survival probability
- `std_error::Float64`: Standard error of survival probability
- `start_seed::Int`: First random seed used
- `end_seed::Int`: Last random seed used
- `process_info::Dict{String, Any}`: Process configuration from extract_process_info

# Returns
- `Bool`: true if row was appended, false if duplicate entry was skipped

# Example
```julia
zim = create_zim_simulation(100, 100, 2.0)
info = extract_process_info(zim)
append_survival_result("study.csv", 2.0, 1000, 450, 0.450, 0.016, 1000, 1999, info)
```
"""
function append_survival_result(
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
    
    # Create new row
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
    
    # Check if file exists
    if isfile(filename)
        # Read existing data
        existing_df = CSV.read(filename, DataFrame)
        
        # Check for duplicate entry (same config + overlapping seed range)
        for row in eachrow(existing_df)
            # Parse existing config
            existing_config = parse_process_info_json(row.process_config)
            
            # Check if configurations match
            if process_info_to_config_string(existing_config) == process_info_to_config_string(process_info) &&
               row.parameter == parameter_value
                
                # Check for seed range overlap
                if !(end_seed < row.start_seed || start_seed > row.end_seed)
                    @warn "Skipping duplicate entry: parameter=$parameter_value, seed range [$start_seed, $end_seed] overlaps with existing [$row.start_seed, $row.end_seed]"
                    return false
                end
            end
        end
        
        # No duplicate found, append to existing file
        CSV.write(filename, vcat(existing_df, new_row), append=false)
    else
        # Create new file
        CSV.write(filename, new_row)
    end
    
    return true
end

# =============================================================================
# Exports
# =============================================================================

export extract_process_info, process_info_to_config_string
export process_info_to_json, parse_process_info_json
export append_survival_result