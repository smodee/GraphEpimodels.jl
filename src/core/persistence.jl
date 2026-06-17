"""
core/persistence.jl

Functions for serializing epidemic process configurations and managing
CSV-based result storage for survival probability analyses.

Provides process introspection, JSON serialization, and utilities for
reading/writing analysis results to CSV files.

The CSV-backed result store (`get_next_start_seed`,
`update_or_append_survival_result`) is implemented in the package extension
ext/GraphEpimodelsPersistenceExt.jl, which loads only when the user brings in
`CSV` and `DataFrames`. The process-introspection and JSON helpers below need
neither and stay in the package.
"""

# =============================================================================
# Process Information Extraction
# =============================================================================

# Serialization descriptors. Rather than poke at process/graph fields with
# `hasfield`/`isa` (which silently misses any model whose rate fields aren't named
# λ/μ — e.g. SIR's β/γ, Maki-Thompson's α/β), each type advertises the parameters
# it wants serialized via these two small dispatch points. The dict keys become
# part of the reproducibility config string, so they must stay stable.

"""
Model-specific parameters to serialize for `process`, as a `String => value` dict.
Default: none; each concrete process overrides this to advertise its rate
parameters.
"""
model_parameters(::AbstractEpidemicProcess)::Dict{String, Any} = Dict{String, Any}()
model_parameters(p::ZIMProcess)::Dict{String, Any} =
    Dict{String, Any}("lambda" => p.λ, "mu" => p.μ)
model_parameters(p::SIRProcess)::Dict{String, Any} =
    Dict{String, Any}("beta" => p.β, "gamma" => p.γ)
model_parameters(p::MakiThompsonProcess)::Dict{String, Any} =
    Dict{String, Any}("alpha" => p.α, "beta" => p.β, "stifler_contact" => p.stifler_contact)
model_parameters(p::ChaseEscapeProcess)::Dict{String, Any} =
    Dict{String, Any}("lambda" => p.λ, "mu" => p.μ, "ghost" => p.ghost)

"""
Graph-shape parameters to serialize for `graph`, as a `String => value` dict.
Default: none; lattices override to record their dimensions and boundary.
"""
graph_descriptor(::AbstractEpidemicGraph)::Dict{String, Any} = Dict{String, Any}()
# 2D hypercubic (SquareLattice): width/height/boundary, keeping the config string
# identical to the pre-refactor field-poking version.
graph_descriptor(lattice::HypercubicLattice{2})::Dict{String, Any} =
    Dict{String, Any}("width" => lattice.dims[1], "height" => lattice.dims[2],
                      "boundary" => string(lattice.boundary))
function graph_descriptor(lattice::HypercubicLattice)::Dict{String, Any}
    d = Dict{String, Any}("boundary" => string(lattice.boundary))
    for (i, s) in enumerate(lattice.dims)
        d["dim$i"] = s
    end
    return d
end
graph_descriptor(lattice::Union{TriangularLattice, HexagonalLattice})::Dict{String, Any} =
    Dict{String, Any}("width" => lattice.width, "height" => lattice.height,
                      "boundary" => string(lattice.boundary))

"""
Extract comprehensive process and graph information for serialization.

Captures the process type, the model's rate parameters ([`model_parameters`](@ref)),
the graph type and shape ([`graph_descriptor`](@ref)), and basic graph properties,
as a dictionary suitable for JSON serialization and reproducibility config strings.

# Arguments
- `process::AbstractEpidemicProcess`: The process to extract information from

# Returns
- `Dict{String, Any}`: Dictionary containing process type, parameters, and graph info

# Example
```julia
zim = create_zim_process(100, 100, 2.0, 1.0)
info = extract_process_info(zim)
# Returns: Dict("process_type" => "ZIMProcess", "lambda" => 2.0, ...)
```
"""
function extract_process_info(process::AbstractEpidemicProcess)::Dict{String, Any}
    info = Dict{String, Any}("process_type" => string(nameof(typeof(process))))
    merge!(info, model_parameters(process))

    graph = get_graph(process)
    info["graph_type"] = string(nameof(typeof(graph)))
    merge!(info, graph_descriptor(graph))

    info["num_nodes"]    = num_nodes(graph)
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
# CSV Persistence Functions (implemented in the persistence extension)
# =============================================================================
#
# `get_next_start_seed` and `update_or_append_survival_result` read/write the CSV
# survival-result store; their implementations live in
# ext/GraphEpimodelsPersistenceExt.jl, which loads with `using CSV, DataFrames`.
# They are declared here (as generic functions) so the package can export them
# and the extension can add the concrete methods. Without CSV/DataFrames loaded,
# only these fallbacks exist and they give an actionable error.

const _CSV_HINT = "requires CSV and DataFrames. Run `using CSV, DataFrames` to enable CSV result persistence."

get_next_start_seed(args...; kwargs...) =
    error("get_next_start_seed $_CSV_HINT")
update_or_append_survival_result(args...; kwargs...) =
    error("update_or_append_survival_result $_CSV_HINT")

# Batched in-memory survival-result store used by `run_parameter_sweep`. The
# single-row `get_next_start_seed` / `update_or_append_survival_result` above each
# re-read (and re-parse) the whole CSV — and the latter rewrites it — *per
# parameter*, so a P-parameter sweep does O(P) reads + O(P) full rewrites and
# re-parses every existing row O(P) times. These four functions instead load the
# file once (`load_survival_results`), answer seed lookups and record results
# against the in-memory store (`next_start_seed`, `record_survival_result!`), and
# write the file once at the end (`save_survival_results`). Implemented in the
# persistence extension; unexported (internal sweep machinery). The single-row
# public functions are kept for direct/ad-hoc use.
load_survival_results(args...; kwargs...) =
    error("load_survival_results $_CSV_HINT")
next_start_seed(args...; kwargs...) =
    error("next_start_seed $_CSV_HINT")
record_survival_result!(args...; kwargs...) =
    error("record_survival_result! $_CSV_HINT")
save_survival_results(args...; kwargs...) =
    error("save_survival_results $_CSV_HINT")

# =============================================================================
# Exports
# =============================================================================

export extract_process_info, process_info_to_config_string
export process_info_to_json, parse_process_info_json
