"""
core/json.jl — a small, dependency-free JSON reader.

The package core deliberately carries no JSON dependency (CSV/DataFrames and
CairoMakie are the only weak deps). The country-graph bundles
(`graphs/geograph.jl`) and their GeoJSON basemaps are JSON, and the package only
ever *reads* them, so a compact recursive-descent parser is all that is needed —
no serializer, no schema validation beyond well-formedness.

`parse_json(str)` returns native Julia values:

| JSON          | Julia            |
|---------------|------------------|
| object        | `Dict{String,Any}` |
| array         | `Vector{Any}`    |
| string        | `String`         |
| integer       | `Int`            |
| real          | `Float64`        |
| `true`/`false`| `Bool`           |
| `null`        | `nothing`        |

Numbers without a fractional/exponent part parse to `Int`, otherwise `Float64`
(so node ids and populations come back as integers, coordinates as floats).

This is an internal utility (unexported); the CairoMakie extension reaches it as
`GraphEpimodels.parse_json` to read the basemap.
"""

# A cursor over the input's characters. We `collect` to a `Vector{Char}` so
# indexing is O(1) and Unicode (Norwegian place names: "Bø", "Tromsø", "Ålesund")
# is handled correctly without byte-offset bookkeeping. The files are small
# (kilobytes for a country bundle, tens of KB for a simplified basemap), so the
# one-time `collect` is negligible.
mutable struct _JSONCursor
    chars::Vector{Char}
    pos::Int
end

@inline _eof(c::_JSONCursor) = c.pos > length(c.chars)
@inline _peek(c::_JSONCursor) = c.chars[c.pos]

function _err(c::_JSONCursor, msg::AbstractString)
    throw(ArgumentError("Invalid JSON at position $(c.pos): $msg"))
end

function _skip_ws!(c::_JSONCursor)
    @inbounds while !_eof(c)
        ch = c.chars[c.pos]
        (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') || break
        c.pos += 1
    end
    return nothing
end

"""Parse a complete JSON document from `str`, erroring on trailing junk."""
function parse_json(str::AbstractString)
    c = _JSONCursor(collect(str), 1)
    _skip_ws!(c)
    _eof(c) && _err(c, "empty document")
    value = _parse_value!(c)
    _skip_ws!(c)
    _eof(c) || _err(c, "trailing characters after top-level value")
    return value
end

function _parse_value!(c::_JSONCursor)
    _skip_ws!(c)
    _eof(c) && _err(c, "unexpected end of input")
    ch = _peek(c)
    if ch == '{'
        return _parse_object!(c)
    elseif ch == '['
        return _parse_array!(c)
    elseif ch == '"'
        return _parse_string!(c)
    elseif ch == 't' || ch == 'f'
        return _parse_bool!(c)
    elseif ch == 'n'
        return _parse_null!(c)
    elseif ch == '-' || ('0' <= ch <= '9')
        return _parse_number!(c)
    else
        _err(c, "unexpected character '$ch'")
    end
end

function _expect!(c::_JSONCursor, ch::Char)
    (!_eof(c) && _peek(c) == ch) || _err(c, "expected '$ch'")
    c.pos += 1
    return nothing
end

function _parse_object!(c::_JSONCursor)::Dict{String,Any}
    _expect!(c, '{')
    obj = Dict{String,Any}()
    _skip_ws!(c)
    if !_eof(c) && _peek(c) == '}'
        c.pos += 1
        return obj
    end
    while true
        _skip_ws!(c)
        (!_eof(c) && _peek(c) == '"') || _err(c, "expected string key")
        key = _parse_string!(c)
        _skip_ws!(c)
        _expect!(c, ':')
        obj[key] = _parse_value!(c)
        _skip_ws!(c)
        _eof(c) && _err(c, "unterminated object")
        ch = _peek(c)
        if ch == ','
            c.pos += 1
        elseif ch == '}'
            c.pos += 1
            break
        else
            _err(c, "expected ',' or '}' in object")
        end
    end
    return obj
end

function _parse_array!(c::_JSONCursor)::Vector{Any}
    _expect!(c, '[')
    arr = Any[]
    _skip_ws!(c)
    if !_eof(c) && _peek(c) == ']'
        c.pos += 1
        return arr
    end
    while true
        push!(arr, _parse_value!(c))
        _skip_ws!(c)
        _eof(c) && _err(c, "unterminated array")
        ch = _peek(c)
        if ch == ','
            c.pos += 1
        elseif ch == ']'
            c.pos += 1
            break
        else
            _err(c, "expected ',' or ']' in array")
        end
    end
    return arr
end

function _parse_string!(c::_JSONCursor)::String
    _expect!(c, '"')
    io = IOBuffer()
    @inbounds while true
        _eof(c) && _err(c, "unterminated string")
        ch = c.chars[c.pos]
        c.pos += 1
        if ch == '"'
            break
        elseif ch == '\\'
            _eof(c) && _err(c, "unterminated escape")
            esc = c.chars[c.pos]
            c.pos += 1
            if esc == '"'
                print(io, '"')
            elseif esc == '\\'
                print(io, '\\')
            elseif esc == '/'
                print(io, '/')
            elseif esc == 'b'
                print(io, '\b')
            elseif esc == 'f'
                print(io, '\f')
            elseif esc == 'n'
                print(io, '\n')
            elseif esc == 'r'
                print(io, '\r')
            elseif esc == 't'
                print(io, '\t')
            elseif esc == 'u'
                print(io, _parse_unicode_escape!(c))
            else
                _err(c, "invalid escape '\\$esc'")
            end
        else
            print(io, ch)
        end
    end
    return String(take!(io))
end

# Parse the four hex digits following a `\u` escape into a Char. Surrogate pairs
# (`😀`) are combined when a low surrogate follows a high one.
function _parse_unicode_escape!(c::_JSONCursor)::Char
    code = _read_hex4!(c)
    if 0xD800 <= code <= 0xDBFF                      # high surrogate
        (c.pos + 1 <= length(c.chars) && c.chars[c.pos] == '\\' &&
         c.chars[c.pos + 1] == 'u') || _err(c, "expected low surrogate")
        c.pos += 2
        low = _read_hex4!(c)
        (0xDC00 <= low <= 0xDFFF) || _err(c, "invalid low surrogate")
        return Char(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00))
    end
    return Char(code)
end

function _read_hex4!(c::_JSONCursor)::UInt32
    (c.pos + 3 <= length(c.chars)) || _err(c, "truncated \\u escape")
    val = UInt32(0)
    @inbounds for _ in 1:4
        d = c.chars[c.pos]
        c.pos += 1
        digit = if '0' <= d <= '9'
            UInt32(d - '0')
        elseif 'a' <= d <= 'f'
            UInt32(d - 'a' + 10)
        elseif 'A' <= d <= 'F'
            UInt32(d - 'A' + 10)
        else
            _err(c, "invalid hex digit '$d'")
        end
        val = val * 16 + digit
    end
    return val
end

function _parse_number!(c::_JSONCursor)
    start = c.pos
    is_float = false
    @inbounds while !_eof(c)
        ch = c.chars[c.pos]
        if ch == '-' || ch == '+' || ('0' <= ch <= '9')
            c.pos += 1
        elseif ch == '.' || ch == 'e' || ch == 'E'
            is_float = true
            c.pos += 1
        else
            break
        end
    end
    token = String(c.chars[start:(c.pos - 1)])
    if is_float
        v = tryparse(Float64, token)
        v === nothing && _err(c, "malformed number '$token'")
        return v
    else
        iv = tryparse(Int, token)
        iv !== nothing && return iv
        fv = tryparse(Float64, token)          # very large integers fall back to Float64
        fv === nothing && _err(c, "malformed number '$token'")
        return fv
    end
end

function _parse_bool!(c::_JSONCursor)::Bool
    if _match_literal!(c, "true")
        return true
    elseif _match_literal!(c, "false")
        return false
    else
        _err(c, "invalid literal")
    end
end

function _parse_null!(c::_JSONCursor)
    _match_literal!(c, "null") || _err(c, "invalid literal")
    return nothing
end

function _match_literal!(c::_JSONCursor, lit::String)::Bool
    n = length(lit)
    (c.pos + n - 1 <= length(c.chars)) || return false
    @inbounds for (k, lc) in enumerate(lit)
        c.chars[c.pos + k - 1] == lc || return false
    end
    c.pos += n
    return true
end
