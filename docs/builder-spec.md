# Country-graph builder — specification (hand-off)

This document specifies a **standalone tool** (intended for its own repository, in
Python) that builds *country-graph bundles* for
[GraphEpimodels.jl](https://github.com/smodee/GraphEpimodels.jl). It is written to
be handed to an implementer (human or agent) who does **not** have the
GraphEpimodels codebase in front of them.

The builder's only job: turn open geographic data into a bundle directory that
GraphEpimodels' `load_geograph` can read. The simulation package consumes that
bundle and runs epidemic models on it; the builder never touches Julia.

> **The contract is the file format, nothing else.** The builder and the
> simulator are decoupled: they agree only on the on-disk bundle format. The
> canonical, authoritative format spec lives in the GraphEpimodels repo at
> `docs/country-graph-format.md`. §3 below restates the essentials so this
> document stands alone, but if the two ever disagree, `country-graph-format.md`
> wins.

---

## 1. Goal and motivation

GraphEpimodels simulates epidemic processes (SIR, ZIM, etc.) on graphs. We want
**real geographic graphs**: nodes are settlements, edges are multi-modal transport
connections (roads, railways, ferries, flights). The example country is **Norway**,
but the tool must be country-agnostic.

The package ships one hand-authored mock of Norway (the bundle named
`norway_mock`) so the Julia features work today; this tool **replaces the mock with
real, reproducible data** and lets users generate their own countries. The real
output of this tool would be the bundle `norway` (no `_mock` suffix), which simply
sits alongside or supersedes the placeholder.

### What "good output" looks like

- Nodes: a few hundred to low-thousands of named settlements with population and
  `[lon, lat]`.
- Edges: per-layer connections that look like a schematic transport map — sparse,
  geographically sensible, no spaghetti. (We are **not** building a routing graph;
  we are building a settlement-level abstraction.)
- A simplified country outline as a GeoJSON basemap.

---

## 2. Scope

### MVP (build this first)

1. Nodes from a settlement gazetteer (population-filtered).
2. **Road** and **rail** layers from OpenStreetMap.
3. **Ferry** layer from OpenStreetMap (cheap — same source, different tag).
4. **Flight** layer from OpenFlights.
5. Simplified basemap GeoJSON (country outline).
6. A CLI that produces a valid bundle for a named country.
7. Output validation (the bundle must load without error — see §7).

### Later / non-MVP

- Per-edge attributes (distance, travel time) for future weighted models.
- Multiple admin levels / sub-country regions.
- A "projected coordinates" output mode (see §3.5).
- Automatic basemap simplification tuning per country.

### Non-goals

- No Julia code. No dependency on GraphEpimodels.
- Not a turn-by-turn routing engine.
- Not a perfectly accurate transport map — a *plausible schematic* is the target.

---

## 3. Output contract (the bundle)

A bundle is **one directory**, named with the country slug, containing:

```
<slug>/
├── geograph.json     # required
└── basemap.geojson   # optional but expected for MVP
```

### 3.1 `geograph.json`

A single JSON object:

```json
{
  "schema_version": 1,
  "name": "norway",
  "display_name": "Norway",
  "crs": "EPSG:4326",
  "bbox": [4.0, 24.5, 57.5, 71.5],
  "basemap": "basemap.geojson",
  "layers": [
    ["road",   "Roads"],
    ["rail",   "Railways"],
    ["ferry",  "Ferries"],
    ["flight", "Flights"]
  ],
  "nodes": [
    {"id": 1, "name": "Oslo", "lon": 10.7522, "lat": 59.9139, "population": 700000}
  ],
  "edges": [
    {"u": 1, "v": 7, "layer": "road"}
  ]
}
```

| Field | Meaning |
|-------|---------|
| `schema_version` | Integer, currently `1`. |
| `name` | Country slug; **must equal the directory name** (lowercase ASCII, no spaces). |
| `display_name` | Human-readable name. |
| `crs` | `"EPSG:4326"` (WGS84 lon/lat) for MVP. |
| `bbox` | `[lon_min, lon_max, lat_min, lat_max]`; required when a basemap is present. |
| `basemap` | Filename of the GeoJSON backdrop (relative), or `null`. |
| `layers` | **Ordered** `[symbol, label]` pairs; order = UI order. ≥ 1 layer. |
| `nodes` | Settlements. |
| `edges` | Undirected connections tagged by layer. |

**Node object:** `id` (integer, **contiguous 1..N, unique** — you must remap source
ids), `name` (string), `lon`/`lat` (numbers, WGS84 degrees), `population` (integer,
optional, default 0).

**Edge object:** `u`, `v` (integer node ids in 1..N, `u != v`), `layer` (string,
must match a declared layer symbol). Edges are undirected. The same `(u,v)` may
appear in several layers; duplicates are fine (the loader de-duplicates). Never
emit self-loops.

### 3.2 `basemap.geojson`

Standard GeoJSON (`FeatureCollection` or bare geometry) with `Polygon` /
`MultiPolygon` / `LineString` / `MultiLineString` geometries in `[lon, lat]`. Keep
it **simplified** (tens of KB; a few hundred–few thousand vertices). It should
roughly fill `bbox`.

### 3.3 Encoding & numeric notes

- UTF-8 throughout (Norwegian names: `Bø`, `Tromsø`, `Ålesund` are fine — the
  loader handles Unicode, but ASCII-folding names is also acceptable if simpler).
- Integers for `id`/`population`; floats for coordinates. Don't quote numbers.

### 3.4 Validation rules the bundle MUST satisfy

The loader throws if any of these fail; validate before writing (see §7):

1. `layers`, `nodes`, `edges` are arrays; ≥ 1 node and ≥ 1 layer.
2. Node ids are exactly `1..N` — unique, contiguous, no gaps.
3. Each node has numeric `lon`/`lat`; integer `population` if present.
4. Each edge `u`,`v` ∈ `1..N`, `u != v`; `layer` is a declared symbol.
5. If `basemap` set, `bbox` has four numbers.

### 3.5 Coordinate system

Store **raw WGS84 lon/lat** for nodes and basemap (no projection). The simulator
corrects the longitude/latitude aspect at draw time (it scales the plot box by
`cos(latitude)`), so no projection library is needed on either side. Keep the
projection choice configurable internally so a future `"projected"` mode (UTM/Web
Mercator metres) is easy to add — but do **not** pre-project for MVP.

---

## 4. Data sources

All sources are open and free. Cache raw downloads locally; builds should be
reproducible (pin source snapshots / record their dates in build metadata).

### 4.1 Nodes — settlement gazetteer (GeoNames)

[GeoNames](https://download.geonames.org/export/dump/) — use a per-country dump
(e.g. `NO.zip`) or the `cities*` tiers (`cities500`, `cities1000`, …).

- Keep feature **class `P`** (populated places).
- Filter by a **population threshold** (a CLI parameter; e.g. ≥ 5,000 for a
  ~hundreds-of-nodes graph). This is the main knob for graph size.
- De-duplicate near-coincident places (same town listed twice); keep the most
  populous.
- Fields you need: `name`, `latitude`, `longitude`, `population`. Keep the
  GeoNames id internally for traceability, but the bundle uses fresh `1..N` ids.

### 4.2 Roads & rail & ferries — OpenStreetMap

Two practical routes; pick one:

- **Geofabrik country extract** (`<country>-latest.osm.pbf`) + **pyrosm** or
  **osmium**/**pyosmium** — recommended for whole countries (fast, offline,
  filterable by tag).
- **OSMnx** (`osmnx`) — convenient for the road network of a place, but pulls
  intersection-level graphs that are large and need heavy contraction (§5). Fine
  for small countries / prototyping.

Tag filters:

| Layer  | OSM selector (typical) |
|--------|------------------------|
| road   | `highway in {motorway, trunk, primary, secondary}` (drop residential/service for a country-scale graph). |
| rail   | ways `railway = rail` (optionally `+ light_rail`); stations `railway = station`. |
| ferry  | ways/relations `route = ferry` (and `ferry = *`). |

### 4.3 Flights — OpenFlights

[OpenFlights](https://openflights.org/data.html): `airports.dat` (id, name, city,
country, IATA, lat, lon) and `routes.dat` (source/destination airport).

- Filter airports to the target country.
- Map each airport to the **nearest settlement node** (or to its `city` field
  matched against node names).
- An OpenFlights route between two airports ⇒ a `flight` edge between the two
  settlements those airports map to. Drop self-edges (two airports → same
  settlement) and de-duplicate.

---

## 5. The core problem: settlement-level edge contraction

The hard part is **roads and rail**: OSM gives a network whose nodes are
intersections, not settlements. You must contract this into edges *between
settlements*. This is the main algorithmic design decision; the simulator does not
care how you do it, only that the result is a sparse, sensible simple graph.

Recommended pipeline:

1. **Snap** each settlement to the nearest node of the (road / rail) network.
2. **Build settlement-to-settlement edges** using one of:

   - **"No intermediate settlement" rule (recommended).** Connect settlements `a`
     and `b` with an edge iff the shortest network path from `a` to `b` does not
     pass closer to some third settlement `c` than to both `a` and `b` — i.e. there
     is a *direct corridor* with no town in between. This yields a planar-ish,
     geographically meaningful graph (towns linked to their road neighbours, long
     edges only where the country is sparse). Implementable via a network Voronoi /
     nearest-settlement labelling of network nodes: an edge exists between `a` and
     `b` if their Voronoi cells are adjacent along the network.
   - **k-nearest-neighbours in *network* distance.** Simpler: connect each
     settlement to its `k` nearest settlements by shortest-path distance on the
     network (k ≈ 3–5). Cheap and robust; can produce a few odd long edges — prune
     by a max-distance cap.
   - **Gabriel / relative-neighbourhood graph** on settlement coordinates,
     **filtered** to keep only pairs that are actually connected on the network
     under a distance ratio. Good geometric fallback when routing is expensive.

3. **Prune**: drop edges longer than a configurable cap; ensure the largest
   component covers most settlements (report coverage).

Ferries and flights need little contraction: a ferry way already connects two
terminals → map terminals to nearest settlements; a flight route already connects
two airports → map to settlements (§4.3).

**Expose the rule and its parameters** (`k`, distance caps, the contraction method)
as CLI options — different countries will want different tunings. Record the chosen
parameters in a build-metadata file for reproducibility.

---

## 6. Basemap generation

Produce a simplified country outline as GeoJSON in lon/lat:

- Source: Natural Earth (admin-0 countries) or OSM administrative boundary
  (`admin_level = 2`), or the GADM country polygon.
- **Simplify** aggressively (Douglas–Peucker via `shapely.simplify`, or
  `mapshaper`/`topojson`) to tens of KB. Coastline fidelity beyond "recognizable
  country shape" is wasted — the package draws it as a light backdrop.
- Set the bundle's `bbox` to the polygon's bounds (optionally padded a few %).
- Save as `basemap.geojson` and reference it from `geograph.json`.

---

## 7. Output validation (required)

The tool must verify its own output is loadable before declaring success. Two
layers:

1. **Schema self-check (in the tool):** assert all rules in §3.4 — contiguous ids,
   valid edge endpoints, declared layers, no self-loops, bbox length, every node
   referenced has coordinates.
2. **Round-trip check (recommended, optional):** if a Julia toolchain is
   available in CI, load the bundle with GraphEpimodels and assert it builds and a
   trivial SIR runs:

   ```julia
   using GraphEpimodels
   g = load_geograph(:slug; dir = "out")
   @assert num_nodes(g) > 0
   run_simulation(create_sir_process(g, 3.0, 1.0; initial_infected = [1]))
   ```

   This is the ultimate contract test, but the tool should not *depend* on Julia —
   keep it an optional CI step.

Also report build stats: node count, per-layer edge counts, union edge count,
largest-component coverage, basemap vertex count, source snapshot dates.

---

## 8. CLI / UX

Suggested interface (adapt as needed):

```
country-graph-builder build \
    --country NO \                 # ISO code or name
    --slug norway \                # output folder / bundle name
    --display-name "Norway" \
    --min-population 5000 \        # node filter
    --layers road,rail,ferry,flight \
    --contraction no-intermediate \  # or knn / gabriel
    --knn 4 --max-edge-km 400 \
    --out ./out \
    --cache ./cache
```

- Each layer should be independently toggleable; the produced `layers` array must
  match what was actually built.
- Be idempotent and cache raw downloads (`--cache`).
- Emit a `build_meta.json` (sources, dates, parameters) alongside the bundle for
  reproducibility (this is *builder* metadata, not part of the bundle contract).

---

## 9. Suggested tech stack

- Python 3.11+.
- `pyrosm` or `pyosmium`/`osmium` for OSM PBF; or `osmnx` for the road network.
- `networkx` (or `scipy.sparse.csgraph`) for shortest paths / contraction.
- `geopandas` + `shapely` for geometry, snapping, simplification.
- `scipy.spatial` (KD-tree) for nearest-settlement snapping.
- `pandas` for the GeoNames / OpenFlights tables; `json` for output.

---

## 10. Repository layout (builder repo)

```
country-graph-builder/
├── README.md
├── pyproject.toml
├── src/country_graph_builder/
│   ├── nodes.py          # GeoNames -> settlement table
│   ├── osm.py            # OSM extract -> road/rail/ferry networks
│   ├── flights.py        # OpenFlights -> flight edges
│   ├── contract.py       # network -> settlement-level edges (§5)
│   ├── basemap.py        # outline -> simplified GeoJSON
│   ├── bundle.py         # assemble + validate + write geograph.json
│   └── cli.py
├── tests/
└── examples/             # e.g. a tiny fixture country for fast tests
```

Vendor a copy of (or link to) `country-graph-format.md` so the contract travels
with the builder.

---

## 11. Acceptance criteria

The tool is "done" for MVP when:

1. `build --country NO --slug norway ...` produces `out/norway/{geograph.json,
   basemap.geojson}`.
2. The bundle passes the §7 schema self-check.
3. GraphEpimodels' `load_geograph(:norway; dir="out")` loads it, `with_layers`
   selects any subset, and a SIR run spreads beyond the seed.
4. The four layers each look sensible when plotted (no spaghetti; largest
   component covers the bulk of settlements).
5. Re-running with the same inputs/parameters yields an identical bundle.

---

## 12. Open decisions for the implementer

- **Contraction method default** — start with `no-intermediate`; validate on Norway
  and pick whatever yields the cleanest map.
- **Population threshold default** — pick so Norway lands in the few-hundred-node
  range; expose as a knob.
- **Airport→settlement mapping** — nearest-node vs `city`-name match; nearest-node
  is more robust.
- **Name encoding** — keep Unicode, or ASCII-fold for portability. (The simulator
  accepts both.)
```
