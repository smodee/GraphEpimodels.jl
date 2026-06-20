# Country-graph bundle format (v1)

This is the **canonical, authoritative** specification of the on-disk format that
`GraphEpimodels.load_geograph` reads. Anything that produces country graphs — in
particular the separate builder tool — must conform to this document. The loader
in [`src/graphs/geograph.jl`](../src/graphs/geograph.jl) is the reference
implementation; if the two ever disagree, the loader wins and this document is the
bug.

The format is deliberately **plain JSON + GeoJSON**, so that:

- the Julia package can read a bundled example with **no extra dependencies**
  (the core has no CSV/JSON package dependency — it ships a tiny JSON reader);
- bundles are human-inspectable and diff-friendly in git;
- any producer (Python/pandas `to_dict` → `json.dump`, or anything else) can emit
  them trivially.

---

## 1. Directory layout

One bundle = one directory, named after the graph, under a *countries directory*
(the package ships its examples in `data/countries/`):

```
data/countries/
└── norway_mock/
    ├── geograph.json     # required — the graph bundle (see §2)
    └── basemap.geojson   # optional — geographic backdrop (see §3)
```

- The directory name (`norway_mock`) is the bundle's **name**; it is what
  `load_geograph(:norway_mock)` and the explorer dropdown use. Use a lowercase,
  ASCII, no-spaces slug.
- A directory is "a bundle" iff it contains a `geograph.json`. `available_country_graphs()`
  discovers bundles by scanning for that file.
- The basemap file may have any name; `geograph.json` points to it by name.

Users can drop additional bundle directories alongside the shipped examples (or
point `load_geograph(...; dir=...)` at their own countries directory).

---

## 2. `geograph.json`

A single JSON **object** with the following members.

### 2.1 Top-level fields

| Field            | Type                    | Required | Description |
|------------------|-------------------------|----------|-------------|
| `schema_version` | integer                 | yes      | Format version. Currently **`1`**. |
| `name`           | string                  | yes      | Bundle slug; should equal the directory name. |
| `display_name`   | string                  | yes      | Human-readable name (e.g. `"Norway"`). Shown in titles/UI. |
| `crs`            | string                  | yes      | Coordinate reference system. **MVP requires `"EPSG:4326"`** (WGS84 lon/lat degrees). See §4. |
| `bbox`           | `[number,number,number,number]` | required if `basemap` is set | `[lon_min, lon_max, lat_min, lat_max]` — the geographic extent the map is framed to. |
| `basemap`        | string or `null`        | no       | Filename (relative to the bundle dir) of a GeoJSON backdrop, or `null`/omitted for none. |
| `layers`         | array of `[string,string]` | yes   | Edge-layer declarations, **in display order** (see §2.2). |
| `nodes`          | array of objects        | yes      | Settlements (see §2.3). |
| `edges`          | array of objects        | yes      | Connections, tagged by layer (see §2.4). |

### 2.2 `layers`

An **ordered** array of `[symbol, label]` pairs. The `symbol` is the stable
machine key used by edges and the API (e.g. `"road"`); the `label` is the
human-readable name shown as a checkbox in the explorer (e.g. `"Roads"`). Order is
preserved and defines the UI order and the canonical ordering of "active layers".

```json
"layers": [
  ["road",   "Roads"],
  ["rail",   "Railways"],
  ["ferry",  "Ferries"],
  ["flight", "Flights"]
]
```

- At least one layer is required.
- Layer symbols must be unique.

### 2.3 `nodes`

An array of settlement objects:

| Field        | Type    | Required | Notes |
|--------------|---------|----------|-------|
| `id`         | integer | yes      | **Contiguous 1-based** node id. Ids must be exactly `1..N` with no gaps or duplicates. |
| `name`       | string  | yes      | Settlement name (e.g. `"Oslo"`). Used for name-based seeding (`find_node`). |
| `lon`        | number  | yes      | Longitude, WGS84 degrees. |
| `lat`        | number  | yes      | Latitude, WGS84 degrees. |
| `population` | integer | no       | Settlement population (default `0`). Used for the default "seed largest city" and, in future, for marker sizing. |

```json
{"id": 1, "name": "Oslo", "lon": 10.7522, "lat": 59.9139, "population": 700000}
```

> **Why contiguous 1-based ids?** The simulation engine indexes nodes `1..N`
> directly. The builder must remap whatever source ids it uses (GeoNames ids, OSM
> node ids, …) onto `1..N` before writing the bundle.

### 2.4 `edges`

An array of edge objects. Edges are **undirected**.

| Field   | Type    | Required | Notes |
|---------|---------|----------|-------|
| `u`     | integer | yes      | Endpoint node id, in `1..N`. |
| `v`     | integer | yes      | Endpoint node id, in `1..N`. |
| `layer` | string  | yes      | Must equal one of the declared layer symbols. |

```json
{"u": 1, "v": 7, "layer": "road"}
```

De-duplication and merging are handled by the loader:

- The same `(u, v)` may appear in **multiple layers** (e.g. Oslo–Bergen by both
  rail and flight). When several layers are activated, shared edges collapse to a
  single edge in the simulated graph.
- Duplicates of `(u, v)` *within* one layer are tolerated (collapsed).
- `(u, u)` self-loops are **rejected** — do not emit them.

A node that has no edges in any active layer is allowed (it simply participates as
an isolated vertex — realistic, e.g. a town with no railway).

### 2.5 Validation rules (enforced by the loader)

Producing a bundle that violates any of these makes `load_geograph` throw. The
builder should validate before writing:

1. Root is a JSON object; `layers`, `nodes`, `edges` are arrays.
2. At least one node and at least one layer.
3. Node ids cover `1..N` exactly — unique, contiguous, no gaps.
4. Every node has numeric `lon`/`lat`; `population` (if present) is an integer.
5. Every edge `u`,`v` is an integer in `1..N`; `u != v`.
6. Every edge `layer` is one of the declared layer symbols.
7. If `basemap` is set, `bbox` is present and has exactly four numbers.

---

## 3. `basemap.geojson`

A standard [GeoJSON](https://datatracker.ietf.org/doc/html/rfc7946) file giving a
geographic backdrop (coastline / administrative outline) drawn behind the
node-link diagram.

- Either a `FeatureCollection` (each feature a geometry) or a bare geometry object.
- Geometry types used: **`Polygon`**, `MultiPolygon`, `LineString`,
  `MultiLineString`. `Point`/`MultiPoint` are ignored. Other members are ignored.
- Coordinates are `[lon, lat]` in WGS84 degrees (GeoJSON's mandated CRS), the
  **same coordinate system as the nodes** — this is the whole reason the format is
  lon/lat (see §4).
- Keep it **simplified**: a few hundred to a few thousand vertices is plenty for a
  schematic country map. The package draws filled land + a coastline stroke. Aim
  for a small, git-friendly file (tens of KB). Simplify with e.g. `shapely`'s
  `simplify`, `mapshaper`, or `topojson` → GeoJSON.
- The outline should roughly fill the declared `bbox`.

The package only ever **reads** the basemap, and only when something is actually
rendered (it is parsed lazily by the CairoMakie extension), so it never affects
simulation or the dependency-free core.

---

## 4. Coordinate system and projection

**MVP stores raw lon/lat (WGS84, `EPSG:4326`) everywhere** — nodes and basemap
alike — and does **not** pre-project.

At a country's latitude, one degree of longitude covers less ground than one
degree of latitude (by `cos(latitude)`), so plotting lon/lat directly would
squash the map east–west. The visualization layer corrects this by giving the plot
box the aspect ratio `(Δlon · cos lat₀) / Δlat` and framing to `bbox`; nodes,
edges and coastline all live in one lon/lat space and end up in correct
proportion. **No projection library is needed on either side.**

This is an approximation (equirectangular + aspect correction), which is fine for
a country-scale schematic. A future schema version may add an optional
`"projected"` coordinate mode (pre-projected `x`/`y` in metres, e.g. UTM/Web
Mercator) for finer fidelity; if so, the basemap would need to be in the matching
projection. The builder should keep its projection choice configurable to make
that easy later.

---

## 5. Complete minimal example

```json
{
  "schema_version": 1,
  "name": "example",
  "display_name": "Example",
  "crs": "EPSG:4326",
  "bbox": [9.0, 12.0, 59.0, 64.0],
  "basemap": "basemap.geojson",
  "layers": [
    ["road",   "Roads"],
    ["flight", "Flights"]
  ],
  "nodes": [
    {"id": 1, "name": "Aville", "lon": 10.75, "lat": 59.91, "population": 500000},
    {"id": 2, "name": "Beburg", "lon": 10.40, "lat": 63.43, "population": 200000},
    {"id": 3, "name": "Cetown", "lon":  9.20, "lat": 61.10, "population":  40000}
  ],
  "edges": [
    {"u": 1, "v": 3, "layer": "road"},
    {"u": 3, "v": 2, "layer": "road"},
    {"u": 1, "v": 2, "layer": "flight"}
  ]
}
```

See [`data/countries/norway_mock/`](../data/countries/norway_mock/) for the shipped
example (a hand-authored mock pending the builder — hence the `_mock` suffix).

---

## 6. Consuming a bundle (Julia)

```julia
using GraphEpimodels

available_country_graphs()          # ["norway_mock", ...]  — discovered bundles
country_edge_sets(:norway_mock)     # [(:road,"Roads"), (:rail,"Railways"), ...]

g  = load_geograph(:norway_mock)                       # all layers active
gr = load_geograph(:norway_mock; edges = [:road, :rail])
gr = with_layers(g, [:flight])                    # re-select cheaply, no re-read

# Then simulate exactly as on any other graph:
sir = create_sir_process(g, 3.0, 1.0; initial_infected = [find_node(g, "Oslo")])
run_simulation(sir)
```
