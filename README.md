# GraphEpimodels.jl

Fast, extensible Julia library for simulating epidemic processes on graphs.

Implements four epidemic models — ZIM, SIR, Maki-Thompson, and Chase-Escape — on a rich collection of graph types, with exact Gillespie scheduling, 2D/3D visualization, animation, and parallel Monte Carlo analysis. Based on research by Bethuelsen, Broman & Modée (2024).

---

## Models

| Type | Constructor | Description |
|------|-------------|-------------|
| `ZIMProcess` | `create_zim_process` | Zombie Infection Model |
| `SIRProcess` | `create_sir_process` | SIR epidemic model |
| `MakiThompsonProcess` | `create_maki_thompson_process` | Rumor-spreading model |
| `ChaseEscapeProcess` | `create_chase_escape_process` | Predator–prey chase-escape model |

All processes share a common interface: `step!`, `reset!`, `run_simulation`, `is_active`, `has_escaped`, `get_statistics`.

---

## Graph Types

### Lattices (`AbstractLatticeGraph`)

Implicit graphs — no adjacency lists are stored; neighbors are computed by O(1) coordinate arithmetic. Memory is O(n) (state vector only).

| Type | Alias / Constructor | Description |
|------|---------------------|-------------|
| `HypercubicLattice{2}` | `SquareLattice` / `create_square_lattice` | 2D square lattice (4 neighbors) |
| `HypercubicLattice{3}` | `CubeLattice` / `create_cube_lattice` | 3D cubic lattice (6 neighbors) |
| `HypercubicLattice{D}` | `create_hypercubic_lattice` | Arbitrary-dimension lattice; `D` is a type parameter |
| `TriangularLattice` | `create_triangular_lattice` | 2D triangular lattice (6 neighbors) |
| `HexagonalLattice` | `create_hexagonal_lattice` | 2D honeycomb lattice (3 neighbors) |

Boundary conditions: `ABSORBING` (default) or `PERIODIC` (torus via `create_torus`).

### Structured graphs (`AbstractImplicitGraph`)

Implicit like lattices — only the state vector is stored.

| Type | Constructor | Description |
|------|-------------|-------------|
| `CompleteGraph` | `create_complete_graph` | Every pair connected |
| `PathGraph` | `create_path_graph` | Linear chain |
| `CycleGraph` | `create_cycle_graph` | Ring |
| `StarGraph` | `create_star_graph` | Hub-and-spoke |
| `RegularTree` | `create_regular_tree` / `create_dary_tree` | Cayley tree / balanced *d*-ary tree |

`create_regular_tree(d, height)` gives the graph-theory regular tree (every internal node has degree *d*; branching ratio *d* − 1). `create_dary_tree(k, height)` gives the CS convention (*k* children per internal node, including root).

### General graphs (`AdjacencyGraph`)

Explicit adjacency-list representation for arbitrary topologies.

| Constructor | Description |
|-------------|-------------|
| `create_graph_from_matrix` | From adjacency matrix |
| `create_graph_from_edges` | From edge list |
| `create_erdos_renyi` / `create_gnp` / `create_gnm` | Erdős–Rényi random graphs |

---

## Quick Start

```julia
using GraphEpimodels

# ZIM on a 100×100 square lattice
zim = create_zim_process(100, 100, 2.0)   # λ = 2.0
results = run_simulation(zim; max_time=50.0)
println("Escaped: ", has_escaped(zim))

# SIR on a regular tree (Cayley tree, degree 4, height 8)
tree = create_regular_tree(4, 8)
sir = create_sir_process(tree, 0.6, 1.0)
run_simulation(sir)

# Erdős–Rényi graph
er = create_erdos_renyi(500, 0.01)
```

---

## Visualization

Visualization requires [CairoMakie](https://github.com/MakieOrg/Makie.jl) (loaded as a package extension — the core package stays lightweight without it).

```julia
using GraphEpimodels, CairoMakie

# Static snapshot — lattice heatmap
sir = create_sir_process(50, 50, 0.6, 1.0; initial_infected=:center)
run_simulation(sir)
fig = visualize_state(create_auto_visualizer(sir), sir)
save_plot("sir_final.png", fig)

# Animate every transition (small lattice)
sir = create_sir_process(30, 30, 0.6, 1.0; initial_infected=:center, rng_seed=1)
animate_simulation(sir; sampler=EveryStep(), color_scheme=:sir, filename="sir.gif")

# Equal-time sampling (large lattice — faithful temporal playback)
big = create_sir_process(200, 200, 0.6, 1.0; initial_infected=:center, rng_seed=1)
animate_simulation(big; sampler=TimeInterval(0.5), max_time=40.0, filename="sir_large.mp4")
```

**Visualizer dispatch** (`visualizer_for` / `create_auto_visualizer`) picks the right visualizer automatically:
- Lattices → `LatticeVisualizer` (dual-tiling cells; square cells for `SquareLattice`, hexagonal cells for `TriangularLattice`, triangular cells for `HexagonalLattice`)
- General graphs → `NetworkVisualizer` (node-link diagram)

**Layout dimensions:** `SquareLattice`, `TriangularLattice`, `HexagonalLattice` have 2D layouts; `CubeLattice`, `RegularTree`, `StarGraph`, `CompleteGraph` have both 2D and 3D closed-form layouts. Higher-dimensional `HypercubicLattice{D}` (D ≥ 4) falls back to a computed layout.

---

## Survival Analysis

Parallel Monte Carlo estimation of survival probabilities, using Julia threads.

```julia
# Start Julia with multiple threads for parallel analysis:
#   julia --threads=4

using GraphEpimodels

# Sweep λ over a range, 1000 simulations each, on a 100×100 lattice
λ_values = 1.0:0.1:3.0
results = run_zim_lattice_survival_analysis(λ_values, 100, 100; num_simulations=1000)

# Lower-level: estimate survival probability for a single parameter
p = estimate_survival_probability(zim_process; num_simulations=500,
                                  criterion=EscapeCriterion(),
                                  mode=DETAILED)

check_threading_setup()   # confirm thread count
```

**Survival criteria:** `EscapeCriterion` (reached boundary), `PersistenceCriterion` (still active at end), `ThresholdCriterion(k)` (cluster size ≥ k).

---

## Persistence

CSV/JSON serialization requires `CSV` and `DataFrames` (loaded as an extension).

```julia
using GraphEpimodels, CSV, DataFrames

info = extract_process_info(process)
config_str = process_info_to_config_string(info)
json_str = process_info_to_json(info)

# Append/update a survival result in a CSV file
update_or_append_survival_result("results.csv", params, survival_prob)
```

---

## Threading

```julia
# Check available threads
check_threading_setup()
get_recommended_threads()

# The Gillespie loop is single-threaded per simulation;
# parallelism comes from running independent replicas concurrently.
# Start Julia with --threads=N (N ≥ 4 recommended for sweep analysis).
```
