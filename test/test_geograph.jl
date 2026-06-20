using Test
using GraphEpimodels
using Random

@testset "geograph" begin

    # -------------------------------------------------------------------------
    # JSON reader (dependency-free; underpins the bundle + basemap loaders)
    # -------------------------------------------------------------------------
    @testset "json reader" begin
        pj = GraphEpimodels.parse_json
        d = pj("""{"a":1,"b":2.5,"c":"x","d":true,"e":null,"f":[1,2,3]}""")
        @test d isa Dict{String,Any}
        @test d["a"] === 1            # integer stays Int
        @test d["b"] === 2.5          # real becomes Float64
        @test d["c"] == "x"
        @test d["d"] === true
        @test d["e"] === nothing
        @test d["f"] == [1, 2, 3]

        @test pj("[[1,2],[3,4]]") == [[1, 2], [3, 4]]
        @test pj("-3.0e2") == -300.0
        @test pj("  \n  42 \t ") === 42

        # Unicode: both a \u escape and raw UTF-8 round-trip.
        @test pj("\"Troms\\u00f8\"") == "Tromsø"
        @test pj("\"Bø\"") == "Bø"
        @test pj("\"a\\nb\\tc\"") == "a\nb\tc"

        @test_throws ArgumentError pj("{bad}")
        @test_throws ArgumentError pj("[1,2,")
        @test_throws ArgumentError pj("{\"a\":1} junk")
    end

    # -------------------------------------------------------------------------
    # Registry / discovery
    # -------------------------------------------------------------------------
    @testset "registry" begin
        @test "norway_mock" in available_country_graphs()
        sets = country_edge_sets(:norway_mock)
        @test sets == [(:road, "Roads"), (:rail, "Railways"),
                       (:ferry, "Ferries"), (:flight, "Flights")]
        @test_throws ArgumentError country_edge_sets(:atlantis)
    end

    # -------------------------------------------------------------------------
    # Loading + node table
    # -------------------------------------------------------------------------
    @testset "load + nodes" begin
        g = load_geograph(:norway_mock)
        @test g isa GeoGraph
        @test num_nodes(g) == 15

        # Names + populations + name lookup (case-insensitive).
        @test node_name(g, 1) == "Oslo"
        @test node_population(g, 1) == 700000
        @test find_node(g, "oslo") == 1
        @test find_node(g, "TROMSO") == 5
        @test find_node(g, "nowhere") === nothing
        @test largest_settlement(g) == 1          # Oslo is most populous

        # Geometry: geographic coordinates are a 2D layout.
        @test supported_layout_dims(g) == (2,)
        @test layout_dim(g) == 2
        @test size(node_positions(g)) == (2, 15)
        @test node_positions(g)[:, 1] ≈ [10.7522, 59.9139]
        @test_throws ArgumentError node_positions(g; dim = 3)

        # Basemap handle present and resolvable.
        @test has_basemap(g)
        bm = basemap(g)
        @test bm isa Basemap
        @test isfile(bm.path)
        @test bm.bbox == (4.0, 24.5, 57.5, 71.5)
    end

    # -------------------------------------------------------------------------
    # Edge layers + selection
    # -------------------------------------------------------------------------
    @testset "layers" begin
        g = load_geograph(:norway_mock)
        @test available_layers(g) == [:road, :rail, :ferry, :flight]
        @test active_layers(g) == [:road, :rail, :ferry, :flight]   # :all by default
        @test layer_label(g, :rail) == "Railways"

        # Single-layer subsets have exactly that layer's edge count (no dups within
        # a layer); the union is smaller than the naive sum (shared edges collapse).
        @test num_edges(with_layers(g, [:road])) == 16
        @test num_edges(with_layers(g, [:rail])) == 10
        @test num_edges(with_layers(g, [:ferry])) == 6
        @test num_edges(with_layers(g, [:flight])) == 15
        @test num_edges(g) < 16 + 10 + 6 + 15                       # dedup across layers

        # Rail-only leaves the non-rail towns isolated (e.g. Tromsø).
        gr = with_layers(g, [:rail])
        @test active_layers(gr) == [:rail]
        @test get_node_degree(gr, find_node(g, "Tromso")) == 0
        @test get_node_degree(gr, 1) > 0                            # Oslo still connected

        # Selection is normalized to declared order, regardless of request order,
        # and accepts strings.
        @test active_layers(load_geograph(:norway_mock; edges = [:flight, :road])) == [:road, :flight]
        @test active_layers(with_layers(g, ["ferry", "road"])) == [:road, :ferry]

        @test_throws ArgumentError with_layers(g, [:metro])
        @test_throws ArgumentError load_geograph(:norway_mock; edges = [:metro])
        @test_throws ArgumentError load_geograph(:atlantis)
    end

    # -------------------------------------------------------------------------
    # Simulations run on a GeoGraph exactly like any other graph
    # -------------------------------------------------------------------------
    @testset "simulation" begin
        g = load_geograph(:norway_mock)
        sir = create_sir_process(g, 3.0, 1.0;
                                 initial_infected = [find_node(g, "Oslo")], rng_seed = 1)
        run_simulation(sir; max_steps = 10_000)
        # The (well-connected) full graph should have spread beyond the seed.
        @test count_states(g)[SUSCEPTIBLE] < num_nodes(g)

        # ZIM also constructs + runs.
        zim = create_zim_process(load_geograph(:norway_mock), 2.0;
                                 initial_infected = [1], rng_seed = 2)
        @test num_nodes(get_graph(zim)) == 15
    end

    # -------------------------------------------------------------------------
    # Persistence identity (country + active layers)
    # -------------------------------------------------------------------------
    @testset "persistence identity" begin
        g = load_geograph(:norway_mock; edges = [:road, :flight])
        sir = create_sir_process(g, 3.0, 1.0; initial_infected = [1], rng_seed = 1)
        info = extract_process_info(sir)
        @test info["graph_type"] == "GeoGraph"
        @test info["country"] == "norway_mock"
        @test info["edges"] == "flight+road"     # sorted for a stable key
        @test info["num_nodes"] == 15
    end

    # -------------------------------------------------------------------------
    # Rendering smoke test (needs CairoMakie; OFF unless GRAPHEPIMODELS_VIZ_RENDER
    # or CI is set, matching test_visualization.jl).
    # -------------------------------------------------------------------------
    run_render = haskey(ENV, "GRAPHEPIMODELS_VIZ_RENDER") || get(ENV, "CI", "false") == "true"
    if run_render
        using CairoMakie
        @testset "rendering (CairoMakie)" begin
            dir = mktempdir()
            g = load_geograph(:norway_mock)

            # Auto-selected visualizer is the node-link one (a geograph is not a
            # cell graph), and render_frame draws the basemap behind it.
            @test visualizer_for(g) isa NetworkVisualizer
            fig = render_frame(visualizer_for(g), g, node_states_raw(g))
            @test fig isa Figure

            proc = create_sir_process(g, 3.0, 1.0; initial_infected = [1], rng_seed = 1)
            png = joinpath(dir, "norway_mock.png")
            save_plot(proc, png)
            @test isfile(png) && filesize(png) > 0

            gif = joinpath(dir, "norway_mock.gif")
            rec = animate_simulation(create_sir_process(load_geograph(:norway_mock), 3.0, 1.0;
                                                        initial_infected = [1], rng_seed = 1);
                                     sampler = EveryStep(), max_steps = 20, filename = gif)
            @test rec isa SimulationRecording
            @test isfile(gif) && filesize(gif) > 0
        end
    else
        @info "Skipping GeoGraph CairoMakie rendering tests (set GRAPHEPIMODELS_VIZ_RENDER=1 to run them)"
    end
end
