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
        # The package ships three Nordic sample bundles; users build any others
        # themselves with the country-graph builder.
        @test available_country_graphs() == ["denmark", "norway", "sweden"]
        @test !("norway_mock" in available_country_graphs())   # mock retired
        sets = country_edge_sets(:norway)
        @test sets == [(:road, "Roads"), (:rail, "Railways"),
                       (:ferry, "Ferries"), (:flight, "Flights")]
        @test_throws ArgumentError country_edge_sets(:atlantis)
    end

    # -------------------------------------------------------------------------
    # Every shipped bundle loads, is well-formed, and simulates. Norway is
    # exercised in depth below; this guards the others (and any bundle added
    # later) against a malformed or unloadable ship.
    # -------------------------------------------------------------------------
    @testset "all shipped bundles" begin
        expected = Dict(:denmark => (159, 269, "Copenhagen"),
                        :norway  => (114, 287, "Oslo"),
                        :sweden  => (277, 553, "Stockholm"))
        for name in available_country_graphs()
            slug = Symbol(name)
            g = load_geograph(slug)
            @test g isa GeoGraph
            @test num_nodes(g) >= 1
            @test available_layers(g) == [:road, :rail, :ferry, :flight]

            # Basemap is declared and the file resolves.
            @test has_basemap(g)
            @test isfile(basemap(g).path)

            # Every single-layer subset is a subgraph of the full union.
            for layer in available_layers(g)
                @test num_edges(with_layers(g, [layer])) <= num_edges(g)
            end

            # Node ids are usable and populations are non-negative.
            @test all(node_population(g, i) >= 0 for i in 1:num_nodes(g))
            @test 1 <= largest_settlement(g) <= num_nodes(g)

            # Known sizes + the capital is both findable and the largest settlement.
            n, m, capital = expected[slug]
            @test num_nodes(g) == n
            @test num_edges(g) == m
            cid = find_node(g, capital)
            @test cid !== nothing
            @test largest_settlement(g) == cid

            # A run seeded at the capital spreads beyond the seed.
            sir = create_sir_process(g, 3.0, 1.0; initial_infected = [cid], rng_seed = 1)
            run_simulation(sir; max_steps = 50_000)
            @test count_states(g)[SUSCEPTIBLE] < num_nodes(g)
        end
    end

    # -------------------------------------------------------------------------
    # Loading + node table
    # -------------------------------------------------------------------------
    @testset "load + nodes" begin
        g = load_geograph(:norway)
        @test g isa GeoGraph
        @test num_nodes(g) == 114

        # Names + populations + name lookup (case-insensitive, Unicode-aware).
        @test node_name(g, 1) == "Oslo"
        @test node_population(g, 1) == 1_082_575
        @test find_node(g, "oslo") == 1
        @test find_node(g, "BERGEN") == 2         # case-insensitive (ASCII)
        @test find_node(g, "tromsø") == 17        # case-insensitive Unicode name
        @test find_node(g, "nowhere") === nothing
        @test largest_settlement(g) == 1          # Oslo is most populous

        # Geometry: geographic coordinates are a 2D layout.
        @test supported_layout_dims(g) == (2,)
        @test layout_dim(g) == 2
        @test size(node_positions(g)) == (2, 114)
        @test node_positions(g)[:, 1] ≈ [10.74609, 59.91273]
        @test_throws ArgumentError node_positions(g; dim = 3)

        # Basemap handle present and resolvable.
        @test has_basemap(g)
        bm = basemap(g)
        @test bm isa Basemap
        @test isfile(bm.path)
        @test all(bm.bbox .≈ (4.0328, 24.68092, 57.0274, 71.66313))
    end

    # -------------------------------------------------------------------------
    # Edge layers + selection
    # -------------------------------------------------------------------------
    @testset "layers" begin
        g = load_geograph(:norway)
        @test available_layers(g) == [:road, :rail, :ferry, :flight]
        @test active_layers(g) == [:road, :rail, :ferry, :flight]   # :all by default
        @test layer_label(g, :rail) == "Railways"

        # Single-layer subsets have exactly that layer's (deduped) edge count; the
        # union is smaller than the naive sum (edges shared across layers collapse).
        @test num_edges(with_layers(g, [:road]))   == 151
        @test num_edges(with_layers(g, [:rail]))   == 107
        @test num_edges(with_layers(g, [:ferry]))  == 29
        @test num_edges(with_layers(g, [:flight])) == 81
        @test num_edges(g) == 287                                   # union of all layers
        @test num_edges(g) < 151 + 107 + 29 + 81                   # dedup across layers

        # Rail-only is sparse: some towns end up isolated, but Oslo (a rail hub) stays
        # connected.
        gr = with_layers(g, [:rail])
        @test active_layers(gr) == [:rail]
        @test any(get_node_degree(gr, i) == 0 for i in 1:num_nodes(gr))
        @test get_node_degree(gr, 1) > 0                            # Oslo still connected

        # Selection is normalized to declared order, regardless of request order,
        # and accepts strings.
        @test active_layers(load_geograph(:norway; edges = [:flight, :road])) == [:road, :flight]
        @test active_layers(with_layers(g, ["ferry", "road"])) == [:road, :ferry]

        @test_throws ArgumentError with_layers(g, [:metro])
        @test_throws ArgumentError load_geograph(:norway; edges = [:metro])
        @test_throws ArgumentError load_geograph(:atlantis)
    end

    # -------------------------------------------------------------------------
    # Simulations run on a GeoGraph exactly like any other graph
    # -------------------------------------------------------------------------
    @testset "simulation" begin
        g = load_geograph(:norway)
        sir = create_sir_process(g, 3.0, 1.0;
                                 initial_infected = [find_node(g, "Oslo")], rng_seed = 1)
        run_simulation(sir; max_steps = 10_000)
        # The (well-connected) full graph should have spread beyond the seed.
        @test count_states(g)[SUSCEPTIBLE] < num_nodes(g)

        # ZIM also constructs + runs.
        zim = create_zim_process(load_geograph(:norway), 2.0;
                                 initial_infected = [1], rng_seed = 2)
        @test num_nodes(get_graph(zim)) == 114
    end

    # -------------------------------------------------------------------------
    # Persistence identity (country + active layers)
    # -------------------------------------------------------------------------
    @testset "persistence identity" begin
        g = load_geograph(:norway; edges = [:road, :flight])
        sir = create_sir_process(g, 3.0, 1.0; initial_infected = [1], rng_seed = 1)
        info = extract_process_info(sir)
        @test info["graph_type"] == "GeoGraph"
        @test info["country"] == "norway"
        @test info["edges"] == "flight+road"     # sorted for a stable key
        @test info["num_nodes"] == 114
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
            g = load_geograph(:norway)

            # Auto-selected visualizer is the node-link one (a geograph is not a
            # cell graph), and render_frame draws the basemap behind it.
            @test visualizer_for(g) isa NetworkVisualizer
            fig = render_frame(visualizer_for(g), g, node_states_raw(g))
            @test fig isa Figure

            # Dark theme renders too (geo basemap path).
            @test render_frame(visualizer_for(g; dark = true), g, node_states_raw(g)) isa Figure

            proc = create_sir_process(g, 3.0, 1.0; initial_infected = [1], rng_seed = 1)
            png = joinpath(dir, "norway.png")
            save_plot(proc, png)
            @test isfile(png) && filesize(png) > 0

            gif = joinpath(dir, "norway.gif")
            rec = animate_simulation(create_sir_process(load_geograph(:norway), 3.0, 1.0;
                                                        initial_infected = [1], rng_seed = 1);
                                     sampler = EveryStep(), max_steps = 20, filename = gif)
            @test rec isa SimulationRecording
            @test isfile(gif) && filesize(gif) > 0
        end
    else
        @info "Skipping GeoGraph CairoMakie rendering tests (set GRAPHEPIMODELS_VIZ_RENDER=1 to run them)"
    end
end
