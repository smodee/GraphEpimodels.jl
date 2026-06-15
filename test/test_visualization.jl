using Test
using GraphEpimodels
using Random

# -----------------------------------------------------------------------------
# Graph fixtures, one per type. Lattices draw as dual-tiling cells; everything
# else (structured implicit graphs, general adjacency, random ER) draws node-link.
# -----------------------------------------------------------------------------
lattice_graphs() = [
    ("square",     create_square_lattice(6, 6)),
    ("triangular", create_triangular_lattice(6, 6)),
    ("hexagonal",  create_hexagonal_lattice(6, 6)),
]

network_graphs() = [
    ("complete", create_complete_graph(8)),
    ("cycle",    create_cycle_graph(10)),
    ("path",     create_path_graph(10)),
    ("star",     create_star_graph(7)),
    ("er",       create_gnp(15, 0.3; rng = MersenneTwister(1))),
    ("adjacency", create_graph_from_edges(4, [(1, 2), (2, 3), (3, 4)])),
]

# An SIR process on a graph, infected starting from node 1 (deterministic and
# valid for every graph type, including those without a center node).
sir_on(g) = create_sir_simulation(g, 0.6, 1.0; initial_infected = [1], rng_seed = 1)

@testset "visualization" begin

    # -------------------------------------------------------------------------
    # Backend-independent: dispatch, compatibility, schemes, geometry. These run
    # without CairoMakie and are cheap; they guard the integration wiring that is
    # the substance of the visualization layer.
    # -------------------------------------------------------------------------
    @testset "visualizer_for dispatch" begin
        for (_, g) in lattice_graphs()
            @test visualizer_for(g) isa LatticeVisualizer
        end
        for (_, g) in network_graphs()
            @test visualizer_for(g) isa NetworkVisualizer
        end
    end

    @testset "can_visualize" begin
        net = NetworkVisualizer()
        lat = LatticeVisualizer()
        # A node-link visualizer can draw any graph.
        for (_, g) in vcat(lattice_graphs(), network_graphs())
            @test can_visualize(net, g)
        end
        # A lattice visualizer handles lattices, not node-link graphs.
        for (_, g) in lattice_graphs()
            @test can_visualize(lat, g)
        end
        @test !can_visualize(lat, create_complete_graph(5))
    end

    @testset "color schemes" begin
        @test :general in available_color_schemes()
        @test :maki_thompson in available_color_schemes()
        # Model-aware resolution by name.
        @test default_color_scheme("SIR") == :sir
        @test default_color_scheme("ZIM") == :zim
        @test default_color_scheme("ChaseEscape") == :chaseescape
        @test default_color_scheme("MakiThompson") == :maki_thompson
        @test default_color_scheme("something-else") == :general
        # Model-aware resolution by process.
        @test default_color_scheme(sir_on(create_complete_graph(5))) == :sir
        # Constructors default to the neutral scheme.
        @test NetworkVisualizer().color_scheme == :general
        @test LatticeVisualizer().color_scheme == :general
    end

    @testset "node_positions shape" begin
        # Every graph type that reports a layout returns a 2 × N matrix.
        for (_, g) in vcat(lattice_graphs(), network_graphs())
            has_layout(g) || continue
            @test size(node_positions(g)) == (2, num_nodes(g))
        end
    end

    # -------------------------------------------------------------------------
    # Rendering smoke test (needs CairoMakie). Loading CairoMakie (~10 s) plus
    # first-call compilation of the Makie draw paths dominate the suite, so this
    # tier is OFF by default and runs only when GRAPHEPIMODELS_VIZ_RENDER is set
    # (or under CI). The cases below cover every distinct draw path exactly once:
    #   - square     -> lattice `image!` fast path
    #   - triangular -> lattice `poly!` dual-cell path (shared by hexagonal)
    #   - complete   -> node-link with an intrinsic layout
    #   - adjacency  -> node-link with the spring-layout fallback (no coords)
    # plus the two animation record paths (lattice cells and node-link) and the
    # visualizer override.
    # -------------------------------------------------------------------------
    run_render = haskey(ENV, "GRAPHEPIMODELS_VIZ_RENDER") || get(ENV, "CI", "false") == "true"

    if !run_render
        @info "Skipping CairoMakie rendering tests (set GRAPHEPIMODELS_VIZ_RENDER=1 to run them)"
    else
        using CairoMakie
        @testset "rendering (CairoMakie)" begin
            dir = mktempdir()
            render_cases = [("square",     create_square_lattice(6, 6)),
                            ("triangular", create_triangular_lattice(6, 6)),
                            ("complete",   create_complete_graph(8)),
                            ("adjacency",  create_graph_from_edges(4, [(1, 2), (2, 3), (3, 4)]))]

            @testset "render_frame + save: $name" for (name, g) in render_cases
                proc = sir_on(g)
                fig = render_frame(visualizer_for(g), g, node_states_raw(g))
                @test fig isa Figure
                file = joinpath(dir, "static_$name.png")
                save_plot(proc, file)
                @test isfile(file) && filesize(file) > 0
            end

            @testset "animate_simulation (lattice cells path)" begin
                file = joinpath(dir, "anim_square.gif")
                rec = animate_simulation(sir_on(create_square_lattice(6, 6));
                                         sampler = EveryStep(), max_steps = 20, filename = file)
                @test rec isa SimulationRecording
                @test num_frames(rec) >= 2
                @test isfile(file) && filesize(file) > 0
            end

            # Forcing a NetworkVisualizer on a lattice covers both the override
            # feature and the node-link animation record path in one go.
            @testset "lattice as node-link via visualizer override" begin
                file = joinpath(dir, "lattice_nodelink.gif")
                animate_simulation(sir_on(create_square_lattice(6, 6));
                                   sampler = EveryStep(), max_steps = 20,
                                   filename = file, visualizer = NetworkVisualizer())
                @test isfile(file) && filesize(file) > 0
            end
        end
    end
end
