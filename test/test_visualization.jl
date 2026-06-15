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
sir_on(g) = create_sir_process(g, 0.6, 1.0; initial_infected = [1], rng_seed = 1)

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
        # A lattice without a cell tiling (3D cube, d≥4 hypercubic) can't draw as
        # cells, so it routes to the node-link visualizer instead.
        @test visualizer_for(create_cube_lattice(3, 3, 3)) isa NetworkVisualizer
        @test visualizer_for(create_hypercubic_lattice(2, 2, 2, 2)) isa NetworkVisualizer
    end

    @testset "can_visualize" begin
        net = NetworkVisualizer()
        lat = LatticeVisualizer()
        # A node-link visualizer can draw any graph.
        for (_, g) in vcat(lattice_graphs(), network_graphs())
            @test can_visualize(net, g)
        end
        # A lattice visualizer handles cell lattices, not node-link graphs.
        for (_, g) in lattice_graphs()
            @test can_visualize(lat, g)
        end
        @test !can_visualize(lat, create_complete_graph(5))
        # A cell-less lattice (3D cube) is out of scope for the cell visualizer but
        # fine for the node-link one.
        cube = create_cube_lattice(3, 3, 3)
        @test !can_visualize(lat, cube)
        @test can_visualize(net, cube)
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
        # Every graph type that reports a layout returns a 2 × N matrix by default.
        for (_, g) in vcat(lattice_graphs(), network_graphs())
            has_layout(g) || continue
            @test size(node_positions(g)) == (2, num_nodes(g))
        end
    end

    @testset "layout dimensions" begin
        # Default (preferred) dim is 2 for every type that has a layout, so the
        # 2D rendering path is unchanged.
        for (_, g) in vcat(lattice_graphs(), network_graphs())
            has_layout(g) || continue
            @test layout_dim(g) == 2
            @test 2 in supported_layout_dims(g)
        end

        # Lattices, cycle and path are planar-only: they advertise dim 2 only and
        # reject a 3D request rather than returning wrong coordinates.
        for (_, g) in vcat(lattice_graphs(),
                           [("cycle", create_cycle_graph(8)), ("path", create_path_graph(5))])
            @test supported_layout_dims(g) == (2,)
            @test_throws ArgumentError node_positions(g; dim = 3)
        end

        # Star / complete / tree carry a closed-form 3D layout (3 × N). Both tree
        # conventions advertise (2, 3); detailed tree geometry is checked in
        # test_regular_tree.jl.
        for (_, g) in [("star", create_star_graph(7)),
                       ("complete", create_complete_graph(6)),
                       ("dary-tree", create_dary_tree(2, 4)),
                       ("regular-tree", create_regular_tree(3, 3))]
            @test supported_layout_dims(g) == (2, 3)
            p3 = node_positions(g; dim = 3)
            @test size(p3) == (3, num_nodes(g))
        end

        # A cube lattice is 3D-only: it advertises dim 3 (not 2), so it routes to
        # the node-link visualizer and the rendering layer defaults it to 3D. This
        # is the geometry that drives that default.
        cube = create_cube_lattice(4, 4, 4)
        @test supported_layout_dims(cube) == (3,)
        @test layout_dim(cube) == 3
        @test size(node_positions(cube; dim = 3)) == (3, num_nodes(cube))
        @test_throws ArgumentError node_positions(cube; dim = 2)
    end

    @testset "3D layout geometry" begin
        # Star: center at origin, every leaf on the unit sphere.
        star = create_star_graph(9)
        p = node_positions(star; dim = 3)
        @test all(iszero, p[:, 1])
        for i in 2:num_nodes(star)
            @test isapprox(hypot(p[1, i], p[2, i], p[3, i]), 1.0; atol = 1e-9)
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

            # 3D node-link: static render + save (star sphere) and a turntable
            # animation (tree shells), covering the Axis3 draw path end to end.
            @testset "3D static render + save" begin
                g = create_star_graph(8)
                fig = render_frame(NetworkVisualizer(dim = 3), g, node_states_raw(g))
                @test fig isa Figure
                file = joinpath(dir, "star_3d.png")
                save_plot(sir_on(g), file; dim = 3)
                @test isfile(file) && filesize(file) > 0
            end

            @testset "3D turntable animation" begin
                file = joinpath(dir, "tree_3d.gif")
                rec = animate_simulation(sir_on(create_dary_tree(2, 4));
                                         sampler = EveryStep(), max_steps = 20,
                                         filename = file, dim = 3, turntable = true)
                @test rec isa SimulationRecording
                @test isfile(file) && filesize(file) > 0
            end

            # A 3D cube lattice renders as a node-link diagram and DEFAULTS to 3D
            # (no explicit dim), exercising the has_cells routing + dim auto-select.
            @testset "3D cube lattice render + save (defaults to 3D)" begin
                g = create_cube_lattice(4, 4, 4)
                @test visualizer_for(g) isa NetworkVisualizer
                file = joinpath(dir, "cube_3d.png")
                save_plot(sir_on(g), file)
                @test isfile(file) && filesize(file) > 0
            end

            # A lattice has no 3D layout: requesting dim=3 is a clear error, not a
            # silently-wrong drawing.
            @testset "lattice rejects dim=3" begin
                @test_throws ArgumentError save_plot(sir_on(create_square_lattice(4, 4)),
                                                     joinpath(dir, "nope.png"); dim = 3)
            end
        end
    end
end
