# Tests for the Chase-escape (predator–prey) model.
#
# State mapping: White = SUSCEPTIBLE, Red = INFECTED, Blue = REMOVED.
# A few deterministic event-bookkeeping tests reach into the non-exported event
# handlers via the module qualifier (GraphEpimodels._ce_spread! / _ce_catch!);
# everything else goes through the public API.

@testset "ChaseEscape" begin

    @testset "Construction & validation" begin
        g = create_path_graph(5)

        # Builds with valid rates.
        p = ChaseEscapeProcess(g, 1.5, 1.0)
        @test p.λ == 1.5
        @test p.μ == 1.0
        @test p.ghost == true

        # Non-positive rates throw.
        @test_throws ArgumentError ChaseEscapeProcess(g, -1.0, 1.0)
        @test_throws ArgumentError ChaseEscapeProcess(g, 0.0, 1.0)
        @test_throws ArgumentError ChaseEscapeProcess(g, 1.0, -1.0)
        @test_throws ArgumentError ChaseEscapeProcess(g, 1.0, 0.0)
        @test_throws ArgumentError create_chase_escape_simulation(g, 0.0)

        # Red/blue seed sets must be disjoint.
        @test_throws ArgumentError create_chase_escape_simulation(
            g, 1.0, 1.0; ghost = false, initial_red = [1], initial_blue = [1])

        # Out-of-range blue seed.
        @test_throws ArgumentError create_chase_escape_simulation(
            g, 1.0, 1.0; ghost = false, initial_red = [1], initial_blue = [99])
    end

    @testset "Reset / ghost mechanism" begin
        g = create_path_graph(5)

        # Single red seed, ghost on: the ghost makes the seed catchable but is
        # never placed in the graph (so REMOVED count stays 0).
        p = create_chase_escape_simulation(g, 2.0, 1.0; ghost = true, initial_red = [1])
        st = get_chase_escape_statistics(p)
        @test st[:infected] == 1
        @test st[:removed]  == 0          # ghost is not a real node
        @test st[:susceptible] == 4
        @test st[:catch_boundary] >= 1    # the ghost contributes +1
        @test st[:spread_boundary] == 1   # node 1's only white neighbour is node 2
        @test is_active(p)

        # ghost = false with an explicit blue seed adjacent to the red seed.
        p2 = create_chase_escape_simulation(
            g, 2.0, 1.0; ghost = false, initial_red = [2], initial_blue = [1])
        st2 = get_chase_escape_statistics(p2)
        @test st2[:infected] == 1
        @test st2[:removed]  == 1         # the explicit blue IS a real node
        @test st2[:catch_boundary] == 1   # node 2 has one blue neighbour (node 1)
        @test is_active(p2)

        # ghost = false with no blue seeds warns (degenerate pure-growth regime).
        @test_logs (:warn,) match_mode=:any create_chase_escape_simulation(
            g, 2.0, 1.0; ghost = false, initial_red = [1])
    end

    @testset "Rate calculation" begin
        # Path 1-2-3-4-5; reds at {1,2}, ghost on, no blue.
        #   spread_boundary: node 2 has one white neighbour (3)        => 1
        #                    node 1 has no white neighbour (2 is red)  => 0
        #   catch_boundary : nodes 1 and 2 each carry a ghost (+1)     => 2
        g = create_path_graph(5)
        p = create_chase_escape_simulation(g, 1.5, 1.0; ghost = true, initial_red = [1, 2])
        st = get_chase_escape_statistics(p)
        @test st[:spread_boundary] == 1
        @test st[:catch_boundary]  == 2
        @test get_total_rate(p) ≈ 1.5 * 1 + 1.0 * 2
    end

    @testset "Spread event bookkeeping" begin
        # [R, W, W, W, W]; spread from node 1 must convert node 2.
        g = create_path_graph(5)
        p = create_chase_escape_simulation(g, 1.0, 1.0; ghost = true, initial_red = [1])
        GraphEpimodels._ce_spread!(p, 1)

        @test get_node_state(g, 2) == INFECTED
        st = get_chase_escape_statistics(p)
        @test st[:infected] == 2
        # Node 1 lost its only white neighbour -> dropped from the spread tracker;
        # node 2 (white neighbour 3) is the only remaining active red.
        @test st[:active_red] == 1
        @test st[:spread_boundary] == 1
        # Node 1 keeps its ghost in the catch tracker; node 2 has no blue neighbour.
        @test st[:catch_boundary] == 1
    end

    @testset "Catch event bookkeeping" begin
        # [R, R, W, W, W]; catching node 1 turns it blue and increments the catch
        # weight of its red neighbour (node 2).
        g = create_path_graph(5)
        p = create_chase_escape_simulation(g, 1.0, 1.0; ghost = true, initial_red = [1, 2])
        GraphEpimodels._ce_catch!(p, 1)

        @test get_node_state(g, 1) == REMOVED
        st = get_chase_escape_statistics(p)
        @test st[:removed]  == 1
        @test st[:infected] == 1
        # Node 2: ghost (+1) plus the newly-blue node 1 (+1) = 2.
        @test st[:catch_boundary] == 2
        # Node 1 went red->blue, so node 2's white-neighbour count is unchanged.
        @test st[:spread_boundary] == 1

        # Catching the last red ends the process.
        GraphEpimodels._ce_catch!(p, 2)
        @test get_node_state(g, 2) == REMOVED
        @test !is_active(p)
    end

    @testset "Monotonicity & conservation" begin
        p = create_chase_escape_simulation(15, 15, 3.0; rng_seed = 7)
        g = get_graph(p)
        n = num_nodes(g)
        states = node_states_raw(g)
        prev = copy(states)

        monotone  = true
        conserved = true
        steps = 0
        while is_active(p) && steps < 100_000
            dt = step!(p)
            dt == Inf && break
            monotone &= all(states .>= prev)                 # only W(0)->R(1)->B(2)
            st = get_statistics(p)
            conserved &= (st[:susceptible] + st[:infected] + st[:removed] == n)
            copyto!(prev, states)
            steps += 1
        end

        @test monotone
        @test conserved
        @test !is_active(p)                                  # ran to completion
        @test get_chase_escape_statistics(p)[:total_ever_infected] >= 1
    end

    @testset "Determinism" begin
        p1 = create_chase_escape_simulation(20, 20, 2.5; rng_seed = 123)
        run_simulation(p1)
        s1 = get_chase_escape_statistics(p1)

        p2 = create_chase_escape_simulation(20, 20, 2.5; rng_seed = 123)
        run_simulation(p2)
        s2 = get_chase_escape_statistics(p2)

        @test s1[:infected]   == s2[:infected]
        @test s1[:removed]    == s2[:removed]
        @test s1[:step_count] == s2[:step_count]
        @test s1[:time]       == s2[:time]
    end

    @testset "Edge cases" begin
        # Single-node lattice: only a ghost-catch is possible.
        p = create_chase_escape_simulation(1, 1, 2.0; ghost = true)
        g = get_graph(p)
        @test num_nodes(g) == 1
        @test is_active(p)                 # ghost makes the lone red catchable
        run_simulation(p)
        @test !is_active(p)
        st = get_chase_escape_statistics(p)
        @test st[:removed] == 1            # the seed was caught (now blue)
        @test st[:infected] == 0

        # Red seed with no white neighbours but an explicit blue neighbour.
        g2 = create_path_graph(2)
        p2 = create_chase_escape_simulation(
            g2, 2.0, 1.0; ghost = false, initial_red = [1], initial_blue = [2])
        @test is_active(p2)
        run_simulation(p2)
        st2 = get_chase_escape_statistics(p2)
        @test st2[:removed] == 2           # node 1 caught by node 2
        @test st2[:infected] == 0
    end

end
