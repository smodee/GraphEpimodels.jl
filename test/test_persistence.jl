using Test
using GraphEpimodels

# Loading CSV + DataFrames activates GraphEpimodelsPersistenceExt, which provides
# both the single-row public functions and the batched in-memory store that
# run_parameter_sweep now uses (read once up front, write once at the end).
using CSV, DataFrames

# Shorthand for the unexported batched-store API.
const GE = GraphEpimodels

@testset "CSV survival-result persistence" begin

    # Two distinct process configs and their reproducibility info dicts.
    info_a = extract_process_info(create_zim_process(10, 10, 1.5))
    info_b = extract_process_info(create_zim_process(10, 10, 2.5))

    @testset "batched store: load / lookup / record / save" begin
        mktempdir() do dir
            csv = joinpath(dir, "results.csv")

            # Missing file → empty store, fresh seed.
            store = GE.load_survival_results(csv)
            @test nrow(store.df) == 0
            @test GE.next_start_seed(store, 1.5, info_a) == 1

            # First record for (info_a, λ=1.5) appends a new row.
            appended = GE.record_survival_result!(store, 1.5, 100, 40, 0.40,
                                                  0.049, 1, 100, info_a)
            @test appended == false
            @test nrow(store.df) == 1
            # Continuation seed now picks up after end_seed.
            @test GE.next_start_seed(store, 1.5, info_a) == 101

            # Same (config, param) → cumulative update of the existing row.
            updated = GE.record_survival_result!(store, 1.5, 100, 60, 0.60,
                                                 0.049, 101, 200, info_a)
            @test updated == true
            @test nrow(store.df) == 1
            row = store.df[1, :]
            @test row.num_simulations == 200          # 100 + 100
            @test row.num_survivals == 100            # 40 + 60
            @test row.survival_probability ≈ 0.5      # 100/200
            @test row.start_seed == 1                 # unchanged
            @test row.end_seed == 200
            @test GE.next_start_seed(store, 1.5, info_a) == 201

            # A different config (info_b) is independent → appends, fresh seed.
            @test GE.next_start_seed(store, 1.5, info_b) == 1
            GE.record_survival_result!(store, 1.5, 50, 10, 0.20, 0.057, 1, 50, info_b)
            @test nrow(store.df) == 2

            # Same config, different parameter → independent row too.
            GE.record_survival_result!(store, 3.0, 50, 25, 0.50, 0.071, 1, 50, info_a)
            @test nrow(store.df) == 3

            # Persist once; reloading reproduces every continuation seed (proving the
            # cached config strings survive a JSON round-trip).
            GE.save_survival_results(store, csv)
            @test isfile(csv)
            reloaded = GE.load_survival_results(csv)
            @test nrow(reloaded.df) == 3
            @test GE.next_start_seed(reloaded, 1.5, info_a) == 201
            @test GE.next_start_seed(reloaded, 1.5, info_b) == 51
            @test GE.next_start_seed(reloaded, 3.0, info_a) == 51
            @test GE.next_start_seed(reloaded, 2.0, info_a) == 1   # no such row
        end
    end

    @testset "run_parameter_sweep persists and continues" begin
        mktempdir() do dir
            csv = joinpath(dir, "sweep.csv")
            params = [1.5, 2.0]
            gen = λ -> (() -> create_zim_process(10, 10, λ))
            center = get_center_node(create_square_lattice(10, 10))

            # First sweep creates the file: one row per parameter, seeds 1–5.
            run_parameter_sweep(params, gen, [center], EscapeCriterion();
                                save_to = csv, num_simulations = 5,
                                use_threading = false)
            df1 = CSV.read(csv, DataFrame)
            @test nrow(df1) == 2
            @test sort(df1.parameter) == params
            @test all(df1.num_simulations .== 5)
            @test all(df1.start_seed .== 1)
            @test all(df1.end_seed .== 5)

            # Re-running continues from seed 6 and folds in cumulative counts.
            run_parameter_sweep(params, gen, [center], EscapeCriterion();
                                save_to = csv, num_simulations = 5,
                                use_threading = false)
            df2 = CSV.read(csv, DataFrame)
            @test nrow(df2) == 2                       # updated in place, not appended
            @test all(df2.num_simulations .== 10)      # 5 + 5
            @test all(df2.start_seed .== 1)            # original start preserved
            @test all(df2.end_seed .== 10)
        end
    end

    @testset "batched store matches single-row public functions" begin
        mktempdir() do dir
            # The batched store must produce a file the public get_next_start_seed
            # reads identically (the two APIs share the same config-string logic).
            csv = joinpath(dir, "parity.csv")
            store = GE.load_survival_results(csv)
            GE.record_survival_result!(store, 1.5, 100, 40, 0.40, 0.049, 1, 100, info_a)
            GE.save_survival_results(store, csv)

            @test get_next_start_seed(csv, 1.5, info_a) == 101
            @test get_next_start_seed(csv, 1.5, info_b) == 1
        end
    end
end
