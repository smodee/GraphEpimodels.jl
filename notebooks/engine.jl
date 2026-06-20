"""
engine.jl — the non-UI logic behind `epidemics_explorer.jl`.

These pure functions translate the notebook's control values (a `cfg` NamedTuple)
into a graph, a process, and an adaptively-sampled `SimulationRecording` (a single
pass — no measurement run), plus the preview/playback helpers. They use only the
public GraphEpimodels API.

The notebook `include`s this file so the same code is exercised headlessly by
`smoke_test.jl`. Assumes `GraphEpimodels`, `Random`, `NetworkLayout`, and `Base64`
are in scope (the notebook's setup cell and the smoke test both load them).
"""

# --- Adaptive sampling / playback tuning -------------------------------------
const MAX_FRAMES   = 512         # adaptive capture cap; halved on overflow (≥256 frames survive)
const PREVIEW_PX    = 440        # preview render resolution (player scales it to fit)
const PREVIEW_CAP   = 64         # cap on rendered preview frames — kept low for fast Run; some choppiness at long play times is acceptable
const FPS_TARGET    = 16         # fps the preview player aims for
const FPS_FLOOR     = 4          # min playback fps before holding the end frame; = PREVIEW_CAP ÷ max play time (16 s), so a surviving run plays full-length (down to this fps) without ever freezing
const EXPORT_FPS    = 30         # cap on exported-clip fps (smoothness)
const SAFETY_STEPS  = 5_000_000  # hard stop for the single recording run

"""
Best single 'center' seed node for a graph type.

For a rooted tree or a star the meaningful center is node 1 (the root / hub) — the
center of their radial layouts — not the package's generic `:center`, which falls
back to `num_nodes ÷ 2` (a leaf/spoke under BFS numbering). Lattices that define
`get_center_node` use it for the exact geometric center; everything else (path,
cycle, complete, Erdős–Rényi) uses the middle-of-numbering fallback.
"""
function center_node(graph)
    if graph isa RegularTree || graph isa StarGraph
        1
    elseif hasmethod(get_center_node, (typeof(graph),))
        get_center_node(graph)              # square / cube: exact center
    elseif has_cells(graph)
        _centroid_node(graph)               # triangular / hexagonal: interior, not boundary
    else
        max(1, num_nodes(graph) ÷ 2)        # path/cycle/complete/ER: middle-ish
    end
end

"Index of the node nearest the geometric centroid of a graph's 2D layout."
function _centroid_node(graph)
    pos = node_positions(graph; dim = 2)    # 2 × n
    n = size(pos, 2)
    cx = sum(@view pos[1, :]) / n
    cy = sum(@view pos[2, :]) / n
    best, bestd = 1, Inf
    for i in 1:n
        d = (pos[1, i] - cx)^2 + (pos[2, i] - cy)^2
        if d < bestd
            bestd = d
            best = i
        end
    end
    best
end

"Resolve the initial-condition control into a spec the constructors accept."
function init_spec(cfg, graph)
    if cfg.init_kind == "Center"
        [center_node(graph)]
    elseif cfg.init_kind == "Random"
        :random
    else  # "Center patch"
        center_patch_bfs(graph, cfg.patch_r)
    end
end

"Center node (per `center_node`) expanded to its r-hop neighborhood via BFS."
function center_patch_bfs(graph, r::Integer)
    c = center_node(graph)
    r <= 0 && return [c]
    seen = Set{Int}((c,))
    frontier = [c]
    for _ in 1:r
        nxt = Int[]
        for u in frontier, v in get_neighbors(graph, u)
            if !(v in seen)
                push!(seen, v)
                push!(nxt, v)
            end
        end
        frontier = nxt
        isempty(frontier) && break
    end
    collect(seen)
end

"Build the graph for the staged config; `seed` seeds random graphs."
function build_graph(cfg, seed::Integer)
    g = cfg.gsize
    f = cfg.graph_family
    if f == "Square lattice"
        create_square_lattice(g.width, g.height, g.boundary)
    elseif f == "Triangular lattice"
        create_triangular_lattice(g.width, g.height)
    elseif f == "Hexagonal lattice"
        create_hexagonal_lattice(g.width, g.height)
    elseif f == "Cube lattice"
        create_cube_lattice(g.width, g.height, g.depth, g.boundary)
    elseif f == "Complete graph"
        create_complete_graph(g.n)
    elseif f == "Path"
        create_path_graph(g.n)
    elseif f == "Cycle"
        create_cycle_graph(g.n)
    elseif f == "Star"
        create_star_graph(g.n)
    elseif f == "Regular tree"
        create_regular_tree(g.degree, g.height)
    elseif f == "d-ary tree"
        create_dary_tree(g.branching, g.height)
    elseif f == "Erdos-Renyi"
        create_erdos_renyi(g.n; p = g.p, rng = Random.Xoshiro(seed))
    else
        error("Unknown graph family: $f")
    end
end

"Build the process for the staged config on `graph`."
function build_process(cfg, graph, seed::Integer)
    init = init_spec(cfg, graph)
    m = cfg.mparams
    if cfg.model == "SIR"
        create_sir_process(graph, m.beta, m.gamma;
                           initial_infected = init, rng_seed = seed)
    elseif cfg.model == "ZIM"
        create_zim_process(graph, m.lambda, m.mu;
                           initial_infected = init, rng_seed = seed)
    elseif cfg.model == "Maki-Thompson"
        create_maki_thompson_process(graph, m.alpha, m.beta;
                           stifler_contact = m.stifler,
                           initial_infected = init, rng_seed = seed)
    else  # "Chase-Escape"
        create_chase_escape_process(graph, m.lambda, m.mu;
                           ghost = m.ghost,
                           initial_red = init, rng_seed = seed)
    end
end

"""
Build graph + process and record it in a single adaptive pass.

`record_simulation_adaptive` captures frames as the run proceeds and halves the
rate whenever the buffer reaches `MAX_FRAMES`, so the recording is bounded
(≈256–512 frames) without a separate measurement run — and the recording no
longer depends on `target_time` at all: play length is purely a playback choice
(see `frame_player` / `export_fps`). `time_model` is `"Continuous"` (equal
sim-time spacing) or `"Discrete"` (equal event spacing).
"""
function build_recording(cfg, seed::Integer)
    graph = build_graph(cfg, seed)
    time_model = cfg.time_model == "Discrete" ? :discrete : :continuous
    record_simulation_adaptive(build_process(cfg, graph, seed);
                               time_model = time_model, max_frames = MAX_FRAMES,
                               stop_on_escape = cfg.stop_escape, max_steps = SAFETY_STEPS)
end

# --- Playback planning -------------------------------------------------------

"Total nodes ever infected/removed in the final recorded frame."
ever_infected(rec) = rec.counts[end][2] + rec.counts[end][3]

"Did the process fizzle out (cluster stayed tiny relative to the graph)?"
is_trivial(rec) = ever_infected(rec) < max(10, round(Int, 0.03 * num_nodes(rec.graph)))

"Evenly spaced indices (with endpoints) selecting ~`m` of `1:n`."
function subsample_indices(n::Int, m::Int)
    (m >= n || n <= 1) && return collect(1:n)
    m <= 1 && return [1]
    unique(round.(Int, range(1, n; length = m)))
end

"""
Choose which recording frames to rasterize for the in-notebook preview.

The preview renders a fixed, target-independent budget of up to `PREVIEW_CAP`
evenly-spaced frames *once*; the client-side player then controls playback
duration/speed (and holds the end frame for short runs) entirely in the browser.
So changing the play-time slider re-runs neither the simulation nor the render.
"""
function preview_indices(rec)
    n = num_frames(rec)
    (indices = subsample_indices(n, clamp(n, 1, PREVIEW_CAP)),
     trivial = is_trivial(rec))
end

"""
Export fps so the recording lasts ~`target` seconds, clamped to
`[FPS_FLOOR, EXPORT_FPS]`. A very short run plays at `FPS_FLOOR` (a brief clip)
rather than being slowed to a crawl; the exported file is not padded.
"""
export_fps(rec, target) = clamp(round(Int, num_frames(rec) / max(target, 1e-9)), FPS_FLOOR, EXPORT_FPS)

"Fixed node positions for a node-link graph (mirrors the package's resolver)."
function frame_positions(graph, dim::Int)
    dims = supported_layout_dims(graph)
    dim in dims && return node_positions(graph; dim = dim)
    ld = layout_dim(graph)
    ld > dim && return node_positions(graph; dim = ld)[1:dim, :]
    n = num_nodes(graph)
    adj = falses(n, n)
    for i in 1:n, j in get_neighbors(graph, i)
        adj[i, j] = true
    end
    pts = NetworkLayout.spring(adj; dim = dim, seed = 1)
    mat = Matrix{Float64}(undef, dim, n)
    for i in 1:n, d in 1:dim
        mat[d, i] = pts[i][d]
    end
    mat
end

"""
Per-frame title string with fixed-width numbers, so the centered caption doesn't
jump around as values change length. Widths are derived from the run's maxima.
"""
function frame_title(rec, i::Int)
    (_, nI, nR) = rec.counts[i]
    sw = ndigits(max(1, rec.steps[end]))                       # max step count
    cw = ndigits(num_nodes(rec.graph))                         # max S/I/R count
    tw = ndigits(max(1, floor(Int, rec.times[end]))) + 3       # "int.dd"
    t  = lpad(Printf.@sprintf("%.2f", rec.times[i]), tw)
    "$(rec.process_name)  (t=$t, step=$(lpad(rec.steps[i], sw)), " *
    "I=$(lpad(nI, cw)), R=$(lpad(nR, cw)))"
end

# --- Pre-rendered frame cache + client-side player ---------------------------
# Rendering a Makie Figure is the slow step. We rasterize the (subsampled) preview
# frames to PNG bytes ONCE (`render_png` / `build_frame_cache`), then `frame_player`
# embeds them in a small HTML/JS widget that animates them in the browser — so
# playback runs client-side and isn't capped by Pluto's per-tick round-trip.

"""
Rasterize one frame to PNG bytes (lattice path uses transparency; network uses
positions). `turn_frac` (0–1), when given, rotates a 3D camera by that fraction of
a full turn relative to its default azimuth — used for an in-notebook turntable.
"""
function render_png(viz, graph, frame; title = "", positions = nothing,
                    transparent = false, turn_frac = nothing)
    fig = positions === nothing ?
        render_frame(viz, graph, frame; title = title, transparent_background = transparent) :
        render_frame(viz, graph, frame; title = title, positions = positions)
    if turn_frac !== nothing
        ax = fig.content[1]                       # the Axis3
        ax.azimuth[] = ax.azimuth[] + 2π * turn_frac
    end
    io = IOBuffer()
    show(io, MIME"image/png"(), fig)
    take!(io)
end

"""
Render the given `indices` of `rec` to PNG bytes (the in-notebook preview frames).
With `turntable = true` on a 3D node-link view, the camera rotates one full turn
across the clip.
"""
function build_frame_cache(rec, viz, indices; positions = nothing,
                           transparent = false, turntable = false)
    is3d = turntable && viz isa NetworkVisualizer && viz.dim == 3
    n = length(indices)
    [render_png(viz, rec.graph, rec.frames[idx];
                title = frame_title(rec, idx),
                positions = positions, transparent = transparent,
                turn_frac = is3d ? (k - 1) / n : nothing)
     for (k, idx) in enumerate(indices)]
end

"""
A self-contained HTML/JS player that animates `pngs` in the browser so the clip
lasts ~`play_seconds`, with play/pause and a scrub slider.

Playback length is decided entirely client-side, so changing it never re-renders
or re-simulates. The player aims for `fps_target`; if the run has too few frames
to fill `play_seconds` at that rate it eases the fps down to `fps_floor`, and if
even that isn't enough (a run that died early) it holds the final frame to fill
the remaining time rather than crawling. Short play times instead show an
evenly-strided subset, so the clip is always about `play_seconds` long.
"""
function frame_player(pngs, play_seconds::Real;
                      fps_target::Real = FPS_TARGET, fps_floor::Real = FPS_FLOOR)
    isempty(pngs) && return Base.HTML("<em>No frames.</em>")
    srcs = join(("\"data:image/png;base64," * Base64.base64encode(p) * "\"" for p in pngs), ",")
    uid = "ep_player_" * string(rand(UInt32); base = 16)
    Base.HTML("""
    <div id="$uid" style="display:flex;flex-direction:column;gap:6px;align-items:center;">
      <img class="ep-frame" style="max-width:100%;height:auto;background:#ffffff;border-radius:4px;" />
      <div style="display:flex;gap:8px;align-items:center;width:100%;max-width:640px;">
        <button class="ep-pp" style="width:3em;cursor:pointer;">⏸</button>
        <input class="ep-scrub" type="range" min="0" max="0" value="0" style="flex:1;">
        <span class="ep-cnt" style="font-variant-numeric:tabular-nums;min-width:5em;text-align:right;"></span>
      </div>
    </div>
    <script>
    (function() {
      const root = document.getElementById("$uid");
      if (!root) return;
      const frames = [$srcs];
      const F = frames.length;
      const D = $(Float64(play_seconds));
      const fpsTarget = $(Float64(fps_target));
      const fpsFloor = $(Float64(fps_floor));

      // Choose an fps and a playlist of frame indices so the clip lasts ~D seconds.
      let fps = fpsTarget;
      let desired = Math.max(2, Math.round(D * fps));
      if (desired > F) {                         // not enough frames at fpsTarget
        fps = Math.min(fpsTarget, Math.max(fpsFloor, F / D));
        desired = Math.max(2, Math.round(D * fps));
      }
      let playlist = [];
      if (desired <= F) {                        // stride down to `desired` frames
        for (let k = 0; k < desired; k++)
          playlist.push(Math.round(k * (F - 1) / (desired - 1)));
      } else {                                    // play all, then hold the last frame
        for (let k = 0; k < F; k++) playlist.push(k);
        while (playlist.length < desired) playlist.push(F - 1);
      }
      const interval = Math.round(1000 / Math.min(60, Math.max(1, fps)));

      const img = root.querySelector(".ep-frame");
      const pp = root.querySelector(".ep-pp");
      const scrub = root.querySelector(".ep-scrub");
      const cnt = root.querySelector(".ep-cnt");
      scrub.max = playlist.length - 1;
      let i = 0, timer = null;
      function show(k) {
        i = (k % playlist.length + playlist.length) % playlist.length;
        img.src = frames[playlist[i]]; scrub.value = i;
        cnt.textContent = (i + 1) + " / " + playlist.length;
      }
      function play() { pp.textContent = "⏸"; clearInterval(timer); timer = setInterval(() => show(i + 1), interval); }
      function pause() { pp.textContent = "▶"; clearInterval(timer); timer = null; }
      pp.onclick = () => (timer ? pause() : play());
      scrub.oninput = () => { pause(); show(parseInt(scrub.value)); };
      show(0); play();
      try { invalidation.then(() => clearInterval(timer)); } catch (e) {}
    })();
    </script>
    """)
end
