# AGENTS.md — CAAD Futures 2027 experiment campaign

You are picking up `semantopology_prod` on the MIT Engaging cluster. Your job is
to run the full experiment campaign for the CAAD Futures 2027 paper and produce
the analysis outputs. **Do not stop or hand back until the Definition of Done
(bottom of this file) is met.** Do not ask the user questions mid-campaign: when
a decision comes up, take the documented default, log it in `CAMPAIGN_LOG.md`,
and keep going. Park only what genuinely needs a human, and list it in the final
report.

Full paper deadline: October 16, 2026 (AoE). All runs must be done by
October 9, 2026.

## What the paper argues (so you know what matters)

Designer intent enters topology optimization through two channels, formal (a
sketch is a spatial prior on where the fixed volume budget goes) and semantic (a
CLIP text prompt). The strength of each channel is a continuous, tunable setting.
A hybrid uses a CLIP "dream" as the formal prior, bridging the two channels.
Unlike image generators, every design is shaped by the physics gradient. Each
experiment below supports one claim. Do not add experiments that are not listed.

## Hard rules

1. **One platform for every reported run.** Every run, including baselines, uses
   the same partition, environment, and device: `--device cpu`. Never mix
   devices.
2. **Never unset `OMP_NUM_THREADS=1`.** `runtime.py` pins it. Without it, CHOLMOD
   segfaults intermittently, with no traceback.
3. **CHOLMOD cannot be retried in-process.** After a CHOLMOD error, the next
   solve in the same process segfaults. Every run is its own Slurm task or
   process. Retry only by resubmitting.
4. **Keep numpy below 2.0** (the HIPS `autograd` package requires it).
5. **Do not change existing physics, loss algebra, or defaults.** Add new
   problems, recipes, and flags alongside the existing ones. `pytest tests -q`
   must stay green.
6. **Do not commit `results/` or `logs/`** (they are gitignored). Commit code,
   manifests, analysis scripts, and `CAMPAIGN_LOG.md`.
7. **No force-pushes, no deleting results.** A rerun writes to a new attempt
   directory.

## Phase 0 — setup (finish before submitting any runs)

### 0.1 Environment
- Build a Linux conda env named `semantopology` with Python 3.11, torch 2.2.2
  (CPU build), numpy 1.26.x, scipy, and scikit-sparse (SuiteSparse/CHOLMOD),
  plus `requirements.txt`.
- Pre-download the CLIP weights (`ViT-B/32`, `RN50`) to a shared cache on the
  login node, and set the cache path in the sbatch script. Compute nodes may
  have no internet access.
- `pytest tests -q` must pass on a compute node (`srun`), not just the login node.

### 0.2 Structural problems (add to `problem/problems.py`, with tests)
Convention: array index `[x, y]`, with `y = 0` at the top. Loads are `-1/width`
per node in Y on each loaded row.
- `tall_building`: the existing `multistory_building` at 128×256, interval 64
  (4 loaded floors). Alias only; do not change it.
- `short_cantilever_building`: 300×150, loaded rows at y = 0, 50, 100 (three
  storeys). Y-support on the bottom row for x in [0, 0.70·width] only; the right
  30% overhangs. Constrain X along the left wall so the system is not singular.
- `double_decker_bridge`: 448×72, pin (X and Y) at the bottom-left corner,
  roller (Y) at the bottom-right corner. Distributed load on the top row
  (y = 0) and the bottom row (y = 71). No point loads.
- Each new problem gets a test that the stiffness system factors (no rank
  deficiency) and that loads and supports sit where specified.

### 0.3 Sketches (put in `inputs/sketches/`, commit them)
- Copy from `semantopology_venice/resources/input_images/sketches/`:
  `1.jpg 3.jpg 6.jpg 9.jpg 11.jpg 12.jpg`. Copy
  `dSketches/dSketch_4_density_0.3_4_story.png` and save it as `col2.png`.
- Generate programmatically at 512×1024 (the aspect ratio of the 128×256
  domain), black ink on white:
  - `col6_grid.png`: 6 evenly spaced vertical columns, plus horizontal floor
    lines at exactly 0, 1/4, 1/2 and 3/4 of the height (aligned with the load
    rows). Line width is 5 elements at 128-wide resolution.
  - `col3.png`: 3 evenly spaced columns plus the same aligned floor lines.
  - `col3_braced.png`: `col3.png` plus one diagonal brace per bay per storey,
    alternating direction. Same line width.
- Sketch preprocessing is the existing occupancy pipeline (invert, threshold
  0.40). Do not change it.

### 0.4 Recipes and flags (extend `run.py` / `recipe/`)
`recipe/dream_layout.py` is the hybrid recipe. Add these, sharing its settings
(`recipe/preset.py` `PAPER`) unless stated otherwise:
- `--mode {unguided, sketch, semantic, hybrid, dream_only}`
  - `unguided`: compliance only. No CLIP, no sketch, neutral initialization.
  - `sketch`: sketch occupancy prior, CLIP off.
  - `semantic`: CLIP on the projected physical density, no sketch, no dream.
  - `hybrid`: the existing dream-layout path.
  - `dream_only`: the dream stage only; save it, no physics.
- `--sketch PATH`, `--sketch-init {on,off}`, `--sketch-weight {on,off}`,
  `--sketch-weight-end FLOAT`.
- `--prompt-sketch` for a sketch prior plus a CLIP prompt (designer sketch plus
  text).
- `--blend-rho FLOAT` (the semantic dial).
- `--clip-scales` set to any of `g`, `m`, `e` (global, module, element), for
  example `g,m,e`.
  - Global: the whole elevation, letterboxed.
  - Module: square crops tiled exactly on storeys (buildings) or on full-depth
    panels (bridge). Not random positions.
  - Element: a quarter of the module size.
  - Implement the module tiling as new code with a test. Keep the existing
    random-crop path unchanged.
- `--coadapt {on,off}`.
- Topology-optimization knobs for the conventional baseline: `--filter-width`,
  `--penal`, `--beta-max`, `--resolution-scale`, `--seed`.

### 0.5 Output contract (every run, no exceptions)
Write to `results/<run_id>/attempt_<n>/`:
- `physical_density.npy`: final filtered, projected density at full resolution.
- `final.png`, `comparison.png` (as now), `progress.gif`.
- `run.json`: `run_id`, experiment ID, group (`baseline`, `formal`,
  `semantic`, `hybrid`, `conventional`), structure, prompt, sketch, every CLI
  argument, the git commit, `pip freeze` hash, hostname, Slurm job and array
  IDs, start and end times, exit status, final compliance, achieved volume
  fraction, mean density, gray fraction (share of elements with
  0.1 < density < 0.9), compliance after thresholding at 0.5, connected
  components, and wall-clock seconds.
- `DONE`: an empty file written last, only after everything else succeeded.

### 0.6 Slurm scaffolding (`slurm/`)
- `slurm/campaign.tsv`: the manifest. One row per run: `run_id`, then the
  `run.py` arguments. Generate it with a script (`slurm/make_manifest.py`) from
  the experiment table below. Do not hand-edit it.
- `slurm/campaign.sbatch`: one array task per manifest row, on a CPU
  partition, 1 task, 4 CPUs, 16 GB, 6-hour limit, `%K` concurrency cap. The
  task skips rows that already have `DONE`. It activates the env explicitly.
- `slurm/resubmit.py`: scans for rows without `DONE`, resubmits only those
  array indices, and gives up on a row after 3 failed attempts (logging it as
  permanently failed in `CAMPAIGN_LOG.md`, with the log tail).
- `slurm/status.py`: prints counts of done / running / failed / pending per
  experiment.

## Phase 1 — smoke tests (gate)
Run one row from each of: unguided on each structure, sketch, semantic, hybrid,
dream_only and the scale arms. Record the wall-clock time per mode in
`CAMPAIGN_LOG.md`. Check visually that the three unguided baselines give
sensible topology. If any structure gives a degenerate result (disconnected,
all gray, solver error), fix the problem definition before Phase 2.

## Phase 2 — the campaign

Unless stated otherwise: tall building, volume fraction 0.30, seed 12,
`PAPER` settings, coupling fixed (CLIP on the projected physical density).

| ID | Claim | Runs |
|---|---|---|
| B | Baselines | `unguided` × {tall, short, bridge} = 3 |
| F1 | Initialization and weight are both needed | sketches {12, 3} × {init only, weight only, both} = 6 |
| F2 | The formal channel is a dial | sketch 12, weight end {200, 400, 800, 1200, 2000} = 5 |
| F3 | Generality across drawings | `sketch` on {1, 3, 6, 9, 11, 12, col2, col3, col6_grid, col3_braced} = 10 |
| S1 | Scale pilot (gate) | `semantic`, "butterfly", scales {g}, {m}, {e}, {g,m,e} = 4 |
| S1b | Scale replication (only if S1 passes) | same 4 arms × {short, bridge} = 8 |
| S2 | The semantic channel is a dial | `semantic`, "unfurling fern fronds", `blend_rho` {0, 0.25, 0.5, 0.75, 1.0} = 5 |
| S3 | Physics mediates the prompt | `semantic` × {"unfurling fern fronds", "butterfly", "skeletons"} × {tall, short, bridge} = 9, plus "human skull" on tall = 1 |
| H1 | The dream as a formal prior | {`hybrid`, `dream_only`} × the 3 prompts on tall = 6 |
| H2 | Same prompt, different structures | `hybrid`, "unfurling fern fronds" × {short, bridge} = 2 |
| H3 | Sketch plus prompt | `--prompt-sketch` with {12, col3_braced, col6_grid} × the 3 prompts = 9 |
| H4 | Is co-adaptation needed? | `hybrid` with `--coadapt off` × {fern, butterfly} = 2 |
| D | Conventional baseline | `unguided`, 24 Latin-hypercube samples over filter width [1.5, 4], penalization [3, 4], `beta` max [4, 16], resolution scale {0.5, 1}, seed [0, 1000]. Volume fraction fixed at 0.30 |

**S1 gate.** Write the criterion in `CAMPAIGN_LOG.md` *before* viewing the
results: "the {m} arm shows one butterfly instance per storey in at least 3 of
4 storeys". Judge it from `final.png`. Pass: queue S1b. Fail: skip S1b, and log
"two-scale result: element vs global".

**H4 decision.** If co-adaptation on vs off shows no visible difference in
`final.png` and the compliance difference is under 3%, log "cut co-adaptation
from the method". Do not rerun the rest of the campaign; just record it.

Submit everything except S1b at once. Then loop: run `status.py`, run
`resubmit.py`, and check the logs, until every row is `DONE` or permanently
failed.

## Phase 3 — analysis (`analysis/`, commit the scripts and the figures)

1. `analysis/evaluate.py`: one fixed CLIP evaluator (`ViT-B/32`,
   deterministic crops: the whole image letterboxed plus a fixed 3×3 grid of
   crops, no random augmentation). For every `DONE` run, compute and save the
   image embedding, plus the similarity to each of the 4 prompts. Never reuse
   training-time CLIP losses, because different runs used different scoring
   functions.
2. **Cross-prompt matrix:** rows are designs, grouped by prompt (including the
   unguided baselines); columns are the 4 prompts. Report the mean per block,
   and whether each design's own prompt scores highest. Save it as a heatmap
   and a CSV.
3. **Compliance table:** every run's compliance divided by its structure's
   unguided baseline, plus gray fraction, compliance after thresholding, and
   connected components.
4. **Diversity** (tall building only):
   - Groups: conventional (D), formal (F1–F3), semantic (S1–S3 on tall),
     hybrid (H1, H3, H4).
   - Binarize at 0.5, downsample to 32×64, and compute the Tanimoto kernel.
   - Semantic kernel: cosine similarity of the evaluator embeddings.
   - Quality \(q = C_{\text{unguided}} / C\).
   - For each group, report the formal Vendi score, the semantic Vendi score
     and the quality-weighted Vendi score, with equal n per group (n = the
     smallest group, 1000 bootstrap subsamples, 95% confidence intervals).
   - PCA fitted on the pooled downsampled designs, scatter colored by group and
     shaded by \(q\).
5. `analysis/figures.py`: contact sheets per experiment ID (`final.png` grid
   with compliance ratios underneath), the two dial curves (F2: mass-on-occupancy
   vs compliance ratio; S2: evaluator similarity vs compliance ratio), the scale
   figure, and the prompt × structure grid.

## Definition of Done (only then hand back)

- Every manifest row is `DONE`, or permanently failed after 3 attempts with a
  logged reason.
- The S1 gate and the H4 decision are logged.
- Every Phase 3 output exists under `analysis/out/`.
- `REPORT.md` is written for a reader who did not watch the run:
  - what ran, and what failed and why;
  - every default or decision you took, with its rationale;
  - wall-clock and compute totals;
  - the key numbers from each table, with their conditions (structure,
    group, criterion);
  - parked items that need a human.
- Code, manifests, analysis scripts, figures, `CAMPAIGN_LOG.md` and `REPORT.md`
  are committed. `results/` stays uncommitted, but its location on the cluster
  is stated in `REPORT.md`.