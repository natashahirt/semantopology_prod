# AGENTS.md — CAAD Futures 2027 experiment campaign

You are picking up `semantopology_prod` on the MIT Engaging cluster. The
code, sketches, problems, flags, manifest generator, Slurm scripts, and
analysis scripts are already in this repo. **Do not reimplement them.**
Your job is to build the environment, submit the jobs, resubmit failures,
run analysis, and write `REPORT.md`. **Do not stop or hand back until the
Definition of Done (bottom of this file) is met.** Do not ask the user
questions mid-campaign: when a decision comes up, take the documented
default, log it in `CAMPAIGN_LOG.md`, and keep going.

Full paper deadline: October 16, 2026 (AoE). All runs must be done by
October 9, 2026.

## What the paper argues (so you know what matters)

Designer intent enters topology optimization through two channels, formal (a
sketch is a spatial prior on where the fixed volume budget goes) and semantic (a
CLIP text prompt). The strength of each channel is a continuous, tunable setting.
A hybrid uses a CLIP "dream" as the formal prior, then **co-adapts** density and
that occupancy map together during physics. Unlike image generators, every
design is shaped by the physics gradient. Each experiment below supports one
claim. Do not add experiments that are not listed.

## Hard rules

1. **One platform for every reported run.** Every run, including baselines, uses
   the same partition, environment, device, and code revision:
   `--device cpu`. Never mix devices. `slurm/campaign.sbatch` is already
   CPU-only.
2. **Never unset `OMP_NUM_THREADS=1`.** `runtime.py` pins it. Without it, CHOLMOD
   segfaults intermittently, with no traceback.
3. **CHOLMOD cannot be retried in-process.** After a CHOLMOD error, the next
   solve in the same process segfaults. Every run is its own Slurm task.
   Retry only by resubmitting (`python slurm/resubmit.py`).
4. **Keep numpy below 2.0** (the HIPS `autograd` package requires it).
5. **Do not change existing physics, loss algebra, or defaults.** If a smoke
   test shows a degenerate structure, fix only the new problem definition.
   `pytest tests -q` must stay green.
6. **Do not commit `results/` or `logs/`** (they are gitignored). Commit
   analysis outputs, `CAMPAIGN_LOG.md`, and `REPORT.md`.
7. **No force-pushes, no deleting results.** A rerun writes to a new
   `attempt_<n>/`. Give up after 3 attempts.

## Phase 0 — cluster setup (no code to write)

### 0.1 Environment
- Build a Linux conda env named `semantopology` with Python 3.11, torch 2.2.2
  (CPU build), numpy 1.26.x, scipy, and scikit-sparse (SuiteSparse/CHOLMOD),
  plus `requirements.txt`.
- Pre-download CLIP weights (`ViT-B/32`, `RN50`) on the login node. Set
  `CLIP_CACHE` / `TORCH_HOME` in `slurm/campaign.sbatch` (the commented
  lines). Compute nodes may have no internet.
- Uncomment the `module load` / `source activate` lines in
  `slurm/campaign.sbatch` for this cluster.
- `pytest tests -q` must pass on a compute node (`srun`), not just the login
  node.

### 0.2 What is already implemented
- Problems: `tall_building` (alias of `multistory_building` 128×256,
  interval 64), `short_cantilever_building` (300×150, loads at y=0,50,100,
  Y-support on the left 70%, X-fix on the left wall; AdaptivePixel
  `resize_num=1` because 150 is not divisible by 4),
  `double_decker_bridge` (448×72, pin-roller, UDL on y=0 and y=71).
  `three_decker_bridge` adds a middle deck at y=36 and is available as
  `--structure bridge3` for an exploratory gate; it is not in the manifest.
- Sketches in `inputs/sketches/`. `col3_braced.png` has columns flush with
  the domain edges. Do not regenerate.
- `run.py` modes: `unguided`, `sketch`, `semantic`, `hybrid`, `dream_only`,
  plus `--prompt-sketch`, `--clip-scales {g,m,e,gme}`, `--coadapt`,
  `--blend-rho`, conventional knobs.
- CLIP modules are one centered 64×64 tile per tall-building storey, two
  centered 50×50 tiles per short-building storey, and full-depth square
  panels for the bridge. Element crops remain square and sit inside each
  module. This lives in `guidance/clip_scales.py`. RandomResizedCrop is
  unchanged and is the default semantic/hybrid scorer; `--clip-scales`
  replaces it for S1.
- Output contract and folder layout: see below. `slurm/make_manifest.py`
  writes `slurm/campaign.tsv`.

### 0.3 Folder layout (already the `run.py` contract)

```
results/<experiment>/<structure>/<…tokens…>/attempt_<n>/
  physical_density.npy
  physical_density.png  # exact native-grid raster of the structure
  final.png              # sharp-ink paper render of the final raw design
  progress.gif           # sharp-ink render of every physics step
  comparison.png
  run.json
  DONE                 # written last, only on success
logs/<jobid>_<array>.out
analysis/out/
  evaluate/embeddings.npy
  evaluate/similarities.json
  tables/cross_prompt_matrix.csv
  tables/compliance.csv
  tables/vendi.csv
  figures/
```

`final.png` is the declared result for every mode. It is the sharp-ink render
used by the hardfork fern and skeleton figures: resize the unbounded raw design
(`final_design_raw.npy`) to a 512-pixel short edge with Torch bilinear
antialiasing, then clamp and invert (512×1024 for the tall building).
`progress.gif` renders every recorded physics step the same way, from each
step's native AdaptivePixel grid. `run.json` records `presentation_source`
(`final_design_raw`; `scaffold` for `dream_only`), `presentation_shape`, and
`presentation_resampling`.

The sharp-ink render is the raw design, not the analysed structure. Use
`physical_density.npy` (or the native `physical_density.png`) for measurements,
connectivity, and exact pixel inspection, and show it beside `final.png` when a
figure makes a structural claim.

Tokens: `tall|short|bridge`, prompt slugs
`fern_fronds|butterfly_wing_venation|skeletons|human_skull`,
`sketch-12`, `g|m|e|gme`, `rho-0.50`, `wend-400`, `hybrid|dream_only|coadapt-off`,
`lhs-00`.

### 0.4 Submit
```bash
python slurm/make_manifest.py          # 109 rows; do not hand-edit the TSV
# campaign.sbatch --array is 0-108%24; regenerate if the count changes
sbatch slurm/campaign.sbatch
```
After the S1 gate passes:
```bash
python slurm/make_manifest.py --include-s1b   # appends rows 109-132
sbatch --array=109-132%24 slurm/campaign.sbatch
```

Concurrency: ORCD documents `mit_normal` at 12 h maximum wall time and a base
limit of 96 cores per user (256 on a Standard account). Each task asks for
4 CPUs, so `%24` uses 96 cores. Confirm the cap before raising it:
`sacctmgr show assoc user=$USER format=partition,qos,grptres,maxjobs` and
`scontrol show partition mit_normal`; log what you find in `CAMPAIGN_LOG.md`.

Stay on `mit_normal`. `mit_normal_gpu` allows only 2 GPUs per user, so it runs
far fewer rows at once than CPU, and the environment has CPU-only Torch.

Rows finished before commit "Reuse grad-match term gradients" must be rerun.
That commit changes CLIP-guided rows (semantic, hybrid, prompt-sketch, and S1)
at the rounding level; unguided, sketch, and `dream_only` rows are unchanged.
Move the affected `results/<row>/` directories to `results_superseded/`
before resubmitting so every reported row comes from one code revision.

Optional three-deck gate (run separately; do not append to the main manifest):

```bash
python run.py --run-id GATE/bridge3/unguided --experiment GATE \
  --group baseline --mode unguided --structure bridge3
python run.py --run-id GATE/bridge3/fern_fronds --experiment GATE \
  --group semantic --mode semantic --structure bridge3 --clip "fern fronds"
```

If the extra middle deck produces a clearer and structurally credible section,
run butterfly wing venation and skeletons too, then decide whether `bridge3` replaces
`bridge` in the reported prompt × structure figure. Do not report both as
independent evidence without accounting for the selection gate.

### 0.5 Projection gate (run before Phase 2; the user decides)

Unguided smoke runs leave 19–26% of the domain gray, and their thresholded
compliance is about 10⁷. The structures lean on filter-blurred material to
reach the loaded floors. `--physics-beta-max 8` ramps a Heaviside projection
on the physics density from beta 1 to 8 over the run, and it always runs
the full 200 steps. The default (0) is the current physics. Run all four rows
at the same commit:

```bash
sbatch --array=0-3 --export=ALL,CAMPAIGN_MANIFEST=slurm/projection_gate.tsv \
  slurm/campaign.sbatch
```

Report a table for the four rows with these `run.json` fields: `steps`,
`converged`, `compliance`, `thresholded_compliance` (unguided only),
`load_on_solid_fraction`, `gray_fraction`, `validity.component_count`,
`validity.floating_mass_fraction`, `clip_loss_raw` (hybrid only),
`physics_projection_beta_final`, and `wall_clock_seconds`. Also send each
row's `final.png` and `physical_density.png`.

Criteria for switching the campaign to `--physics-beta-max 8`, fixed before
the runs:
- `gray_fraction` below 0.05 on both beta8 rows;
- unguided beta8 `load_on_solid_fraction` at least 0.99;
- hybrid beta8 `clip_loss_raw` no more than 0.005 above hybrid beta0;
- the user judges that the beta8 fern keeps its frond detail.

Do not submit Phase 2 until the user has made this call. If the projection
is adopted, it applies to every row, and finished rows are rerun.

## Phase 1 — smoke tests (gate)
Run, on a compute node, one row from each of: unguided on each structure,
sketch, semantic, hybrid, dream_only, and one S1 scale arm. Easiest: submit
those array indices from the TSV (B/*, one F3, one S3, H1 hybrid, H1
dream_only, S1/tall/butterfly_wing_venation/m). Record wall-clock per mode in
`CAMPAIGN_LOG.md`. Check that the three unguided `final.png` files are
sensible topology. If a structure is degenerate (disconnected, all gray,
solver error), fix that problem definition before Phase 2.

## Phase 2 — the campaign

Unless stated otherwise: tall building, volume fraction 0.30, seed 12,
`PAPER` settings, coupling = CLIP on the projected physical density.

**Prompts.** Whenever a run has a CLIP prompt, use all three:
`"fern fronds"`, `"butterfly wing venation"`, `"skeletons"`. Exceptions:
formal (F) and
conventional (B, D) have no prompt; `"human skull"` is S3 on tall only;
the S1 **gate** is judged on butterfly-wing-venation `{m}`.

The hybrid algorithm and numeric recipe match the hardfork skeleton recipe,
but two campaign texts are intentional wording changes: hardfork's archived
fern result used `"unfurling fern fronds"` and its dual-CLIP skeleton
diagnostic used `"human skeleton"`. Therefore butterfly can replay the
hardfork path exactly; fern and skeleton use the same method but are not
expected to produce bit-identical arrays.

| ID | Claim | Runs |
|---|---|---|
| B | Baselines | `unguided` × {tall, short, bridge} = 3 |
| F1 | Initialization and weight are both needed | sketches {12, 3} × {init only, weight only, both} = 6 |
| F2 | The formal channel is a dial | sketch 12, weight end {200, 400, 800, 1200, 2000} = 5 |
| F3 | Generality across drawings | `sketch` on {1, 3, 6, 9, 11, 12, col2, col3, col6_grid, col3_braced} = 10 |
| S1 | Scale pilot (gate) | `semantic` × 3 prompts × scales {g}, {m}, {e}, {g,m,e} on tall = 12 |
| S1b | Scale replication (only if S1 passes) | same 4 arms × 3 prompts × {short, bridge} = 24, **appended** as rows 109–132 |
| S2 | The semantic channel is a dial | `semantic` × 3 prompts × `blend_rho` {0, 0.25, 0.5, 0.75, 1.0} = 15 |
| S3 | Physics mediates the prompt | `semantic` × 3 prompts × {tall, short, bridge} = 9, plus "human skull" on tall = 1 |
| H1 | The dream as a formal prior | {`hybrid`, `dream_only`} × 3 prompts on tall = 6 |
| H2 | Same prompts, different structures | `hybrid` × 3 prompts × {short, bridge} = 6 |
| H3 | Sketch plus prompt | `--prompt-sketch` with {12, col3_braced, col6_grid} × 3 prompts = 9 |
| H4 | Co-adaptation is load-bearing | `hybrid` `--coadapt off` × 3 prompts on tall = 3. Default hybrid stays **on**. |
| D | Conventional baseline | `unguided`, 24 Latin-hypercube samples on tall |

Without S1b: 109 runs. With S1b: 133.

**S1 gate.** Write the criterion in `CAMPAIGN_LOG.md` *before* viewing the
results: "the {m} arm shows one butterfly-wing-venation instance per storey in
at least 3 of 4 storeys". Judge it from
`results/S1/tall/butterfly_wing_venation/m/attempt_*/final.png`.
Pass: queue S1b. Fail: skip S1b, and log "two-scale result: element vs global".

**H4.** Co-adaptation is part of the hybrid method. Compare each
`H4/.../coadapt-off` `final.png` to the matching `H1/.../hybrid` row. Do
not drop co-adaptation from the method.

Loop: `python slurm/status.py`, `python slurm/resubmit.py`, until every
row is `DONE` or permanently failed (3 attempts, logged with a log tail).

## Phase 3 — analysis
```bash
python analysis/evaluate.py
python analysis/figures.py
```
Evaluator is a fixed `ViT-B/32` with a letterboxed full frame plus a 3×3
grid of square crops (no random augmentation). Never reuse training-time
CLIP losses. Outputs land under `analysis/out/`.

## Definition of Done (only then hand back)

- Every manifest row is `DONE`, or permanently failed after 3 attempts with a
  logged reason.
- The S1 gate and the H4 comparison are logged.
- Every Phase 3 output exists under `analysis/out/`.
- `REPORT.md` is written for a reader who did not watch the run:
  - what ran, and what failed and why;
  - every default or decision you took, with its rationale;
  - wall-clock and compute totals;
  - the key numbers from each table, with their conditions (structure,
    group, criterion);
  - parked items that need a human.
- Analysis figures, `CAMPAIGN_LOG.md` and `REPORT.md` are committed.
  `results/` stays uncommitted; state its cluster path in `REPORT.md`.
