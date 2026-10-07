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
claim. Do not add experiments that are not listed (Phase 2c lists the
extensions).

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
- Pre-download CLIP weights (`ViT-B/32`, `RN50`, and the evaluator-only
  `ViT-L/14`) on the login node. Set
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
  `three_decker_bridge` adds a middle deck at y=36, and gives the top and
  middle decks vertical rollers at both side walls (X stays fixed only at
  the bottom-left pin). It is available as `--structure bridge3` for an
  exploratory gate; it is not in the manifest.
- Sketches in `inputs/sketches/`. `col3_braced.png` has columns flush with
  the domain edges. Do not regenerate.
- `run.py` modes: `unguided`, `sketch`, `semantic`, `hybrid`, `dream_only`,
  plus `--prompt-sketch`, `--clip-scales {g,m,e,gme}`, `--coadapt`,
  `--blend-rho`, conventional knobs. `--clip-weight` / `--clip-weight-z`
  (semantic only, set together) replace grad-match with fixed weights for C2.
  `--blend-rho-z` (grad-match only; default 0.75) weights raw-z CLIP for A.
  `--volume-fraction` (default 0.3 from the structure) sets the target for V.
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
  progress.gif           # sharp-ink render, every 2nd physics step
  comparison.png
  run.json
  DONE                 # written last, only on success
logs/<jobid>_<array>.out
analysis/out/
  evaluate/embeddings.npy               # ViT-B/32, physical density
  evaluate/similarities.json
  evaluate/embeddings_z.npy             # ViT-B/32, final.png
  evaluate/similarities_z.json
  evaluate/embeddings_vit_l_14.npy      # ViT-L/14, physical density
  evaluate/similarities_vit_l_14.json
  evaluate/embeddings_vit_l_14_z.npy    # ViT-L/14, final.png
  evaluate/similarities_vit_l_14_z.json
  evaluate/*_vit_b_32_laion2b_s34b_b79k*       # LAION ViT-B-32, both views
  evaluate/*_convnext_base_w_laion2b_s13b_b82k*  # LAION ConvNeXt, both views
  tables/cross_prompt_matrix.csv        # one per evaluator x view
  tables/semantic_floor.csv             # one per evaluator x view
  tables/evaluator_view_gap_vit_l_14.csv
  tables/compliance.csv
  tables/weight_headroom.csv
  diversity/energy/                     # cached per-element re-solves
  diversity/tables/diversity.csv
  diversity/tables/diversity_points.csv
  diversity/figures/diversity_structural.png
  diversity/figures/diversity_geometric.png
  diversity/figures/diversity_perceptual.png
  figures/                              # campaign overview figures
```

`final.png` is the declared result for every mode. It is the sharp-ink render
used by the hardfork fern and skeleton figures: resize the unbounded raw design
(`final_design_raw.npy`) to a 512-pixel short edge with Torch bilinear
antialiasing, then clamp and invert (512×1024 for the tall building).
`progress.gif` renders every 2nd physics step (plus the last) the same way,
from each step's native AdaptivePixel grid, at 20 steps per second with a
1 s hold on the final frame. `run.json` records `presentation_source`
(`final_design_raw`; `scaffold` for `dream_only`), `presentation_shape`, and
`presentation_resampling`.

The sharp-ink render is the raw design, not the analysed structure. Use
`physical_density.npy` (or the native `physical_density.png`) for measurements,
connectivity, and exact pixel inspection, and show it beside `final.png` when a
figure makes a structural claim.

Tokens: `tall|short|bridge`, prompt slugs
`fern_fronds|butterfly_wing_venation|skeletons|human_skull`,
`sketch-12`, `g|m|e|gme`, `rho-0.50`, `wend-400`, `hybrid|dream_only|coadapt-off`,
`lhs-00`. H5 is the standard hybrid plus `--gravity-load 0.05`.
S4 slugs: `fern_frond|many_fern_fronds|field_of_ferns|unfurling_fern_fronds`.
C1 slugs: `structure|qzv_xlrp_mnek`. C2 uses the S3 prompt slugs.

### 0.4 Submit
```bash
python slurm/make_manifest.py          # 118 rows; do not hand-edit the TSV
# campaign.sbatch --array is 0-117%24; regenerate if the count changes
sbatch slurm/campaign.sbatch
```
After the S1 gate passes:
```bash
python slurm/make_manifest.py --include-s1b   # appends rows 118-141
sbatch --array=118-141%24 slurm/campaign.sbatch
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
  --group baseline --mode unguided --structure bridge3 --physics-beta-max 8
python run.py --run-id GATE/bridge3/fern_fronds --experiment GATE \
  --group semantic --mode semantic --structure bridge3 --clip "fern fronds" \
  --physics-beta-max 8
```

Optional fern-wording panel (run separately; do not append to the main
manifest). Official campaign fern stays `"fern fronds"`. S4 only runs the
other count/plural texts on tall, hybrid and semantic; the official column
is H1/S3.

```bash
python slurm/make_manifest.py --fern-wording   # slurm/fern_wording.tsv
sbatch --array=0-7 --export=ALL,CAMPAIGN_MANIFEST=slurm/fern_wording.tsv \
  slurm/campaign.sbatch
```

Texts: `"fern frond"`, `"fern fronds"` (reuse H1/S3), `"many fern fronds"`,
`"field of ferns"`, `"unfurling fern fronds"`.

Optional low sketch-weight panel (run separately; do not append to the main
manifest, and do not change the 4000→400 default). Sketch 12 on tall, init
and weight both on, coarse weight 200, fine-grid ends {0, 10, 40, 100, 200}.

```bash
python slurm/make_manifest.py --sketch-weight-low   # slurm/sketch_weight_low.tsv
sbatch --array=0-4 --export=ALL,CAMPAIGN_MANIFEST=slurm/sketch_weight_low.tsv \
  slurm/campaign.sbatch
```

If the extra middle deck produces a clearer and structurally credible section,
run butterfly wing venation and skeletons too, then decide whether `bridge3` replaces
`bridge` in the reported prompt × structure figure. Do not report both as
independent evidence without accounting for the selection gate. Those two
semantic rows are `slurm/bridge3_prompts.tsv` (fern is already
`GATE/bridge3/fern_fronds/beta8`):

```bash
sbatch --array=0-1 --export=ALL,CAMPAIGN_MANIFEST=slurm/bridge3_prompts.tsv \
  slurm/campaign.sbatch
```

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
`converged`, `compliance`, `thresholded_compliance`,
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

**Decision: adopted.** The user judged the beta8 rows (crisp floor trusses
on tall unguided; frond detail kept on the fern hybrid) and switched the
campaign to the projection. `make_manifest.py` appends `--physics-beta-max 8`
to every row, including D. D still samples conventional `--beta-max` (4–16)
on top of that ramp. Every row in `results/` that ran without the ramp
(check `physics_projection_beta_max` in its `run.json`) is superseded:
move it to `results_superseded/` and rerun it. Phase 2 is the 118-row
table below (H5 included) at that recipe.

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
| S1b | Scale replication (only if S1 passes) | same 4 arms × 3 prompts × {short, bridge} = 24, **appended** as rows 118–141 |
| S2 | The semantic channel is a dial | `semantic` × 3 prompts × `blend_rho` {0, 0.25, 0.5, 0.75, 1.0} = 15 |
| S3 | Physics mediates the prompt | `semantic` × 3 prompts × {tall, short, bridge} = 9, plus "human skull" on tall = 1 |
| H1 | The dream as a formal prior | {`hybrid`, `dream_only`} × 3 prompts on tall = 6 |
| H2 | Same prompts, different structures | `hybrid` × 3 prompts × {short, bridge} = 6 |
| H3 | Sketch plus prompt | `--prompt-sketch` with {12, col3_braced, col6_grid} × 3 prompts = 9 |
| H4 | Co-adaptation is load-bearing | `hybrid` `--coadapt off` × 3 prompts on tall = 3. Default hybrid stays **on**. |
| H5 | A small per-pixel self-weight | `hybrid` `--gravity-load 0.05` × 3 prompts × {tall, short, bridge} = 9. Default hybrid stays gravity off. |
| D | Conventional baseline | `unguided`, 24 Latin-hypercube samples on tall over filter width, penalty, Heaviside `--beta-max` (4–16), resolution, seed |

Without S1b: 118 runs. With S1b: 142. Every row also runs the physics
projection ramp to `--physics-beta-max 8`, so every row runs the full 200
steps.

**S1 gate.** Write the criterion in `CAMPAIGN_LOG.md` *before* viewing the
results: "the {m} arm shows one butterfly-wing-venation instance per storey in
at least 3 of 4 storeys". Judge it from
`results/S1/tall/butterfly_wing_venation/m/attempt_*/final.png`.
Pass: queue S1b. Fail: skip S1b, and log "two-scale result: element vs global".

**H4.** Co-adaptation is part of the hybrid method. Compare each
`H4/.../coadapt-off` `final.png` to the matching `H1/.../hybrid` row. Do
not drop co-adaptation from the method.

**H5.** Self-weight is opt-in. Compare each `H5/.../hybrid` `final.png` to
the matching gravity-off hybrid (`H1` on tall, `H2` on short and bridge).

Loop: `python slurm/status.py`, `python slurm/resubmit.py`, until every
row is `DONE` or permanently failed (3 attempts, logged with a log tail).

## Phase 2b — controls (off the main table)

These rows answer two reviewer questions the main table leaves open: how
much of a prompt's effect comes from its meaning, and whether a hand-set
CLIP weight carries over between problems the way the gradient-norm
coupling does. They run at the same recipe and code revision as Phase 2.
Never append them to `slurm/campaign.tsv`; the 0–117 indices stay fixed.

| ID | Claim | Runs |
|---|---|---|
| C1 | The prompt's meaning, not generic CLIP pressure, moves the design | `semantic` × {`"structure"`, `"qzv xlrp mnek"`} × {tall, short, bridge} = 6 |
| C2 | A fixed CLIP weight does not transfer between problems | `semantic` × 3 prompts × {tall, short, bridge} at one fixed pair of weights calibrated on tall = 9 (after S3 tall) |

**C1.** `"structure"` is a neutral subject (it becomes "a minimal ink
drawing of a structure"). `"qzv xlrp mnek"` is a fixed meaningless string.
Each row matches its S3 row except for the prompt.

```bash
python slurm/make_manifest.py --controls   # slurm/control_prompts.tsv, 6 rows
sbatch --array=0-5 --export=ALL,CAMPAIGN_MANIFEST=slurm/control_prompts.tsv \
  slurm/campaign.sbatch
```

Report each C1 row beside the S3 rows on the same structure: `final.png`,
`physical_density.png`, `compliance`, `thresholded_compliance`, and both
Phase 3 evaluators' scores against all three campaign prompts.

**C2.** The semantic coupling weights two CLIP terms: density CLIP and
raw-z CLIP. C2 keeps both terms and replaces each grad-matched weight with
one fixed number, the way a person would hand-tune it on one problem and
reuse it. The calibration rule is fixed now so it cannot be tuned after
the fact:
- Every semantic `run.json` records `coupling`, `clip_weight_mean`, and
  `clip_raw_z_weight_mean` (each the mean over that run's steps).
- `W_DENSITY` is the mean of `clip_weight_mean` over the three S3 tall
  rows; `W_Z` is the mean of their `clip_raw_z_weight_mean`. Compute both
  once, log them in `CAMPAIGN_LOG.md` before any C2 row runs, and never
  change them.
- If the S3 tall `run.json` files predate these fields, resubmit those three
  rows (physics is unchanged; the change only adds logging), log the
  differing `git_commit`, then calibrate from the new attempts.

```bash
python slurm/make_manifest.py --fixed-weight W_DENSITY W_Z   # slurm/fixed_weight.tsv, 9 rows
sbatch --array=0-8 --export=ALL,CAMPAIGN_MANIFEST=slurm/fixed_weight.tsv \
  slurm/campaign.sbatch
```

Each C2 row matches its S3 row except for `--clip-weight` and
`--clip-weight-z`; check that each C2 `run.json` says `coupling: fixed`.
Report, per structure, C2 vs S3 `clip_loss_raw`, `compliance`,
`thresholded_compliance`, and the two weights, and send both `final.png`
files. Either outcome is a result: the claim is that the tall calibration
is right on tall and wrong (too strong or too weak) elsewhere.

## Phase 2c — extensions (off the main table)

Phase 2 and C1 worked, so these rows sharpen the paper's two claims: how
much leeway a prompt gets depends on the structure, and prompting spans
more of the design space than tuning the conventional knobs. All of them
live in one manifest, `slurm/extensions.tsv`, in priority order, so the
array index is the queue order and the seed replicates run last. Same
recipe and code revision rules as Phase 2. Never append them to
`slurm/campaign.tsv`.

| ID | Claim | Runs |
|---|---|---|
| S2b | The dial's slope (the leeway) depends on the structure | S2 on {short, bridge}: 3 prompts × `blend_rho` {0, 0.25, 0.5, 0.75, 1.0} × 2 = 30 |
| A | Which CLIP term carries the style | density-CLIP only (`--blend-rho-z 0`) × 3 prompts on tall = 3 |
| P | Prompts that suit the physics cost less | 9 prompts on tall: structural {tree branches, bone trabeculae, spider web, gothic tracery, honeycomb}, non-structural {clouds, smoke, fur, a cat} |
| N | Is the C1 face specific to one string? | 7 prompts on tall: 4 more random strings, plus the 3 campaign prompts with their letters scrambled within each word |
| L | Nearby meanings give nearby designs | 3 prompts on tall: {bracken, lightning, brick wall} |
| V | More material, more leeway | `--volume-fraction` {0.2, 0.4, 0.5} × {3 prompts, unguided} × {tall, bridge} = 24 |
| M | Prompts span more designs than parameter tuning | 24 varied prompts on tall, matched in count to D |
| R | Variation from the prompt vs from the seed | S3 grid and B on seeds {101, 202, 303, 404}: 12 × 4 = 48 |

148 rows in total.

```bash
python slurm/make_manifest.py --extensions   # slurm/extensions.tsv, 148 rows
sbatch --array=0-147%24 --export=ALL,CAMPAIGN_MANIFEST=slurm/extensions.tsv \
  slurm/campaign.sbatch
```

Every semantic row matches its S3 row except for the prompt, plus the one
flag its panel varies. Every V and R unguided row matches its B row except
for `--volume-fraction` or `--seed`. Do not tune anything per row.

**S2b.** As in S2, `blend_rho` scales only density CLIP. Raw-z CLIP stays at
0.75 in every row, so the `rho-0.00` arm is a raw-z-only run, not an
unprompted one; B is the unprompted reference. Report a per-structure
curve: x = `compliance` relative to that structure's B row, y = the Phase 3
score for the row's own prompt, both evaluators, with S2 (tall) on the same
axes.

**A.** The full term pairing is already on the table: both terms is S3 tall,
raw-z only is S2 `rho-0.00`, neither is B. A adds the missing arm, density
CLIP only. Report the four arms side by side per prompt: `final.png`,
`physical_density.png`, `gray_fraction`, `compliance`, and both evaluator
scores.

**P, N, L.** All on tall, beside S3 tall and C1 tall.
- P: report `compliance` relative to B tall, and the prompt's own score,
  ranked, with the structural and non-structural groups marked.
- N: say whether the C1 face recurs, and for which strings.
- L: the ladder runs fern fronds (S3), bracken, tree branches (P),
  lightning, brick wall (L). Report its rows and columns of the cross-prompt
  matrix for both evaluators.

**V.** The 0.3 arm is already S3 and B on each structure. Report compliance
relative to the unguided row at the same volume fraction. Check that each
`run.json` `volume_fraction` matches its target, and log any row more than
0.01 off.

**M.** One seed (12), 24 prompts listed in `recipe/campaign_spec.py`
(`DIVERSITY_PROMPTS`). These are compared with D's 24 Latin-hypercube
samples for design-space coverage. The coverage measure is still being
defined; for now just make sure both sets finish and are scored by both
evaluators.

**R.** Lowest priority, last in the queue. If the October 9 cutoff is close,
cancel the remaining R tasks rather than delay anything else, and log
which seeds finished.

Tokens: `rho-0.50`, `density-only`, `vf-0.20`, `seed-101`, and prompt slugs
(`prompt_slug`, e.g. `a_cat`, `bone_trabeculae`).

## Phase 2d — G, geometric compatibility

Submit this after `extensions.tsv` drains, or alongside it if the queue has
room. It does not depend on any Phase 2c result.

```bash
python slurm/make_manifest.py --typology   # slurm/typology.tsv, 18 rows
sbatch --array=0-17%18 --export=ALL,CAMPAIGN_MANIFEST=slurm/typology.tsv \
  slurm/campaign.sbatch
```

Six prompts (`TYPOLOGY_PROMPTS`) on all three structures. Every row is an S3
row with a different prompt and nothing else changed. These prompts were
chosen for how their geometry meets a load path, each conflicting with or
suiting a structure in one nameable way, so the predictions below are stated
in advance and the panel either confirms them or does not:

| Prompt | Expected on tall |
|---|---|
| sedimentary rock layers | most expensive; horizontal layering against a tower's vertical path |
| a solid stone wall | expensive; void-free mass spreads material thin under the volume constraint |
| roman aqueduct arches | cheap despite looking unlike a truss; arcades carry load |
| chainmail | cost paid in `gray_fraction`, not compliance; periodic below element scale |
| an obelisk | material pulled to the centre, outer path starved |
| a spiral staircase | helical organisation on a gravity-driven problem |

Report `compliance` relative to the same structure's B row, `gray_fraction`,
`connected_components`, and the own-prompt score from both evaluators.

**The cross-structure comparison is the point of the panel**, so report it as
a prompt × structure table rather than three separate lists. An arcade and a
horizontal deck are native to a span and awkward on a tower, so if a prompt
is cheap on bridge and expensive on tall, geometric compatibility is a
relation between prompt and structure rather than a property of the wording.
Say explicitly which prompts changed rank between structures.

## Phase 2f — F2b, loosening a rigid sketch prior

Ten rows on tall. Submit whenever there is queue room; nothing depends on it.

```bash
python slurm/make_manifest.py --loose-sketches  # slurm/loose_sketches.tsv, 10 rows
sbatch --array=0-9%10 --export=ALL,CAMPAIGN_MANIFEST=slurm/loose_sketches.tsv \
  slurm/campaign.sbatch
```

`col3_braced.png` and `col6_grid.png` come out rigid in F3 and H3 — the design
keeps the sketch's own geometry instead of negotiating with it. Both have only
ever run at the default `sketch_weight_end` of 400, and `WEIGHT_ENDS` holds
only one rung below that, so the campaign has never actually tested a loose
prior on its two most regular sketches.

The prior is a ramp: `sketch_weight` runs from `sketch_weight_start` (4000) on
the coarsest AdaptivePixel grid to `sketch_weight_end` at full resolution. Two
levers can cause rigidity and the panel separates them.

- **End ladder** (8 rows): both sketches at end ∈ {0, 50, 100, 200}. `wend-0`
  keeps the full coarse-grid prior and lets it decay to nothing, so the sketch
  sets the global posture and physics finishes the design unconstrained.
- **Start sweep** (2 rows): `col3_braced` at start ∈ {1000, 2000}, end left at
  the default 400. New flag `--sketch-weight-start`.

**Stated before the runs, so the outcome can be judged rather than narrated:**
the coarse grid is where AdaptivePixel settles topology, so if the braced
pattern is locked in there, loosening the end can only thin the members, not
rearrange them. The prediction is that the end ladder changes member thickness
and gray fraction while leaving the connected-component count and the brace
pattern essentially intact, and that the start sweep is what actually changes
the topology. If instead the end ladder reorganizes the structure, that
prediction is wrong and the rigidity was never about the coarse grid — say so
plainly.

Report, per row: compliance, gray fraction, connected components, and a visual
call on whether the sketch's geometry survived. Then say which lever moved the
design, because that is the finding, not the individual images. Note that
`wend-0` is not the same run as F1's `--sketch-weight off` arm: that one has no
prior at any stage, while this one has the full prior at the coarse grid.

## Phase 2e — S2c, the dial at more seeds

Nine rows, cheapest panel in the campaign. Submit it whenever the queue has
room; nothing depends on it.

```bash
python slurm/make_manifest.py --dial-seeds   # slurm/dial_seeds.tsv, 9 rows
sbatch --array=0-8%9 --export=ALL,CAMPAIGN_MANIFEST=slurm/dial_seeds.tsv \
  slurm/campaign.sbatch
```

`fern fronds` on tall at ρ ∈ {0, 0.5, 1} and seeds 101, 202, 303. S2 already
ran these three dial settings at the campaign seed, so together they give four
trajectories. The claim being tested is that the dial's *shape* survives
reinitialization: compliance and gray fraction should rise monotonically with ρ
in every seed, with the steepest step between 0.5 and 1.

Report it as a band — per ρ, the mean and range across the four seeds — and say
whether the monotonicity holds in all four or only on average. A single seed
reordering the dial is worth stating plainly; it would mean the 4.1 curve is
one sample rather than a trend.

**Scheduling.** These nine rows and the tail of R compete for the same queue
time. R's unguided tall rows carry the clustering claim in 4.5, so if the queue
cannot take both, finish R first and say in `REPORT.md` that S2c was dropped.

## Phase 2g — X, meaning against surface form

Twelve rows on tall. This panel answers the one campaign question nothing
else does, so prioritise it above G, S2c and F2b.

```bash
python slurm/make_manifest.py --meaning   # slurm/meaning.tsv, 12 rows
sbatch --array=0-11%12 --export=ALL,CAMPAIGN_MANIFEST=slurm/meaning.tsv \
  slurm/campaign.sbatch
```

The N panel established that compliance, grey fraction and component count
cannot tell a meaningful prompt from a meaningless one: those scalars measure
the cost of CLIP pressure, not fidelity. And a similarity score cannot settle
it either, because guidance maximises a similarity — scoring the result with
one partly guarantees the answer. This panel measures geometry instead.

Two conditions at seeds 101, 202 and 303:

- **paraphrase** — `bracken leaves` for `fern fronds`, `bones` for
  `skeletons`. Each shares no word with the prompt it restates.
- **scrambled** — `rnef dsnorf` and `ntseleosk`, the same prompts with their
  letters permuted inside each word. Same letters, same word lengths, no
  meaning. N already ran these, but at the campaign seed only, and one design
  cannot show whether a condition clusters.

`butterfly wing venation` is deliberately absent: it has no natural paraphrase
sharing none of its words, and a partial overlap would forfeit the logic of
the comparison.

**Stated before the runs:** if the words' sense is what moves the geometry,
the paraphrase designs should sit nearer the centroid of that prompt's own
designs than the scrambled ones do, pooled across both families at
p < 0.05. If the gap is absent or reversed, the honest reading is that CLIP is
responding to the string rather than its sense, and the paper should say so
rather than fall back on the similarity numbers.

Note the originals need no new runs: S3 supplies the campaign seed and R all
four replicate seeds for both prompts. That is also why R matters twice over.

## Phase 2h — X, second wave of seeds

Twelve more rows on tall, same two conditions as Phase 2g at three further
seeds. Submit these; they are the last runs the paper needs.

```bash
python slurm/make_manifest.py --meaning-seeds   # slurm/meaning_seeds.tsv, 12 rows
sbatch --array=0-11%12 --export=ALL,CAMPAIGN_MANIFEST=slurm/meaning_seeds.tsv \
  slurm/campaign.sbatch
```

The first wave worked: pooled over both families the paraphrase lands nearer
the original prompt's designs than the scramble does, at p = 0.0008 in the
perceptual measure. But each family on its own returned p = 0.030, which is
exactly the floor a permutation can reach with three paraphrase and four
scrambled designs — both families were pinned at the smallest p available
rather than measured. Six seeds per condition takes that floor to 0.0006 and
lets each family stand without the pooling.

This is a separate manifest from `meaning.tsv` so the finished rows are not
resubmitted. Do not merge the two files; the run ids are disjoint by design
and a test enforces it.

## Phase 2i — three reruns for the weight-cap number

Three rows, and they close an objection rather than add a claim.

```bash
python slurm/make_manifest.py                       # 118-row campaign.tsv
# Rename DONE to DONE.superseded on the three S3 tall attempts first, so
# they rerun and write the new field. Designs stay on disk.
sbatch --array=51-53 slurm/campaign.sbatch
```

Grad-match sets the CLIP weight as a ratio of gradient norms, capped at 2000.
Mean weights sit far below that — 7% to 59% across G, S2c and X — but of the
39 runs that logged a peak, every one of the 36 with a CLIP term hit exactly
2000.0. The three that did not are the rho=0 arms with no CLIP term at all.
A peak pinned to the cap under means that low is the signature of a brief
spike, and the likely cause is in `GradNormEma`: `reset()` fires at each
AdaptivePixel upsample and the next observation seeds the averages directly,
so one step's raw norm ratio sets the weight unsmoothed.

But that is inference, not measurement, and no per-step log survives on disk.
Until it is measured, "the weight is derived rather than hand-tuned" has a
hole in it: it is derived except at the ceiling, where a hand-chosen 2000
does the clipping. `clip_weight_cap_share` now records the share of steps at
the cap, so three reruns turn the objection into a number. Report it.

If the share comes back near zero the claim stands as written. If it is
large, say so plainly and describe the coupling as grad-matched below a fixed
ceiling, which is what it would then be.

## Phase 3 — analysis

### Run it early, on partial results, before the campaign finishes

Do this now, as soon as any panel has finished rows. Do not wait for a
complete campaign. None of these scripts has ever run against real campaign
output, so the first full run is also their first integration test — and if
that run is on the last day, a failure has nowhere to go.

```bash
python analysis/evaluate.py --models ViT-B/32   # one evaluator, fastest check
python analysis/figures.py
python analysis/diversity.py solve
python analysis/diversity.py report
```

Half-empty tables are the expected outcome and are not a failure. What this
pass is for is the questions that only real data answers:

- Do the two LAION checkpoints load on a compute node, or only on the login
  node? Run the full `evaluate.py` once the single-evaluator pass works.
- How long does one `diversity.py solve` re-solve take, and how many designs
  does the 5% compliance check exclude? If the exclusion rate is high the
  tolerance is wrong, and that is better known now.
- Does any re-solve hit a CHOLMOD factorization failure? Each design is
  solved in its own subprocess precisely so one failure cannot poison the
  rest, but that path has never been exercised on cluster data.
- Does `diversity.py report` have enough designs per comparison set to
  bootstrap, and does the quality band hold anything at all?

Report what broke and what it cost in `REPORT.md`, then re-run Phase 3
normally when the campaign completes. A fix found here is cheap; the same fix
found on 9 October is not.

```bash
python analysis/evaluate.py
python analysis/figures.py
python analysis/diversity.py solve
python analysis/diversity.py report
python analysis/specificity.py report
```
Four evaluators score every finished run on the same deterministic crops, a
letterboxed full frame plus a 3×3 grid of square crops (no random
augmentation). They form a ladder away from the backbone guidance uses:
`ViT-B/32` (OpenAI) is that backbone under a protocol training never uses;
`ViT-L/14` (OpenAI) is a scale training never loads;
`ViT-B-32/laion2b_s34b_b79k` holds the architecture fixed and changes the
training corpus; `convnext_base_w/laion2b_s13b_b82k` changes both. The last
two load through `open_clip` and use each checkpoint's own input resolution
and pixel normalization, not CLIP's. Each evaluator scores both the saved
physical density and `final.png`, the sharp-ink raw-z view shown to the
reader. Never reuse training-time CLIP losses.

The point of the ladder is a claim that survives it: report where the prompt
ranking agrees across all four and where it does not. A ranking that holds
only on `ViT-B/32` is a property of that model, not of the designs.

The weights must exist on disk before the job runs. On the login node, where
there is internet:

```bash
python -c "import open_clip; [open_clip.create_model(a, pretrained=p) for a, p in
  (('ViT-B-32','laion2b_s34b_b79k'),('convnext_base_w','laion2b_s13b_b82k'))]"
python -c "import clip; clip.load('ViT-L/14', device='cpu')"
```

If an evaluator still cannot load on the compute node, run the rest with
`--models` and say in `REPORT.md` which one was dropped; do not drop it
silently. A SigLIP evaluator was considered and left out: its tokenizer needs
`transformers`, which this environment does not have.

`diversity.py solve` re-solves every selected tall-building density in a
separate process and caches its element strain energy. A failed CHOLMOD
factorization must not be caught and retried in-process. The saved run
compliance is one update behind the saved density, so the script accepts a
5% relative difference and excludes larger mismatches. `diversity.py report`
fits each PCA once on the union of sets: B plus unguided R seeds, D, the
unguided V volume rows, M, and the lighter S3/P/L comparison set. It writes
structural (load-path), geometric (physical-density), and perceptual
(ViT-L/14 `final.png`) maps, plus a binary-structure check.

`diversity.csv` carries two defences of the diversity claim beside the raw
spread. The `conventional union` row pools every set a conventional pipeline
can reach by turning its own knobs — seeds, parameters, volume — so a reader
cannot object that each baseline is narrow only because it varies one thing
at a time. The second `band` value (default `quality 0.75-1.05`) repeats every
set inside a common quality range, which answers the charge that a wider
spread is just a worse one. Quality is a compliance ratio against the unguided
baseline, so it is only defined at the 0.3 budget: the volume rows drop out of
the banded table by construction, and that is expected, not a bug. Report both
the unbanded and the banded numbers. If the band holds too few designs to
bootstrap, widen it with `--quality-band LOW HIGH` and say what you used.

`specificity.py report` runs the two tests behind the semantic claim. Neither
scores a design by its similarity to the guiding prompt, which is the point:
guidance maximises such a similarity, so judging the result with one partly
guarantees the answer.

Each test runs in up to three representations, and the table reports them in
the order `usable_measures` returns — most sensitive first, which is also
most circular first:

- `perceptual`, ViT-L/14 image embeddings of the render. A model, so say so;
  but not the model guidance climbs (ViT-B/32), and no text enters the test
  at any point. For a claim about appearance this is the sensitive
  instrument and the headline row.
- `geometric`, the physical density map. Consults no model whatsoever, so a
  positive result here is the strongest answer to the circularity worry. It
  measures where material sits, which a prompt constrains far more weakly
  than it constrains appearance, so expect smaller margins.
- `structural`, the re-solved load path. Only present for runs a `solve`
  covered; a measure too few designs share is dropped rather than reported.

`specificity.csv` asks whether designs guided by one prompt cluster apart
from another prompt's, with a label-permutation null and a leave-one-out
nearest-neighbour accuracy against the chance rate implied by the label mix
(three prompts, so about 0.33, not 0.5). Report this as specificity, NOT as
meaning: a meaningless string has a fixed text embedding too, so it would
also pass.

`meaning.csv` is the test that separates them, comparing the paraphrase and
scrambled conditions by distance to the centroid of each prompt's own
designs. `butterfly wing venation` appears in `specificity.csv` but not here:
it has no word-disjoint paraphrase, and a family missing a contrast condition
is dropped rather than printed with blank columns. Once Phase 2h lands, read
the per-family rows directly; before it, read `pooled (stratified)`, because
three runs per condition floors a family's own p at 0.03.

To fill in the structural measure, run `python analysis/specificity.py solve`
first. That is optional — the perceptual and geometric results are the
reported ones, and the energy cache built for the diversity selection does
not cover these runs.

`formal.csv` is the formal-channel and composition table: `mass_on_scaffold`
is the share of material landing on the supplied drawing, and the only
quantitative answer to whether a design followed it. Runs with no drawing are
absent because the field is null for them, not because experiments are
filtered by name, so a new sketch panel appears here without any change.
Alongside it are the connectivity metrics every run already recorded under
`validity` but which no table surfaced: floating mass, spanning mass, and the
two connectivity flags. `compliance.csv` now carries those as well.

On the weight cap: every campaign run that logged a peak hit it exactly, so
`clip_weight_max` alone cannot distinguish one clipped step from a run spent
entirely at the ceiling. New runs record `clip_weight_cap_share`, the share of
steps that reached the cap, which appears as `cap_share` in
`weight_headroom.csv`. Rows from before the field existed leave it blank.

Outputs land under `analysis/out/`. In `REPORT.md`, report each cross-prompt
table per evaluator and view, the z-minus-density evaluator gap, and
`semantic_floor.csv` — which gives, per prompt, how much closer its own
designs sit to it than designs guided by a meaningless string or not guided at
all. That gap is the semantic claim; the mechanical scalars cannot make it,
because compliance and gray fraction move the same way under a nonsense
prompt. Also report `weight_headroom.csv`: the grad-matched weight is
truncated at 2000, so state whether any run reached it. Runs recorded before
`clip_weight_max` was logged leave that column empty — say so rather than
reading the mean as a maximum. Report all three diversity maps,
`diversity/tables/diversity.csv`, excluded re-solves, and where the physical
and binary structural maps disagree.

## Definition of Done (only then hand back)

- Every manifest row is `DONE`, or permanently failed after 3 attempts with a
  logged reason.
- The S1 gate and the H4 comparison are logged.
- Every C1 and C2 row is `DONE` (or permanently failed), and the C2
  calibration weights are logged.
- Every Phase 2c row except R is `DONE` (or permanently failed). R rows that
  did not finish before the cutoff are listed in `REPORT.md`.
- Every Phase 2d (G) row is `DONE` (or permanently failed), reported as a
  prompt × structure table with the stated predictions marked kept or broken.
- Every Phase 2e (S2c) row is `DONE`, permanently failed, or explicitly
  dropped for queue time, with the dial reported as a four-seed band.
- Every Phase 2f (F2b) row is `DONE`, permanently failed, or explicitly
  dropped, with a verdict on which lever — end or start — loosened the prior,
  and the stated prediction marked kept or broken.
- Every Phase 2g (X) row is `DONE` or permanently failed, with the pooled
  paraphrase--scramble gap and its p-value reported, and the stated
  prediction marked kept or broken. This panel is not droppable for queue
  time: no other run answers whether meaning does the work.
- Every Phase 2h row is `DONE` or permanently failed, and `meaning.csv`
  reports each family's own p-value alongside the pooled one.
- The three Phase 2i reruns are `DONE` and `cap_share` is reported, with a
  verdict on whether the derived-weight claim stands as written.
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
