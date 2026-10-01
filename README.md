# semantopology_prod

A runnable slice of neural structural optimization: the solver, the design
parameterizations, and the semantic guidance terms. Copied from
`semantopology_hardfork` at `4cfc627bb38896a3eeaf5cb19f2a7d65cbadda6c`.
The five packages sit at the repository root. `runtime.py` pins
`OMP_NUM_THREADS=1` before any native library loads; each package imports
it first. Numerical behavior, loss algebra, and solver logic were not
rewritten.

## Packages

**problem** holds boundary conditions and the named structural problems.
`StructuralParams` builds a grid, loads, and supports; the rest of the
package never invents those.

**physics** is the finite-element solve. Cone filtering, the sparse
stiffness system, autograd primitives, and the `Environment` API that
hands displacements and compliance back to PyTorch live here.

**model** is the design field: a pixel grid, an adaptive-pixel grid that
starts coarse and upsamples, a CNN, and a hybrid of the two. These
modules own the parameterization, not the loss.

**guidance** is every term that steers the field: compliance, CLIP,
grad-match blending of those two, the coadaptive occupancy mask, and the
sketch / motif-layout prior. Grad-match is guidance, not a solver knob.

**optimize** is the step loop. Adam, adaptive Adam, LBFGS, MMA, and
optimality-criteria live here, as do the progressive trainers and the
image helpers that write a run's frames.

## What was left behind

The hardfork keeps the lab archive. This repo does not copy:

- `script/` (drivers, seed images, paper figures, result folders)
- `experiment.py` (GOLDEN / SMOKE presets, Venice image path, sketch and
  motif-layout configs)
- `original/`, `docs/`, `.memory/`
- tests that import `experiment.py`, load `script/` drivers, or read
  saved result folders / replay JSON:
  `test_experiment.py`, `test_venice_parity.py`, `test_clip_motif_scale.py`,
  `test_sketch.py`, `test_training_loop_coverage.py`.
  `tests/test_semantic_prior.py` keeps the synthetic cases only.

`physical_motif_scale_fracs` was inlined into `guidance/loss_semantic_prior.py`
because the coadaptive prior cannot run without it. Sketch code may still
mention `script/resources/input_images/sketches`; that path string is kept
and the images are not copied.

## Reproduce a paper figure

Fern and butterfly are the same recipe. The CLIP text is the only difference,
and it is passed through as written. A language model is not called.

```bash
python run.py --problem multistory_building --clip "unfurling fern fronds"
python run.py --problem multistory_building --clip "butterfly wing venation"
```

Each run writes `results/<slug>/`. `comparison.png` is the physical density,
the dream scaffold, the coadaptive mask, and CLIP saliency. `progress.gif`
is the physics run. `summary.json` records final compliance, weighted CLIP
loss, raw CLIP loss, step count, realized volume, mean physical density,
mass on the scaffold, connectivity, and wall-clock seconds for the dream,
the physics loop, and the whole run, plus the git commit the run started from.

The default device is CPU, including on a machine that has a GPU. That is
the device the paper numbers were produced on.

`--sentence` is for a new idea. It asks a language model for the image
motive only and does not choose a structure. Pass `--problem` either way.

`slurm/dream_layout.sbatch` submits one array task per line of
`slurm/manifest.txt` on MIT Engaging. It calls the same `run.py`. The
`mit_normal_gpu` partition is limited to 6 hours and, at the base allocation,
2 GPUs. `mit_preemptable` allows more, and a preempted job starts over:
the solver does not resume from a checkpoint.

Bridge, cantilever building, and tall building are not defined yet.
`--problem` takes a name that already exists, such as `multistory_building`.
