# semantopology

A runnable slice of neural structural optimization: the solver, the design
parameterizations, and the semantic guidance terms. Copied from
`semantopology_hardfork` at `4cfc627bb38896a3eeaf5cb19f2a7d65cbadda6c` and rearranged into five packages.
Imports were rewritten; numerical behavior, loss algebra, and solver logic
were not.

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
  `test_sketch.py`, `test_semantic_prior.py`, `test_training_loop_coverage.py`

`physical_motif_scale_fracs` was inlined into `guidance/loss_semantic_prior.py`
because the coadaptive prior cannot run without it. Sketch code may still
mention `script/resources/input_images/sketches`; that path string is kept
and the images are not copied.

A single `run.py` and the three named structures (bridge, cantilever
building, tall building) are a later slice.
