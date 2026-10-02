# CAAD Futures 2027 Campaign Log

- 2026-10-01 23:31 UTC: Started Phase 0 on MIT Engaging from clean commit `ac5f545`.
- 2026-10-01 23:31 UTC: Selected the documented defaults: `mit_normal`, CPU-only execution, the `semantopology` Python 3.11 environment, seed 12, and the `PAPER` settings. These keep every reported run on one platform and preserve the preregistered campaign.
- 2026-10-01 23:31 UTC: Kept `OMP_NUM_THREADS=1`; CHOLMOD failures will only be retried as fresh Slurm tasks, with at most three attempts.
- 2026-10-01 23:31 UTC: Configured the campaign script to load `miniforge/23.11.0-0`, activate `semantopology`, and use the pre-downloaded ViT-B/32 and RN50 cache under `$HOME/.cache`.
- 2026-10-01 23:31 UTC: Preregistered the S1 gate before viewing results: "the {m} arm shows one butterfly instance per storey in at least 3 of 4 storeys".
- 2026-10-02 11:09 UTC: Built `semantopology` on login-node miniforge. Kept numpy 1.26.4 (HIPS autograd). Installed torch 2.2.2+cpu / torchvision 0.17.2 from the official CPU wheels because conda-forge has no `pytorch=2.2.2`.
- 2026-10-02 11:09 UTC: Conda `kornia` pulled `libtorch` 2.10; that leftover `lib/libtorch_python.so` made pip Torch fail to import. Removed the orphan and left Torch as pip-only.
- 2026-10-02 11:09 UTC: Pinned `scikit-sparse=0.4.16` and `imagecodecs=2024.12.30` so they stay on numpy 1.x. Export `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` so compute nodes use conda `libstdc++` instead of the host `GLIBCXX_3.4.29` miss.
- 2026-10-02 11:09 UTC: CLIP ViT-B/32 and RN50 already present at `$HOME/.cache/clip`; `campaign.sbatch` points `CLIP_CACHE` / `TORCH_HOME` there.
- 2026-10-02 11:09 UTC: `pytest tests -q` is green on the login node and on `mit_normal` via `srun` job 24642873: 254 passed, 3 warnings, 121 subtests. Env versions: python 3.11.16, numpy 1.26.4, scipy 1.17.1, sksparse 0.4.16, torch 2.2.2+cpu.
- 2026-10-02 11:14 UTC: Phase 1 smoke array indices from `slurm/campaign.tsv` (0-based, header skipped): 0 B/tall/unguided, 1 B/short/unguided, 2 B/bridge/unguided, 14 F3/tall/sketch-1, 29 S1/tall/butterfly/m, 51 S3/tall/fern_fronds, 61 H1/tall/fern_fronds/hybrid, 62 H1/tall/fern_fronds/dream_only.
- 2026-10-02 11:16 UTC: Submitted smoke as Slurm array 24643266 on `mit_normal`.
- 2026-10-02 11:35 UTC: Finished smoke wall-clocks so far (attempt_1): B/tall 984s C=70.3; B/short 1934s C=19.8; B/bridge 1187s C=262.6; F3/sketch-1 1011s C=106.7; S1/butterfly/m 1914s C=71.3; H1/dream_only 284s. S3 fern and H1 hybrid still running.
- 2026-10-02 11:35 UTC: Unguided `final.png` check: tall is a connected four-storey branching frame; short is a left-supported cantilever tree with a right overhang; bridge is a pin-roller two-deck span. None are all-gray or solver-failed. Short reports 2 components and bridge 6, but the images are coherent topologies, so no problem-definition fix. Phase 2 may proceed after the remaining smoke rows finish.
- 2026-10-02 12:18 UTC: Pulled `a2b86ad` (publication-quality renderings, paper prompt `butterfly wing venation`, three-deck exploratory gate). This does not change the FEA grid or loss algebra. `final.png` is now a bilinear 2400-edge presentation; `physical_density.png` stays native-grid.
- 2026-10-02 12:18 UTC: Re-rendered finished smoke `final.png` / `comparison.png` from saved `.npy`. S1 smoke used the old prompt `butterfly` at `S1/tall/butterfly/m`; the official gate is now `S1/tall/butterfly_wing_venation/m` and will be submitted as a new row. In-flight H1 hybrid and S3 fern (job 24643266) still write with the pre-pull code; their arrays will be re-rendered after DONE.
- 2026-10-02 12:20 UTC: `comparison.png` was clipping tall buildings: matplotlib `tight_layout` plus an 8-inch height cap cut the soffit. Replaced the strip with a PIL compose of the full 2400-edge panels and re-rendered finished smoke comparisons.
- 2026-10-02 15:13 UTC: Pulled `3f8e946`. `final.png` is now crisp nearest-neighbour at 2400-edge; `semantic_design.png` is the hardfork path (Torch bilinear short-edge 512, then clamp). Kept the uncropped PIL comparison strip. Physical panels in the strip use nearest; semantic panels use the hardfork raw-z render.
- 2026-10-02 15:13 UTC: Re-rendered all finished smoke images. H1 hybrid and S3 fern are DONE. Slurm queue is empty. Official S1 gate row `butterfly_wing_venation` still needs a fresh submit.
- 2026-10-02 15:38 UTC: Kept campaign fern text as `"fern fronds"` (not `"unfurling fern fronds"`). The plural is clearer for CLIP; S3 semantic already looks like a frond field. Hardfork wording stays a historical note. Existing `fern_fronds` smoke rows remain the official fern arm.
- 2026-10-02 15:42 UTC: Pulled `d53b113` (uncropped comparison strip). Kept comparison semantic panels on the hardfork raw-z path (resize before clamp) and pass unbounded raw z into those panels. Cluster `campaign.sbatch` loads `semantopology` and the CLIP cache.


