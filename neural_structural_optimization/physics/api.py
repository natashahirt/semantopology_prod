"""API interface for structural optimization.

This module provides a clean interface between physics engine and neural networks,
handling problem parameter setup, boundary conditions, forces, and constraints.
"""

# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=missing-docstring

"""
overview:
- clean interface between physics engine and neural networks
- problem parameter setup (boundary conditions, forces, constraints)
- Environment class wraps physics calcs
- This implementation exclusively uses the legacy autograd physics backend
"""

from typing import Any, Dict, Tuple, Union
import autograd.numpy as np
import numpy as _np
import autograd.core

# Optional torch boundary conversions
try:
  import torch as _torch  # type: ignore
except Exception:
  _torch = None

# Force legacy backend
_HAS_PYFANTOM = False

# Legacy backend (autograd-based)
from neural_structural_optimization.physics import physics
from neural_structural_optimization.problem.problems import resolve_discretization_value

def _to_numpy(x, dtype=np.float64):
    """Convert torch.Tensor / list / np.array to np.ndarray(dtype), no copy if possible."""
    if _torch is not None and _torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
  
    if hasattr(x, "values"): # xarray.DataArray / Dataset-like
        return np.asarray(x.values, dtype=dtype)

    # Use 'np.asarray' (autograd) to preserve tracking if 'x' is an ArrayBox
    if isinstance(x, (np.ndarray, _np.ndarray)) and x.dtype != object:
        return np.asarray(x).astype(dtype, copy=False)

    if isinstance(x, (list, tuple)) and len(x) == 1: # 1-element lists/tuples that wrap an array
        return _to_numpy(x[0], dtype=dtype)

    # Use 'np.array' (autograd) to ensure the result is differentiable if inputs are
    arr = np.array(x, dtype=dtype)
    if arr.dtype == object:
        raise TypeError(f"Expected a numeric array/tensor; got sequence of objects: {type(x)}")
    return arr

def _args_to_numpy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure physics args are NumPy-native (arrays/scalars)."""
    out = dict(args)

    # arrays we expect to be array-like
    for key in ("mask", "freedofs", "fixdofs", "forces"):
        if key in out:
            # freedofs/fixdofs should be int64
            if key in ("freedofs", "fixdofs"):
                arr = _to_numpy(out[key], dtype=np.int64).ravel()
                out[key] = arr
            elif key == "forces":
                out[key] = _to_numpy(out[key], dtype=np.float64).ravel()
            else:  # mask
                out[key] = _to_numpy(out[key], dtype=np.float64)

    # scalar-ish fields: make sure they are plain Python/NumPy scalars
    for key in ("young", "young_min", "poisson", "g",
                "volfrac", "xmin", "xmax", "nelx", "nely",
                "penal", "filter_width", "rmin", "beta", "eta"):
        if key in out:
            v = out[key]
            if _torch is not None and _torch.is_tensor(v):
                v = v.detach().cpu().item()
            elif hasattr(v, "item") and not isinstance(v, (bytes, str)):
                try:
                    v = v.item()
                except Exception:
                    pass
            out[key] = v
    if "heavyside" in out:
        out["heavyside"] = bool(out["heavyside"])

    return out


def _coerce_discretization_args(problem) -> Dict[str, Union[float, bool]]:
    """Ensure discretization fields on *problem* are concrete floats/bools."""
    rmin = float(problem.rmin)
    nelx, nely = int(problem.width), int(problem.height)
    coerced = {}
    for name in ("filter_width", "rmin", "beta", "heavyside", "eta"):
        raw = getattr(problem, name)
        if isinstance(raw, str):
            coerced[name] = resolve_discretization_value(
                name, raw, rmin=rmin, nelx=nelx, nely=nely)
        elif name == "heavyside":
            coerced[name] = bool(raw)
        else:
            coerced[name] = float(raw)
    # Problem fields can be set directly, bypassing resolve_discretization_value.
    physics.check_filter_width(coerced["filter_width"], nelx=nelx, nely=nely)
    return coerced

def specified_task(problem):
    """Given a problem, return parameters for running topology optimization (NumPy)."""
    normals = _to_numpy(problem.normals, dtype=_np.float64)
    forces  = _to_numpy(problem.forces,  dtype=_np.float64)
    mask    = _to_numpy(problem.mask,    dtype=_np.float64) if not _np.isscalar(problem.mask) else problem.mask

    # fixed dofs from normals==1 (flatten the whole 3D array of node DOFs)
    fixdofs = _np.flatnonzero(normals.ravel())
    alldofs = _np.arange(2 * (problem.width + 1) * (problem.height + 1), dtype=_np.int64)
    freedofs = _np.sort(_np.setdiff1d(alldofs, fixdofs, assume_unique=False))

    disc = _coerce_discretization_args(problem)
    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "volfrac": float(problem.density),
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": int(problem.width),
        "nely": int(problem.height),
        "mask": mask,
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces.ravel(),
        "penal": 3.0,
        **disc,
    }
    return params

class Environment:
  """Backend wrapper for structural physics.

  Uses the legacy autograd-based physics backend for topology optimization.

  `__init__` normalizes `args` into its own dict, and every physics entry point
  reads `env.args`. Mutating the dict that was handed in -- including
  `model.args`, which `Model` keeps as a separate reference -- therefore has no
  effect on this environment. To change a discretization parameter on a
  constructed model (to switch the Heaviside projection on, say), set it on
  `model.structural_params` and call `model._refresh_physics_environment()`, or
  write to `model.env.args` directly.
  """

  def __init__(self, args: Dict[str, Any]):
    # normalize args to NumPy once
    self.args = _args_to_numpy(args)

    # Common shape helpers
    self.nelx: int = int(self.args["nelx"])
    self.nely: int = int(self.args["nely"])
    self._num_elems: int = self.nelx * self.nely

    # Normalize mask as 2D array of shape (nely, nelx)
    _mask = self.args.get("mask", 1.0)
    m = _to_numpy(_mask, dtype=np.float64)
    if m.size == 1:
      self.mask2d = np.ones((self.nely, self.nelx), dtype=np.float64) * float(m.item())
    else:
      # mask provided as array; ensure expected (nely, nelx)
      self.mask2d = m.reshape(self.nely, self.nelx)
    self._mask_flat = self.mask2d.ravel()

    # Backend selection: Legacy only
    self._backend = "legacy"
    self._init_legacy_backend()

  # ----------------------- Backend initializers -----------------------
  def _init_legacy_backend(self) -> None:
    # Legacy stiffness matrix (NumPy)
    self.ke = physics.get_stiffness_matrix(self.args["young"], self.args["poisson"])

  # ---------------------------- Utilities -----------------------------
  def reshape(self, params):
    p = _to_numpy(params)  # ensure NumPy
    return p.reshape(self.nely, self.nelx)

  # ----------------------------- Renders ------------------------------
  def render(self, params, volume_constraint=True):
    """Return the physical density the objective uses, not a pre-filter view.

    `cone_filter=True` is the stiffness field. The CNN-to-pixel handoff still
    asks `physical_density(..., cone_filter=False)` itself; that is a view of
    the same volume offset, not a second design, and it is not what we save.
    """
    x2d = self.reshape(params)
    return physics.physical_density(
        x2d, self.args, volume_constraint=volume_constraint, cone_filter=True
    )

  # ----------------------------- Objective ---------------------------
  def objective(self, params, volume_constraint=False):
    # Legacy physics
    x2d = self.reshape(params)
    return physics.objective(
        x2d, self.ke, self.args, volume_constraint=volume_constraint, cone_filter=True
    )

  # ----------------------------- Constraint --------------------------
  def constraint(self, params):
    # Legacy physics
    x2d = self.reshape(params)
    vol = physics.mean_density(x2d, self.args)
    return vol - self.args["volfrac"]