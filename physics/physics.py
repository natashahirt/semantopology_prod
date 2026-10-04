"""Physics computation for structural optimization.

This module contains the core physics computations for topology optimization,
including finite element analysis, compliance calculation, and structural analysis.
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

"""
overview:
- core topopt physics using autograd
- fea for structural compliance minimization
- stiffness matrices, displacement calculation, compliance computation
- filtering operations (cone filter, sigmoid constraints)
"""

"""Autograd implementation of topology optimization for compliance minimization.

Exactly reproduces the result of "Efficient topology optimization in MATLAB
using 88 lines of code":
http://www.topopt.mek.dtu.dk/Apps-and-software/Efficient-topology-optimization-in-MATLAB
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=superfluous-parens

import warnings

import autograd
import autograd.numpy as np
from physics import autograd as topo_autograd
from physics import caching
import numpy as _np

# A note on conventions:
# - forces and freedofs are stored flattened, but logically represent arrays of
#   shape (Y+1, X+1, 2)
# - mask is either a scalar (1) or an array of shape (X, Y).
# Yes, this is confusing. Sorry!

# Logits are clipped here before the sigmoid, so beyond this magnitude every
# element saturates to exactly 0 or exactly 1. sigmoid_with_constrained_mean
# leans on that to bracket its root.
SIGMOID_CLIP = 40.0

# Below this sharpness the Heaviside projection is the identity to within
# floating point, and its normalizing denominator underflows.
MIN_PROJECTION_BETA = 1e-8

# A cone filter of this radius or smaller admits only the (0, 0) offset, which
# normalized_cone_filter_matrix then divides straight back out, so the filter is
# exactly the identity and all checkerboard control is silently lost.
MIN_FILTER_WIDTH = 1.0

# Smallest radius a derived (analysis-grid) width is clamped up to. Anything in
# (1.0, sqrt(2)] admits the same 5-point stencil, but the neighbour weight is
# `radius - 1`, so a radius just above 1.0 is still the identity to within
# floating point. At 1.5 the full 3x3 stencil is in range and the neighbours
# carry real weight (centre 1.5 against 0.5 per edge neighbour).
MIN_EFFECTIVE_FILTER_WIDTH = 1.5


def max_filter_width(nelx, nely):
  """Return the largest cone-filter radius that still resolves local structure.

  Once the radius reaches the grid diagonal every element's cone covers the
  whole domain, so the row-normalized filter is a global weighted average: the
  design is smeared away exactly as thoroughly as `MIN_FILTER_WIDTH` leaves it
  untouched, and `_cone_filter_matrix` pays O(radius**2) to do it.
  """
  return float(_np.hypot(float(nelx), float(nely)))


def check_filter_width(filter_width, *, nelx=None, nely=None):
  """Return `filter_width` as a float, rejecting unusable cone-filter radii.

  Both ends of the range destroy the design silently, so both raise:

  * `_cone_filter_matrix` keeps the offsets satisfying ``dx**2 + dy**2 <
    radius**2``, so a radius of at most `MIN_FILTER_WIDTH` keeps only the
    element itself and row normalization divides that single weight back out.
    The resulting identity filter is indistinguishable from a working one at
    the call site.
  * A radius past the grid diagonal turns the filter into a global average
    (see `max_filter_width`), which is just as destructive and additionally
    costs O(radius**2) to assemble.

  Non-finite radii are rejected here too. They pass every one-sided comparison
  and only surface much later, as `cannot convert float NaN to integer` (or an
  `OverflowError`) from ``int(np.ceil(radius))`` inside `_cone_filter_matrix`.

  The upper bound is only applied when the grid is known; pass `nelx`/`nely`
  wherever they are available.
  """
  width = float(filter_width)
  if not _np.isfinite(width):
    raise ValueError(
        f'filter_width resolves to {width}, which is not a finite radius. '
        '`_cone_filter_matrix` rounds the radius to an integer offset bound, '
        'so this fails later and far from its cause. Pass a finite radius '
        'greater than 1.0; 2.0 is the standard choice.')
  if width <= MIN_FILTER_WIDTH:
    raise ValueError(
        f'filter_width resolves to {width}, which degenerates the cone filter '
        f'to the identity (a radius of {MIN_FILTER_WIDTH} or less reaches no '
        'neighbouring element), silently removing checkerboard control. Use a '
        'radius greater than 1.0; 2.0 is the standard choice. With '
        "filter_width='linear' the radius is 2 * rmin, so rmin must exceed 0.5."
    )
  if nelx is not None and nely is not None:
    limit = max_filter_width(nelx, nely)
    if width > limit:
      raise ValueError(
          f'filter_width resolves to {width}, which exceeds the '
          f'{int(nelx)}x{int(nely)} grid diagonal of {limit:.4g}. Every cone '
          'then covers the whole domain, so the filter is a global average '
          'that erases the design. Use a radius below the grid diagonal; 2.0 '
          "is the standard choice. With filter_width='linear' the radius is "
          '2 * rmin.')
  return width


def clamp_filter_width(filter_width, *, nelx=None, nely=None):
  """Clamp a DERIVED filter radius into the range `check_filter_width` accepts.

  Analysis grids are produced by dividing the design grid (and its radius) by
  an integer factor, so their radius is a quotient nobody authored and the
  division alone can push it below `MIN_FILTER_WIDTH`. Raising there would
  abort a run mid-flight -- the factor is set after a stage has finished
  training -- so an unsatisfiable derived radius is clamped with a warning
  instead. User-authored configuration still goes through `check_filter_width`,
  where failing fast is the right answer.

  Non-finite radii are still rejected: division never produces them, so they
  can only come from a bad parameter that clamping would hide.
  """
  width = float(filter_width)
  if not _np.isfinite(width):
    raise ValueError(
        f'filter_width resolves to {width} on the analysis grid, which is not '
        'a finite radius. Downsampling cannot produce this, so it comes from a '
        'non-finite rmin or filter_width upstream.')
  clamped = max(width, MIN_EFFECTIVE_FILTER_WIDTH)
  if nelx is not None and nely is not None:
    clamped = min(clamped, max_filter_width(nelx, nely))
  if clamped != width:
    warnings.warn(
        f'Analysis-grid filter_width of {width} is outside the usable range; '
        f'clamping to {clamped}. The analysis grid is a downsampling of the '
        'design grid, so its radius is the design radius divided by the '
        'analysis factor; raise rmin (or filter_width) if you want the '
        'analysis grid to filter as widely as the design grid does.',
        stacklevel=2)
  return clamped


def default_args():
  # select the degrees of freedom
  nely = 25
  nelx = 80

  left_wall = list(range(0, 2*(nely+1), 2))
  right_corner = [2*(nelx+1)*(nely+1)-1]
  fixdofs = np.asarray(left_wall + right_corner)
  alldofs = np.arange(2*(nely+1)*(nelx+1))
  freedofs = np.asarray(list(set(alldofs) - set(fixdofs)))

  forces = np.zeros(2*(nely+1)*(nelx+1))
  forces[1] = -1.0

  return {'young': 1,     # material properties
          'young_min': 1e-9,
          'poisson': 0.3,
          'g': 0,  # force of gravity
          'volfrac': 0.4,  # constraints
          'nelx': nelx,     # input parameters
          'nely': nely,
          'freedofs': freedofs,
          'fixdofs': fixdofs,
          'forces': forces,
          'mask': 1,
          'penal': 3.0,
          # filter_width is the radius physics actually uses; rmin is carried
          # alongside it and must stay consistent with the 2 * rmin relation
          # that filter_width='linear' resolves through.
          'rmin': 1.0,
          'opt_steps': 50,
          'filter_width': 2.0,
          'step_size': 0.5,
          'name': 'truss'}


def projection_params(args):
  """Return (beta, eta) if Heaviside projection is enabled, else None.

  With Heaviside enabled, `Environment.render` and the objective share the
  filtered physical density, which holds `volfrac`. `cone_filter=False` is the
  CNN-to-pixel handoff view of the same offset -- already projected, so the
  pixel objective projecting it again is a Stage 8 concern, not a render bug.
  See `CanonicalRenderVolumeTest`.

  Rejects the parameter values that make `heavyside_projection` meaningless:
  `eta` outside [0, 1] drives its normalizing denominator to zero (NaN-ing the
  whole density field), and a `beta` that is non-positive or non-finite is not
  a softer projection but an ill-posed one. Both are validated in
  `StructuralParams` as well; this is the backstop for args dicts assembled by
  hand.
  """
  if not args.get('heavyside', False):
    return None
  beta = float(args.get('beta', 2.0))
  eta = float(args.get('eta', 0.5))
  if not _np.isfinite(beta):
    raise ValueError(
        f'beta must be finite, got {beta}. Every comparison against a NaN beta '
        'is False, so it reaches heavyside_projection and NaNs the whole '
        'density field; the first symptom is a Cholmod failure in the FEA '
        'solve, which reads as a physics problem rather than a bad parameter.')
  if beta <= 0.0:
    raise ValueError(
        f'beta must be positive, got {beta}. The Heaviside projection is even '
        'in beta, so a negative value is not a weaker projection; disable the '
        'projection with heavyside=False instead.')
  if not 0.0 <= eta <= 1.0:
    raise ValueError(
        f'eta must lie in [0, 1], got {eta}. Outside that range the projection '
        'denominator tanh(beta * eta) + tanh(beta * (1 - eta)) cancels to zero '
        'and the density field becomes NaN.')
  if beta <= MIN_PROJECTION_BETA:
    return None  # the projection degenerates to the identity as beta -> 0
  return beta, eta


def physical_density(x, args, volume_constraint=False, cone_filter=True):
  """Map raw design variables onto physical densities.

  Runs the standard SIMP chain: sigmoid, mask, density filter, optional
  Heaviside projection, then volume enforcement. The filter radius is
  args['filter_width'] and nothing else. args['rmin'] only reaches it when the
  caller asked for it: `resolve_discretization_value` turns
  ``filter_width='linear'`` into ``2 * rmin``, but a `StructuralParams` that
  sets `rmin` without `filter_width` leaves the radius at the `Problem`
  default, and `apply_discretization_params` warns when it sees that.

  Neither stage after the sigmoid preserves the mean -- the cone filter is
  row-normalized rather than sum-preserving, and the projection deliberately
  pushes densities apart -- so the volume constraint only holds on the final
  density when it is enforced last. Enforcing it last perturbs the legacy
  result, so it switches on exactly when it is needed (whenever projection is
  enabled) and can be forced either way with args['enforce_volume_last'].

  Heaviside projection (args['heavyside']) is opt-in. With it enabled, the
  volume constraint holds on this function's `cone_filter=True` result, which
  is also what `Environment.render` returns. `cone_filter=False` is the
  pre-filter view of the SAME volume offset -- the CNN-to-pixel handoff
  (`train.utils.constrained_logits`) -- not a second design and not the saved
  image. See `projection_params` and `CanonicalRenderVolumeTest`.

  Note that render and the objective agree only for a given `args`:
  `Model.get_structural_loss` evaluates the objective on `analysis_env`
  whenever `analysis_factor != 1`, while `train/base.py` renders through
  `model.env`, so above the analysis-dimension cap they use a different grid
  and a different radius. That is a resolution split, not a third density.
  """
  shape = (args['nely'], args['nelx'])
  assert x.shape == shape or x.ndim == 1
  x = x.reshape(shape)
  projection = projection_params(args)
  # Constrain the mean of the finished density, matching mean_density().
  volume_last = volume_constraint and args.get(
      'enforce_volume_last', projection is not None)
  if cone_filter or volume_last:
    check_filter_width(
        args['filter_width'], nelx=args['nelx'], nely=args['nely'])

  def filter_and_project(x_full, apply_filter):
    if apply_filter:
      x_full = topo_autograd.cone_filter(
          x_full, args['filter_width'], args['mask'])
    if projection is not None:
      x_full = heavyside_projection(x_full, *projection)
    return x_full

  if not volume_constraint:
    return filter_and_project(x * args['mask'], cone_filter)

  mask = np.broadcast_to(args['mask'], x.shape) > 0
  design_indices = np.flatnonzero(mask)

  def build(x_designed, apply_filter):
    x_flat = topo_autograd.scatter1d(x_designed, design_indices, x.size)
    return filter_and_project(x_flat.reshape(shape), apply_filter)

  if volume_last:
    design_fraction = np.mean(args['mask'])
    measure = lambda x_designed: np.mean(build(x_designed, True)) / design_fraction
  else:
    measure = None
  return build(
      sigmoid_with_constrained_mean(x[mask], args['volfrac'], measure),
      cone_filter)


def mean_density(x, args, volume_constraint=False, cone_filter=True):
  return (np.mean(physical_density(x, args, volume_constraint, cone_filter))
          / np.mean(args['mask']))


def get_stiffness_matrix(young, poisson):
  # Element stiffness matrix
  e, nu = young, poisson
  k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
  return e/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
                              ])


@caching.ndarray_safe_lru_cache(8)
def get_k_indices(nely, nelx):
  # Precompute indices for sparse matrix assembly.
  ely, elx = np.meshgrid(range(nely), range(nelx))  # x, y coords
  ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

  n1 = (nely+1)*(elx+0) + (ely+0)
  n2 = (nely+1)*(elx+1) + (ely+0)
  n3 = (nely+1)*(elx+1) + (ely+1)
  n4 = (nely+1)*(elx+0) + (ely+1)
  edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
  edof = edof.T[0]

  x_list = np.repeat(edof, 8)  # rows (repeat for each column)
  y_list = np.tile(edof, 8).flatten()  # columns (tile for each row)
  return x_list, y_list


@caching.ndarray_safe_lru_cache(1)
def _get_dof_indices(nely, nelx, freedofs, fixdofs):
  k_xlist, k_ylist = get_k_indices(nely, nelx)
  index_map = topo_autograd.inverse_permutation(
      np.concatenate([freedofs, fixdofs]))
  keep = np.isin(k_xlist, freedofs) & np.isin(k_ylist, freedofs)
  i = index_map[k_xlist][keep]
  j = index_map[k_ylist][keep]
  return index_map, keep, np.stack([i, j])


def displace(x_phys, ke, forces, freedofs, fixdofs, *,
             penal=3, e_min=1e-9, e_0=1):
  # Displaces the load x using finite element techniques. The spsolve here
  # occupies the majority of this entire simulation's runtime.
  stiffness = young_modulus(x_phys, e_0, e_min, p=penal)
  nely, nelx = stiffness.shape
  
  # Get precomputed indices
  k_entries = (stiffness.T.reshape(-1, 1, 1) * ke).flatten()

  index_map, keep, indices = _get_dof_indices(
      nely, nelx, freedofs, fixdofs
  )
  u_nonzero = topo_autograd.solve_coo(k_entries[keep], indices, forces[freedofs],
                                     sym_pos=True)
  u_values = np.concatenate([u_nonzero, np.zeros(len(fixdofs))])

  return u_values[index_map]


def get_k(stiffness, ke):
  # Constructs a sparse stiffness matrix, k, for use in the displace function.
  nely, nelx = stiffness.shape
  x_list, y_list = get_k_indices(nely, nelx)

  # make the stiffness matrix entries
  kd = stiffness.T.reshape(nelx*nely, 1, 1)
  value_list = (kd * ke).flatten()
  return value_list, y_list, x_list


def young_modulus(x, e_0, e_min, p=3):
  return e_min + x ** p * (e_0 - e_min)


def compliance(x_phys, u, ke, *, penal=3, e_min=1e-9, e_0=1):
  # Calculates the compliance
  # Read about how this was vectorized here:
  # https://colab.research.google.com/drive/1PE-otq5hAMMi_q9dC6DkRvf2xzVhWVQ4

  # index map
  nely, nelx = x_phys.shape
  ely, elx = np.meshgrid(range(nely), range(nelx))  # x, y coords

  # nodes
  n1 = (nely+1)*(elx+0) + (ely+0)
  n2 = (nely+1)*(elx+1) + (ely+0)
  n3 = (nely+1)*(elx+1) + (ely+1)
  n4 = (nely+1)*(elx+0) + (ely+1)
  all_ixs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])

  # select from u matrix
  u_selected = u[all_ixs]

  # compute x^penal * U.T @ ke @ U in a vectorized way
  ke_u = np.einsum('ij,jkl->ikl', ke, u_selected)
  ce = np.einsum('ijk,ijk->jk', u_selected, ke_u)
  C = young_modulus(x_phys, e_0, e_min, p=penal) * ce.T
  return np.sum(C)


def optimality_criteria_combine(x, dc, dv, args, max_move=0.2, eta=0.5):
  """Fully differentiable version of the optimality criteria."""

  volfrac = args['volfrac']

  def pack(x, dc, dv):
    return np.concatenate([x.ravel(), dc.ravel(), dv.ravel()])

  def unpack(inputs):
    x_flat, dc_flat, dv_flat = np.split(inputs, [x.size, x.size + dc.size])
    return (x_flat.reshape(x.shape),
            dc_flat.reshape(dc.shape),
            dv_flat.reshape(dv.shape))

  def compute_xnew(inputs, lambda_):
    x, dc, dv = unpack(inputs)
    # avoid dividing by zero outside the design region
    dv = np.where(np.ravel(args['mask']) > 0, dv, 1)
    # square root is not defined for negative numbers, which can happen due to
    # small numerical errors in the computed gradients.
    xnew = x * np.maximum(-dc / (lambda_ * dv), 0) ** eta
    lower = np.maximum(0.0, x - max_move)
    upper = np.minimum(1.0, x + max_move)
    # note: autograd does not define gradients for np.clip
    return np.minimum(np.maximum(xnew, lower), upper)

  def f(inputs, lambda_):
    xnew = compute_xnew(inputs, lambda_)
    return volfrac - mean_density(xnew, args)

  # find_root allows us to differentiate through the while loop.
  inputs = pack(x, dc, dv)
  lambda_ = topo_autograd.find_root(f, inputs, lower_bound=1e-9, upper_bound=1e9)
  return compute_xnew(inputs, lambda_)


def sigmoid(x):
  # Stable logistic sigmoid; differentiable under autograd
  x = np.clip(x, -SIGMOID_CLIP, SIGMOID_CLIP)
  return 0.5*np.tanh(0.5*x) + 0.5


def logit(p, eps=1e-12):
  p = np.clip(p, eps, 1.0 - eps)
  return np.log(p) - np.log1p(-p)


def heavyside_projection(x, beta, eta=0.5):
  """Smoothed Heaviside projection (Wang, Lazarov & Sigmund, 2011).

  Pushes densities toward 0/1 with sharpness `beta` about the threshold `eta`,
  holding the endpoints fixed: 0 maps to 0 and 1 maps to 1. Strictly increasing
  in `x`, which the volume-enforcement bisection relies on, but not mean
  preserving.
  """
  offset = np.tanh(beta * eta)
  return ((offset + np.tanh(beta * (x - eta)))
          / (offset + np.tanh(beta * (1.0 - eta))))


# an alternative to the optimality criteria
def sigmoid_with_constrained_mean(x, average, measure=None):
  """Return ``sigmoid(x + b)`` for the offset `b` that hits a target mean.

  By default `b` solves ``mean(sigmoid(x + b)) == average``. Passing `measure`
  constrains that downstream quantity instead, so the volume constraint can be
  imposed on the filtered and projected density rather than on the raw sigmoid.
  Either residual is increasing in `b`, as `find_root` requires.
  """
  if measure is None:
    def f(x_, y):
      return sigmoid(x_ + y).mean() - average
    # Tight bracket, valid only for the plain-sigmoid residual: shifting every
    # element to at most (at least) logit(average) drives the mean below (above)
    # the target.
    lower_bound = logit(average) - np.max(x)
    upper_bound = logit(average) - np.min(x)
    check_bracket = False
  else:
    def f(x_, y):
      return measure(sigmoid(x_ + y)) - average
    # `measure` composes stages whose effect on the mean is unknown here, so
    # bracket on sigmoid saturation instead: past +/-SIGMOID_CLIP every element
    # is exactly 0 or exactly 1, and any increasing measure that maps those two
    # fields to 0 and to at least 1 then straddles any average in [0, 1].
    lower_bound = -SIGMOID_CLIP - np.max(x)
    upper_bound = SIGMOID_CLIP - np.min(x)
    check_bracket = True
  b = topo_autograd.find_root(
      f, x, lower_bound, upper_bound, check_bracket=check_bracket)
  return sigmoid(x + b)


def pixel_gravity(forces, nelx, nely, fraction):
  """Downward force contributed by one fully solid element.

  ``fraction`` is the share of the live-load resultant that a fully solid
  design carries as self-weight, split equally across pixels. Void adds
  nothing because ``calculate_forces`` weights the body force by density.
  ``0`` is off.
  """
  fraction = float(fraction)
  if fraction < 0.0:
    raise ValueError(f'gravity_load must be >= 0, got {fraction}')
  if fraction == 0.0:
    return 0.0
  n_elem = int(nelx) * int(nely)
  if n_elem <= 0:
    return 0.0
  total = float(_np.sum(_np.abs(_np.asarray(forces, dtype=_np.float64))))
  return fraction * total / n_elem


def apply_pixel_gravity(args, fraction):
  """Write ``gravity_load`` and the matching per-element ``g`` onto ``args``."""
  fraction = float(fraction or 0.0)
  args['gravity_load'] = fraction
  args['g'] = pixel_gravity(
      args['forces'], args['nelx'], args['nely'], fraction)
  return args


def calculate_forces(x_phys, args):
  applied_force = args['forces']

  if not args.get('g'):
    return applied_force

  density = 0
  for pad_left in [0, 1]:
    for pad_up in [0, 1]:
      padding = [(pad_left, 1 - pad_left), (pad_up, 1 - pad_up)]
      density += (1/4) * np.pad(
          x_phys.T, padding, mode='constant', constant_values=0
      )
  gravitional_force = -args['g'] * density[..., np.newaxis] * np.array([0, 1])
  return applied_force + gravitional_force.ravel()


def objective(x, ke, args, volume_constraint=False, cone_filter=True):
  """Objective function (compliance) for topology optimization."""
  kwargs = dict(penal=args['penal'], e_min=args['young_min'], e_0=args['young'])
  x_phys = physical_density(x, args, volume_constraint=volume_constraint,
                            cone_filter=cone_filter)
  forces = calculate_forces(x_phys, args)
  u = displace(
      x_phys, ke, forces, args['freedofs'], args['fixdofs'], **kwargs)
  c = compliance(x_phys, u, ke, **kwargs)
  return c


def optimality_criteria_step(x, ke, args):
  """Heuristic topology optimization, as described in the 88 lines paper."""
  c, dc = autograd.value_and_grad(objective)(x, ke, args)
  dv = autograd.grad(mean_density)(x, args)
  x = optimality_criteria_combine(x, dc, dv, args)
  return c, x


def run_toposim(x=None, args=None, loss_only=True, verbose=True):
  # Root function that runs the full optimization routine
  if args is None:
    args = default_args()
  if x is None:
    x = np.ones((args['nely'], args['nelx'])) * args['volfrac']  # init mass

  if not loss_only:
    frames = [x.copy()]
  ke = get_stiffness_matrix(args['young'], args['poisson'])  # stiffness matrix

  losses = []
  for step in range(args['opt_steps']):
    c, x = optimality_criteria_step(x, ke, args)
    losses.append(c)

    if not loss_only and verbose and step % 5 == 0:
      print('step {}, loss {:.2e}'.format(step, c))

    if not loss_only:
      frames.append(x.copy())

  return losses[-1] if loss_only else (losses, x, frames)
