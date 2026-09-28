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
overview
- helper functions for image generation and visualization
- design rendering and problem-specific display logic
- dynamic depth calculation for CNN architectures
- converts optimization results to visual formats
"""

import math
from typing import Any, Dict

import matplotlib.cm
import matplotlib.colors
from neural_structural_optimization.problem import problems
import numpy as np
from PIL import Image
import xarray
from typing import Tuple, List

def image_from_array(
    data: np.ndarray, cmap: str = 'Greys', vmin: float = 0, vmax: float = 1,
) -> Image.Image:
  """Convert a NumPy array into a Pillow Image using a colormap."""
  norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
  frame = np.ma.masked_invalid(data)
  image = Image.fromarray(mappable.to_rgba(frame, bytes=True), mode='RGBA')
  return image


def image_from_design(
    design: xarray.DataArray, problem: problems.Problem,
) -> Image.Image:
  """Convert a design and problem into a Pillow Image."""
  assert design.dims == ('y', 'x'), design.dims
  imaged_designs = []
  if problem.mirror_left:
    imaged_designs.append(design.isel(x=slice(None, None, -1)))
  imaged_designs.append(design)
  if problem.mirror_right:
    imaged_designs.append(design.isel(x=slice(None, None, -1)))
  return image_from_array(xarray.concat(imaged_designs, dim='x').data)


def image_from_design_array(
    design: xarray.DataArray, problem: problems.Problem,
) -> np.ndarray:
  """Convert a design and problem into a numpy array (optimized version)."""
  assert design.dims == ('y', 'x'), design.dims
  imaged_designs = []
  if problem.mirror_left:
    imaged_designs.append(design.isel(x=slice(None, None, -1)))
  imaged_designs.append(design)
  if problem.mirror_right:
    imaged_designs.append(design.isel(x=slice(None, None, -1)))
  return xarray.concat(imaged_designs, dim='x').data

def dynamic_depth_kwargs(params, max_upsamples=float('inf'), kernel_size=(5,5), padding=1):
  """Calculate dynamic depth parameters for CNN architectures."""
  base_h = params.height
  base_w = params.width
  kh, kw = kernel_size
  conv_upsample = []
  max_divisions = min(
        int(np.log2(base_h // (kh + padding))),
        int(np.log2(base_w // (kw + padding))),
        max_upsamples
    )
  # Apply the divisions
  for _ in range(max_divisions):
      base_h //= 2
      base_w //= 2
      conv_upsample.append(2)
  sum_upsamples = np.prod(conv_upsample)
  params = params.copy(width=int(base_w * sum_upsamples), height=int(base_h * sum_upsamples))
  conv_upsample = [1] + conv_upsample + [1]
  conv_filters = [512, 256, 128, 64, 32, 16, 8, 1][-len(conv_upsample):] # experiment with this, also got good results with 16 final conv layer
  assert len(conv_filters) == len(conv_upsample)
  return params, dict(
      conv_upsample=conv_upsample,
      conv_filters=conv_filters,
      dense_channels=conv_filters[0] // 2,
  )
