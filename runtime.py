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

# Keep __init__ lightweight to avoid circular imports; import submodules explicitly where needed.
__version__ = "0.1.0"

# Pin OpenMP to one thread BEFORE anything can load a native library.
#
# CHOLMOD's supernodal factorization races against the other OpenMP runtime in
# this process (libcholmod links libomp and Apple's Accelerate; numpy ships its
# own bundled OpenBLAS) and takes the interpreter down with a bare SIGSEGV -- no
# exception, no traceback, the run simply disappears. It is intermittent, which
# is worse than deterministic: multistory_building at 96x192 crashed 3 runs in 5
# and at 128x256 crashed 4 in 5, so a green run proves nothing.
#
# Serializing OpenMP is not a slow safe fallback here -- single-threaded
# supernodal is FASTER than the racy threaded version (137.5ms vs 171.9ms at
# 96x192) because the contention was pure overhead, and it is ~9x faster than
# the SuperLU fallback at 128x256. Torch is unaffected: it reads this at import
# but `configure_torch_threads()` below restores its thread pool afterwards.
#
# This must run before numpy/scipy/sksparse are imported, so it lives at the top
# of the package __init__ rather than next to the solver that needs it. An
# externally set OMP_NUM_THREADS is respected.
import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "1")

# Import torch NOW, after the OpenMP pin and before any CHOLMOD solve. A late
# torch import (CLIPLoss, physics.api) brings a second OpenMP runtime into
# a process that has already factored, and that race is a bare SIGSEGV.
# CLIP stays lazy: `--print-config` must not download CLIP weights.
try:
    import torch as _torch  # noqa: F401
except Exception:
    _torch = None


def configure_torch_threads(num_threads=None):
  """Restore torch's CPU thread pool after the OpenMP pin above.

  `OMP_NUM_THREADS=1` would otherwise leave torch single-threaded and roughly
  1.8x slower on CPU. Calling this is a net win rather than a repair: setting
  the count explicitly measured faster than torch's own default (44.0ms vs
  79.1ms on a 1500x1500 matmul).

  Args:
    num_threads: threads to allow torch. Defaults to the machine's CPU count.

  Returns:
    The thread count torch actually adopted, or None if torch is unavailable.
  """
  try:
    import torch
  except ImportError:
    return None
  torch.set_num_threads(num_threads or _os.cpu_count() or 1)
  return torch.get_num_threads()

# CLIP is optional and expensive to import (torch + clip + kornia). Do not
# load it at package import: `--print-config` and other config-only paths
# must inspect a run without constructing CLIPLoss or pulling those modules.
# `from runtime import CLIP_AVAILABLE` still works via
# module __getattr__; the first access pays the import, later ones are cached.


def __getattr__(name):
  if name == 'CLIP_AVAILABLE':
    try:
      from guidance.loss_clip import CLIPLoss  # type: ignore  # noqa: F401
      value = True
    except Exception:
      value = False
    globals()['CLIP_AVAILABLE'] = value
    return value
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
