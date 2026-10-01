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

"""Base classes for optimization framework."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import xarray
import torch

from absl import logging


class OptimizationTracker:
    """Tracks optimization progress and manages result collection."""
    
    def __init__(self, save_intermediate_designs: bool = True):
        self.losses: List[float] = []
        self.frames: List[np.ndarray] = []
        self.save_intermediate_designs = save_intermediate_designs
    
    def add_step(self, loss: float, frame: np.ndarray):
        """Add a step to the optimization history."""
        self.losses.append(loss)
        self.frames.append(frame)
    
    def create_dataset(self, model) -> xarray.Dataset:
        """Create xarray dataset from collected results."""
        best_design = np.nanargmin(self.losses)
        logging.info(f'Final loss: {self.losses[best_design]}')
        
        with torch.no_grad():
            designs = [model.env.render(torch.tensor(x), volume_constraint=True) for x in self.frames]
        
        if self.save_intermediate_designs:
            ds = xarray.Dataset({
                'loss': (('step',), self.losses),
                'design': (('step', 'y', 'x'), designs),
            }, coords={'step': np.arange(len(self.losses))})
        else:
            ds = xarray.Dataset({
                'loss': (('step',), self.losses),
                'design': (('y', 'x'), designs[best_design]),
            }, coords={'step': np.arange(len(self.losses))})
        return ds


class BaseOptimizer(ABC):
    """Base class for optimization algorithms."""
    
    def __init__(self, model, max_iterations: int, save_intermediate_designs: bool = True):
        self.model = model
        self.max_iterations = max_iterations
        self.tracker = OptimizationTracker(save_intermediate_designs)
    
    @abstractmethod
    def optimize(self) -> xarray.Dataset:
        """Run the optimization algorithm."""
        pass
    
    def _validate_model_type(self, expected_type, algorithm_name: str):
        """Validate that model is of expected type."""
        if not isinstance(self.model, expected_type):
            raise ValueError(f'{algorithm_name} only defined for {expected_type.__name__} models')
