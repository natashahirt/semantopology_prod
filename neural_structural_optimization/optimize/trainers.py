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

"""Progressive training strategies for neural structural optimization."""

from typing import Callable, List
import torch
import torch.nn.functional as F
import xarray

from neural_structural_optimization.model import PixelModel, CNNModel
from .utils import match_mean_std_in_logit_space


class ProgressiveTrainer:
    """Handles progressive training with upsampling."""
    
    def __init__(self, model, max_iterations: int, resize_num: int = 2, 
                 save_intermediate_designs: bool = True):
        self.model = model
        self.max_iterations = max_iterations
        self.resize_num = resize_num
        self.save_intermediate_designs = save_intermediate_designs
    
    def train(self, optimizer_class: Callable, **optimizer_kwargs) -> List[xarray.Dataset]:
        """Run progressive training."""
        ds_history = []
        
        for stage in range(self.resize_num):
            is_first = (stage == 0)
            is_last = (stage == self.resize_num - 1)
            
            if not (is_first or is_last):
                continue

            print(f"\nTraining stage {stage + 1}/{self.resize_num} at resolution: {self.model.shape[1]}x{self.model.shape[2]}")
            
            optimizer = optimizer_class(self.model, self.max_iterations, 
                                      save_intermediate_designs=self.save_intermediate_designs, 
                                      **optimizer_kwargs)
            ds = optimizer.optimize()
            ds_history.append(ds)
            
            if is_first and self.resize_num > 1:
                jump_scale = 2 ** (self.resize_num - 1)
                self._upsample_model(scale=jump_scale)
        
        return ds_history
    
    def _upsample_model(self, scale: int = 2):
        """Upsample the model for next stage."""
        if isinstance(self.model, PixelModel):
            self.model.upsample(scale=scale, max_dim=500)
        elif isinstance(self.model, CNNModel):
            self.model.upsample(scale=scale, freeze_transferred=True)

class PixelRefineTrainer(ProgressiveTrainer):
    """Handles progressive training with pixel refinement (CNN to PixelModel transition)."""
    
    def __init__(self, model, max_iterations: int, resize_num: int = 2, 
                 save_intermediate_designs: bool = True, switch_threshold: int = 200, coarse_start: bool = True,
                 initial_image: torch.Tensor = None):
        super().__init__(model, max_iterations, resize_num, save_intermediate_designs)
        self.switch_threshold = switch_threshold
        self.coarse_start = coarse_start
        self.initial_image = initial_image
    
    def _initialize_model_with_image(self, model):
        """Initialize model parameters with the provided image."""
        if self.initial_image is not None:
            print(f"Initializing model with provided image of shape: {self.initial_image.shape}")
            
            # Ensure image is in the right format and size
            if self.initial_image.dim() == 3:
                # Add batch dimension if needed
                img = self.initial_image.unsqueeze(0)
            else:
                img = self.initial_image
                
            # Resize image to match model's expected input size
            if img.shape[-2:] != (model.shape[1], model.shape[2]):
                img = F.interpolate(img, size=(model.shape[1], model.shape[2]), 
                                  mode='bilinear', align_corners=False)
            
            # Convert to logits (inverse sigmoid)
            with torch.no_grad():
                # Clamp to avoid log(0) or log(1)
                img_clamped = img.clamp(1e-6, 1.0 - 1e-6)
                logits = torch.logit(img_clamped)
                
                # Initialize model parameters
                if hasattr(model, 'z'):
                    # PixelModel
                    model.z.data.copy_(logits.squeeze(0))
                elif hasattr(model, 'parameters'):
                    # CNNModel - initialize first layer or use a custom initialization
                    # This is a simplified approach - you might need to adapt based on your model architecture
                    for param in model.parameters():
                        if param.dim() >= 2:  # Weight parameters
                            # Initialize with small random values around the image logits
                            param.data.normal_(0, 0.01)
                        else:  # Bias parameters
                            param.data.zero_()
                    
                    # Set the model's initial state to produce something close to the image
                    # This might require model-specific initialization logic
                    pass

    def _switch_to_pixel_model(self):
        """Switch current model from CNNModel to PixelModel at the same resolution."""
        if not isinstance(self.model, CNNModel):
            return
            
        print(f"\nSwitching to PixelModel at resolution: {self.model.shape[1]}x{self.model.shape[2]}")
        pixel_model = PixelModel(
            structural_params=self.model.structural_params,
            clip_loss=self.model.clip_loss,
            seed=self.model.seed
        )
        
        with torch.no_grad():
            cnn_logits = self.model.forward()
            pixel_model.z.data.copy_(cnn_logits)
            
            # Match statistics to ensure smooth transition
            ref_img = torch.sigmoid(cnn_logits)
            match_mean_std_in_logit_space(pixel_model.z, ref_img)
        
        self.model = pixel_model
    
    def train(self, optimizer_class: Callable, **optimizer_kwargs) -> List[xarray.Dataset]:
        """Run progressive training with pixel refinement."""
        ds_history = []
        
        # Initialize model with image if provided
        if self.initial_image is not None:
            self._initialize_model_with_image(self.model)
        
        for stage in range(self.resize_num):
            is_first = (stage == 0)
            is_last = (stage == self.resize_num - 1)
            
            if not (is_first or is_last):
                continue

            # Switch to PixelModel if CNN resolution exceeds threshold (early switch)
            if isinstance(self.model, CNNModel) and max(self.model.shape[1], self.model.shape[2]) > self.switch_threshold:
                self._switch_to_pixel_model()

            model = self.model
            if model.clip_loss is not None:
                if stage == 0:
                    model.clip_loss.use_patch_pyramid = True
                    model.clip_loss.global_downside = 100
                    model.clip_loss.use_pairwise_spread = False
                    model.clip_R = 7.0
                elif stage == self.resize_num - 1: # Use final stage logic
                    model.clip_loss.use_patch_pyramid = True
                    model.clip_loss.patch_fracs = (0.75, 0.5, 0.25)
                    model.clip_loss.crops_per_frac = (8, 16, 24)
                    model.clip_loss.min_patch_px = max(96, min(model.shape[1], model.shape[2]) // 4)
                    model.clip_loss.use_pairwise_spread = True
                    model.clip_R = 2.0

            print(f"\nTraining stage {stage + 1}/{self.resize_num} at resolution: {model.shape[1]}x{model.shape[2]}")
            print(f"Using coarse_start={self.coarse_start}")
            
            # Use the optimizer with the specified coarse_start setting
            optimizer_kwargs.setdefault('coarse_start', self.coarse_start)
            optimizer = optimizer_class(model, self.max_iterations, 
                                      save_intermediate_designs=self.save_intermediate_designs, 
                                      **optimizer_kwargs)
            ds = optimizer.optimize()
            ds_history.append(ds)
            
            # AFTER coarsest level converges, if it's CNN, switch to Pixel and re-run at SAME resolution
            if is_first and isinstance(self.model, CNNModel):
                print(f"\nRefining coarsest stage with PixelModel...")
                self._switch_to_pixel_model()
                model = self.model # update local ref
                
                optimizer = optimizer_class(model, self.max_iterations,
                                          save_intermediate_designs=self.save_intermediate_designs,
                                          **optimizer_kwargs)
                ds = optimizer.optimize()
                ds_history.append(ds)

            if is_first and self.resize_num > 1:
                jump_scale = 2 ** (self.resize_num - 1)
                self._upsample_model(scale=jump_scale)
        
        return ds_history
