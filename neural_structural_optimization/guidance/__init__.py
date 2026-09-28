"""Compliance, CLIP, grad-match blending, coadaptive mask, and sketch prior."""

import importlib

# model_base imports this package's loss_* submodules while it is still
# initializing. blend imports VeniceLossTerms from model_base, so it must
# stay off the eager path.
from .loss_structural import PhysicalDensity, StructuralLoss
from .loss_sketch import (
    apply_sketch_config,
    init_weight_with_occupancy,
    load_site_mask,
    load_sketch_occupancy,
    sketch_mass_prior_loss,
    sketch_motif_descriptor,
    sketch_motif_loss,
    sketch_patch_vocabulary_loss,
)
from .loss_semantic_prior import (
    CLIPSemanticProvider,
    CallableScoreProvider,
    CoadaptiveMask,
    DiffusionSDSProvider,
    FrozenDenoiser,
    SemanticSpatialPrior,
    physical_motif_scale_fracs,
)

__all__ = [
    'StructuralLoss',
    'PhysicalDensity',
    'CLIPLoss',
    'apply_sketch_config',
    'init_weight_with_occupancy',
    'load_site_mask',
    'load_sketch_occupancy',
    'sketch_mass_prior_loss',
    'sketch_motif_descriptor',
    'sketch_motif_loss',
    'sketch_patch_vocabulary_loss',
    'CLIPSemanticProvider',
    'CallableScoreProvider',
    'CoadaptiveMask',
    'DiffusionSDSProvider',
    'SemanticSpatialPrior',
    'FrozenDenoiser',
    'physical_motif_scale_fracs',
    'BlendMode',
    'GradNormEma',
    'GRAD_MATCH_EMA_DECAY',
    'GRAD_MATCH_WEIGHT_MAX',
    'freeze_blend_mode',
    'resolve_blend_mode',
    'snapshot_blend',
    'unweighted_grad_norms',
]

_BLEND_NAMES = {
    'BlendMode': 'BlendMode',
    'GradNormEma': 'GradNormEma',
    'GRAD_MATCH_EMA_DECAY': 'GRAD_MATCH_EMA_DECAY',
    'GRAD_MATCH_WEIGHT_MAX': 'GRAD_MATCH_WEIGHT_MAX',
    'freeze_blend_mode': 'freeze_blend_mode',
    'resolve_blend_mode': 'resolve_blend_mode',
    'snapshot_blend': 'snapshot_blend',
    'unweighted_grad_norms': 'unweighted_grad_norms',
}


def __getattr__(name):
    if name == 'CLIPLoss':
        module = importlib.import_module('.loss_clip', __name__)
        globals()['CLIPLoss'] = module.CLIPLoss
        return module.CLIPLoss
    if name == 'blend' or name in _BLEND_NAMES:
        module = importlib.import_module('.blend', __name__)
        globals()['blend'] = module
        for alias, attr in _BLEND_NAMES.items():
            globals()[alias] = getattr(module, attr)
        return globals()[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
