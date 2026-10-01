"""Run recipes. Each recipe sequences the solver; it does not own a loss."""

from .preset import PAPER, DreamLayoutPreset, prompt_slug

__all__ = ['PAPER', 'DreamLayoutPreset', 'prompt_slug']
