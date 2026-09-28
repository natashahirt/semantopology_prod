"""Finite-element solve, autograd primitives, and the physics API."""

import importlib

# Import order matters. problem.problems imports this package's autograd/physics
# submodules while it is still initializing, so this file must not import api
# (api imports resolve_discretization_value from problems).
from . import caching
from . import autograd
from . import physics
from .physics import logit

__all__ = [
    'api',
    'autograd',
    'caching',
    'physics',
    'Environment',
    'specified_task',
    'logit',
]


def __getattr__(name):
    if name in ('api', 'Environment', 'specified_task'):
        module = importlib.import_module('.api', __name__)
        globals()['api'] = module
        globals()['Environment'] = module.Environment
        globals()['specified_task'] = module.specified_task
        return globals()[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
