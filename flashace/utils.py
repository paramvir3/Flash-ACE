from contextlib import contextmanager
from typing import Iterator

import torch.nn as nn


@contextmanager
def frozen_parameter_grads(module: nn.Module) -> Iterator[None]:
    """Temporarily disable gradients on ``module`` parameters.

    This is useful for evaluation phases where we still need autograd with
    respect to inputs (e.g., positions for forces) but want to avoid keeping
    parameter gradients and their associated storage alive. All original
    ``requires_grad`` flags are restored after the context exits.
    """

    requires_grad_state = [p.requires_grad for p in module.parameters()]
    try:
        for param in module.parameters():
            param.requires_grad_(False)
        yield
    finally:
        for param, prev_state in zip(module.parameters(), requires_grad_state):
            param.requires_grad_(prev_state)
