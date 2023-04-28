from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)

from ..util import is_torch_array, is_xp_array

from .types import KeyT


@dataclass
class OptimizerParamInfo:
    """Data pertaining to a parameter that is tracked and updated by the optimizer."""

    # Persistent, unique ID.
    key: KeyT
    # Persistent array-like object that contains the weights to be optimized.
    # Modified in-place during an optimizer update step.
    parameter: Any
    # Persistent array-like object that contains the gradients pertaining to the weights.
    # Can be `None` for parameters that lazily initialize their gradients.
    # Modified in-place during an optimizer update step.
    gradient: Optional[Any]
    # Optional callback that is invoked after each optimizer update step.
    # Args: The parameter object (after the update).
    update_callback: Optional[Callable[[Any], None]]

    def __init__(
        self,
        key: KeyT,
        param: Any,
        *,
        grad: Optional[Any] = None,
        update_callback: Optional[Callable[[Any], None]] = None,
    ):
        self.key = key
        self.parameter = param
        self.gradient = grad
        self.update_callback = update_callback

    @property
    def from_torch(self) -> bool:
        if self.gradient is None:
            return is_torch_array(self.parameter)
        else:
            return is_torch_array(self.parameter) and is_torch_array(self.gradient)

    @property
    def from_thinc(self) -> bool:
        if self.gradient is None:
            return is_xp_array(self.parameter)
        else:
            return is_xp_array(self.parameter) and is_xp_array(self.gradient)

    @property
    def from_xp(self) -> bool:
        return self.from_thinc
