from dataclasses import dataclass

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)

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
    # Custom optimizer options for this parameter.
    custom_options: Dict[str, Any]
    # Optional callback that is invoked after each optimizer update step.
    # Args: The parameter object (after the update).
    update_callback: Optional[Callable[[Any], None]]

    def __init__(
        self,
        key: KeyT,
        param: Any,
        *,
        grad: Optional[Any] = None,
        custom_options: Optional[Dict[str, Any]] = None,
        update_callback: Optional[Callable[[Any], None]] = None,
    ):
        self.key = key
        self.parameter = param
        self.gradient = grad
        self.custom_options = dict() if custom_options is None else custom_options
        self.update_callback = update_callback
