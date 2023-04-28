from abc import ABC, abstractmethod

from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Tuple,
)

from thinc.types import FloatsXd

from .param import OptimizerParamInfo
from .types import KeyT


class Optimizer(ABC):
    @abstractmethod
    def initialize(
        self,
        initial_params: Iterable[OptimizerParamInfo],
    ) -> None:
        """Called once before the training loop starts to register already-initialized parameters,
        if any.
        """
        pass

    # XXX We need this method to support thinc (torch as well?) models that add/modify their
    # params after init (like the v2 transition parser's precomputible affine layer).
    @abstractmethod
    def register_param(
        self, param: OptimizerParamInfo, overwrite: bool = False
    ) -> None:
        """Register a parameter with the optimizer. If `overwrite` is `True`, any existing
        registration for the given key is replaced with the one passed to the method.
        """
        pass

    # TODO add a method to register multiple params as a single group (for performance)

    @abstractmethod
    def step(self) -> None:
        """Performs a single update step on all registered parameters and invokes their post-update
        callbacks."""
        pass

    # XXX Methods that follow are to ensure backward-compatibility with the current interface.
    # TODO TBD if to what extent we want to preserve this.

    @abstractmethod
    def __call__(
        self,
        key: KeyT,
        weights: FloatsXd,
        gradient: FloatsXd,
        *,
        lr_scale: float = 1.0,
    ) -> Tuple[FloatsXd, FloatsXd]:
        """Invoke the optimizer for a given pair of weights and their gradients. Returns the updated
        weights and the (zeroed) gradients.
        """
        pass

    @property
    @abstractmethod
    def last_score(self) -> Optional[Tuple[int, float]]:
        """Sets/gets the last score recorded during an optimization step."""
        pass

    @last_score.setter
    @abstractmethod
    def last_score(self, score: float) -> None:
        """Record the score for the current optimization step."""
        pass

    @property
    @abstractmethod
    def current_step(self) -> int:
        """Gets the current step of the optimizer."""
        pass

    @property
    @abstractmethod
    def averages(self) -> Optional[Dict[KeyT, FloatsXd]]:
        """Gets the running average of the weights."""
        pass

    @abstractmethod
    def learn_rate(self, step: int, **extra) -> float:
        """Gets the learning rate for the given step."""
        pass
