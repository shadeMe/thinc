from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
import sys

from .abc import Optimizer
from .param import OptimizerParamInfo
from .types import KeyT
from .util import convert_to_schedule
from .thinc_optimizer import ADAM_DEFAULTS

from ..backends import get_array_ops
from ..compat import torch
from ..config import registry
from ..schedules import Schedule
from ..util import xp2torch, assert_pytorch_installed
from ..types import FloatsXd


@registry.optimizers("TorchAdam.v1")
def TorchAdam(
    learn_rate: float = ADAM_DEFAULTS["learn_rate"],
    *,
    L2: float = ADAM_DEFAULTS["L2"],
    beta1: float = ADAM_DEFAULTS["beta1"],
    beta2: float = ADAM_DEFAULTS["beta2"],
    eps: float = ADAM_DEFAULTS["eps"],
    grad_clip: float = ADAM_DEFAULTS["grad_clip"],
    L2_is_weight_decay: bool = cast(bool, ADAM_DEFAULTS["L2_is_weight_decay"]),
    use_averages: bool = True,
):
    kwargs = {"lr": learn_rate, "betas": (beta1, beta2), "eps": eps, "weight_decay": L2}
    factory = torch.optim.AdamW if L2_is_weight_decay else torch.optim.Adam
    return TorchOptimizer(
        factory, kwargs, grad_clip=grad_clip, use_averages=use_averages
    )


@dataclass
class _TrackedParam:
    source: OptimizerParamInfo
    # Wrapper tensor if the source is an XP tensor.
    param: "torch.Tensor"
    # Wrapper tensor if the source is an XP tensor.
    grad: Optional["torch.Tensor"]
    num_updates: int

    def __init__(self, source: OptimizerParamInfo):
        self.source = source

        if source.from_xp:
            assert source.gradient is not None
            self.param = xp2torch(
                source.parameter, requires_grad=True, allow_copy=False
            )
            self.grad = xp2torch(source.gradient, allow_copy=False)
            self.param.grad = self.grad
        else:
            # No special handling required for Torch tensors.
            assert source.from_torch and source.gradient is None
            self.param = source.parameter
            self.grad = None

        self.num_updates = 0


class TorchOptimizer(Optimizer):
    _OPTIMIZER_PARAM_GROUP_TAG_KEY = "__tracked_key__"
    _OPTIMIZER_PARAM_GROUP_TAG_DELETED = "__deleted_param__"

    _optimizer: "torch.optim.Optimizer"
    _tracked_params: Dict[KeyT, _TrackedParam]
    _per_param_group_options: Dict[str, Dict[str, Schedule]]
    _default_options: Dict[str, Schedule]
    _current_step: int
    _last_score: Optional[Tuple[int, float]]
    _averages: Optional[Dict[KeyT, FloatsXd]]
    _grad_clip: float

    def __init__(
        self,
        optimizer_factory: Any,
        default_options: Dict[str, Any],
        *,
        use_averages: bool = True,
        grad_clip: float = 1.0,
    ) -> None:
        assert_pytorch_installed()

        self._tracked_params = {}
        self._per_param_group_options = {}
        self._default_options = {
            k: convert_to_schedule(v, k) for k, v in default_options.items()
        }
        self._current_step = 0
        self._last_score = None
        self._averages = {} if use_averages else None
        self._grad_clip = grad_clip

        optimizer_kwargs = {
            k: v(self._current_step, **{"last_score": self._last_score})
            for k, v in self._default_options.items()
        }
        self._optimizer = optimizer_factory(
            [
                {
                    "params": torch.zeros(1),
                    self._OPTIMIZER_PARAM_GROUP_TAG_KEY: (
                        -sys.maxsize + 1,
                        "__sentinel__",
                    ),
                }
            ],  # Use a dummy, no-grad parameter to initialize the optimizer.
            **optimizer_kwargs,
        )

    def initialize(self, initial_params: Iterable[OptimizerParamInfo]) -> None:
        # We'll only register torch params here to keep things simple.
        # Thinc parameters will be registered in `Model.finish_update` calls.
        for param in initial_params:
            if param.from_torch:
                self.register_param(param)

    def register_param(
        self, param: OptimizerParamInfo, overwrite: bool = False
    ) -> None:
        if param.from_torch:
            self._register_torch_param(param, overwrite=overwrite)
        elif param.from_xp:
            self._register_thinc_param(param, overwrite=overwrite)
        else:
            raise ValueError(
                "Torch optimizer only supports weights and gradients stored in Torch or XP tensors"
            )

    def step(self) -> None:
        self._clip_gradients()
        self._optimizer.step()
        self._perform_post_update_fixups()
        self._step_schedules()
        self._update_averages()

    def __call__(
        self,
        key: KeyT,
        weights: FloatsXd,
        gradient: FloatsXd,
        *,
        lr_scale: float = 1.0,
    ):
        raise NotImplemented

    @property
    def last_score(self) -> Optional[Tuple[int, float]]:
        return self._last_score

    @last_score.setter
    def last_score(self, score: float):
        self._last_score = (self._current_step, score)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def averages(self):
        return self._averages

    def learn_rate(self, step: int, **extra) -> float:
        return self._default_options["lr"](step=step, **extra)

    def _step_schedules(self):
        # Since torch learning rate schedulers are only able to work on all parameter
        # groups at once, we need to roll out own scheduling code to support per-parameter
        # group schedules. The additional benefit of this method is that we can easily
        # extend scheduling support to optimizer options beyond learning rate.
        self._current_step += 1
        schedule_args = {
            "last_score": self.last_score,
        }
        for param_group in self._optimizer.param_groups:
            if self._OPTIMIZER_PARAM_GROUP_TAG_DELETED in param_group:
                continue

            key = param_group.get(self._OPTIMIZER_PARAM_GROUP_TAG_KEY)
            tracked_param = self._tracked_params.get(key)
            assert tracked_param is None

            schedule_source = self._default_options
            if key[1] in self._per_param_group_options:
                schedule_source = self._per_param_group_options[key[1]]

            for name, schedule in schedule_source.items():
                assert name in param_group
                new_val = schedule(self._current_step, **schedule_args)
                param_group[name] = new_val

    def _perform_post_update_fixups(self):
        # `torch.Optimizer.zero_grad()` resets the gradient tensors of
        # tracked parameters to `None` as a way to lower the memory footprint.
        # This is undesirable behaviour for us as we need to preserve the
        # wrapper gradient tensors of thinc parameters to ensure that the
        # correct backing store is updated.
        #
        # We can disable this behaviour by pass `set_to_none=False` to
        # `zero_grad()`, but this applies to all parameters tracked by the
        # optimizer, which includes the pure-torch parameters that can afford to
        # have their gradient tensors reset to `None`.Therefore, we manually
        # reset the gradient tensors of those parameters.
        self._optimizer.zero_grad(set_to_none=False)
        for param_group in self._optimizer.param_groups:
            if self._OPTIMIZER_PARAM_GROUP_TAG_DELETED in param_group:
                continue

            key = param_group.get(self._OPTIMIZER_PARAM_GROUP_TAG_KEY)
            tracked_param = self._tracked_params.get(key)
            assert tracked_param is not None

            tensors: List["torch.Tensor"] = param_group["params"]
            assert len(tensors) == 1
            tensor = tensors[0]
            if tracked_param.source.from_torch and tensor.grad is not None:
                tensor.grad = None

            # Invoke post-update callbacks to ensure that the parameter update
            # has been communicated to the source model. This is generally not
            # necessary for pure-torch parameters.
            if tracked_param.source.update_callback is not None:
                tracked_param.source.update_callback(tracked_param.source.parameter)

    def _clip_gradients(self):
        if self._grad_clip <= 0.0:
            return

        # TODO should we do this with a backward_hook instead of doing it after
        # backprop?
        # c.f https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                torch.nn.utils.clip_grad_norm_(p, self._grad_clip)

    def _update_averages(self):
        if self._averages is None:
            return

        # TODO improve impl
        for key, tracked_param in self._tracked_params.items():
            tracked_param.num_updates += 1
            # TODO do we need to do this for torch params as well?
            if tracked_param.source.from_xp:
                weights = tracked_param.source.parameter
                ops = get_array_ops(weights)
                if key not in self.averages:
                    self._averages[key] = ops.alloc(weights.shape, dtype="float32")
                ops.update_averages(
                    self.averages[key], weights, tracked_param.num_updates
                )

    def _register_torch_param(
        self,
        param: OptimizerParamInfo,
        *,
        overwrite: bool = False,
    ):
        if param.key in self._tracked_params:
            # XXX No reason to support overwrites as we expect all torch tensors to be
            # registered at most once.
            # TODO Can we rely on that as an invariant?
            raise ValueError(
                f"Torch parameter with key `{param.key}` has already been registered"
            )
        elif param.gradient is not None:
            raise ValueError("Unexpected gradient for Torch tensor")
        elif not param.parameter.requires_grad:
            raise ValueError(
                "Attempting to register a Torch parameter that is detached from the graph"
            )

        tracked_param = _TrackedParam(param)
        self._tracked_params[param.key] = tracked_param
        self._add_param_to_optimizer(tracked_param)

    def _register_thinc_param(
        self,
        param: OptimizerParamInfo,
        *,
        overwrite: bool = False,
    ):
        if param.gradient is None:
            raise ValueError("Missing gradient tensor for Thinc parameter")

        existing = self._tracked_params.get(param.key)
        if existing is not None:
            if overwrite:
                self._deregister_param(param.key)
            elif id(existing.source.parameter) != id(param.parameter) or id(
                existing.source.gradient
            ) != id(param.gradient):
                raise ValueError(
                    "Attempting to re-register a Thinc parameter with a different backing store "
                    "than the one found in the previous registration"
                )
            else:
                # Already registered, so it's a no-op.
                return

        assert param.key not in self._tracked_params
        tracked_param = _TrackedParam(param)
        self._tracked_params[param.key] = tracked_param
        self._add_param_to_optimizer(tracked_param)

    def _deregister_param(self, key: KeyT):
        tracked_param = self._tracked_params.get(key)
        assert tracked_param is not None

        self._remove_param_from_optimizer(tracked_param)
        del self._tracked_params[key]

    def _add_param_to_optimizer(self, param: _TrackedParam):
        assert param.param.requires_grad and param.param.grad is not None
        # We only register a single parameter for each torch param group
        # to keep the scheduling and tracking logic simple.

        # TODO check if this has a performance impact. The CPU kernels
        # for commmon optimizers seem to be iterating using a loop, but
        # fused implementations could suffer.
        self._optimizer.add_param_group(
            {
                "params": param.param,
                # Save the tracking key as part of the parameter group's custom
                # options. We can rely on this as PyTorch won't strip custom keys.
                self._OPTIMIZER_PARAM_GROUP_TAG_KEY: param.source.key,
            }
        )

    def _remove_param_from_optimizer(self, param: _TrackedParam):
        # We can't directly remove a param group from the optimizer as
        # it can be using adjacent lists to track state like momentum.
        # So, we'll just "freeze" them by disabling their gradient, which
        # will cause them to be ignored by the optimizer.
        # https://discuss.pytorch.org/t/delete-parameter-group-from-optimizer/46814/8
        assert param.param.requires_grad

        for param_group in self._optimizer.param_groups:
            key = param_group.get(self._OPTIMIZER_PARAM_GROUP_TAG_KEY)
            assert key is not None
            if param.source.key == key:
                assert self._OPTIMIZER_PARAM_GROUP_TAG_DELETED not in param_group

                tensors: List["torch.Tensor"] = param_group["params"]
                assert len(tensors) == 1
                tensor = tensors[0]
                assert tensor.grad is not None
                tensor.grad.zero_()
                tensor.requires_grad_(False)
                tensor.grad = None
                param_group[self._OPTIMIZER_PARAM_GROUP_TAG_DELETED] = True

                break
