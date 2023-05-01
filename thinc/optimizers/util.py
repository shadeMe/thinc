import itertools
from types import GeneratorType
from typing import Any, Generator, Tuple, Union

from ..schedules import constant, Schedule

from .types import InvokableScheduleT, ScheduleT


def is_convertable_to_schedule(schedulable: Any) -> bool:
    if isinstance(schedulable, (float, bool, int, list)):
        return True
    elif isinstance(schedulable, (GeneratorType, Schedule)):
        return True
    else:
        return False


def convert_to_schedule(schedulable: ScheduleT, name: str) -> InvokableScheduleT:
    if isinstance(schedulable, (float, bool, int)):
        return constant(schedulable)
    elif isinstance(schedulable, list):
        value = iter(schedulable)
        return _wrap_generator_as_schedule(name, value)  # type:ignore
    elif isinstance(schedulable, GeneratorType):
        return _wrap_generator_as_schedule(name, schedulable)
    elif isinstance(schedulable, Schedule):
        return schedulable
    elif isinstance(schedulable, tuple) and all(
        (is_convertable_to_schedule(x) for x in schedulable)
    ):
        # Special case for torch optimizer options that store values in tuples.
        return tuple(convert_to_schedule(x, name) for x in schedulable)  # type:ignore
    else:
        raise ValueError(f"Invalid schedule for '{name}' ({type(schedulable)})")


def invoke_schedule(
    schedulable: InvokableScheduleT, step: int, **schedule_args
) -> Union[Any, Tuple[Any, ...]]:
    if isinstance(schedulable, Schedule):
        return schedulable(step=step, **schedule_args)
    elif isinstance(schedulable, tuple) and all(
        (isinstance(x, Schedule) for x in schedulable)
    ):
        return tuple(s(step=step, **schedule_args) for s in schedulable)
    else:
        raise ValueError(
            "Attempting to invoke a schedule that is neither a `Schedule` nor "
            "a `Tuple[Schedule, ...]"
        )


def _wrap_generator_as_schedule(attr_name: str, generator: Generator) -> Schedule[Any]:
    try:
        peek = next(generator)
    except (StopIteration, TypeError) as e:
        err = f"Invalid schedule for '{attr_name}' ({type(generator)})\n{e}"
        raise ValueError(err)
    return Schedule(
        "wrap_generator",
        _wrap_generator_schedule,
        attrs={
            "attr_name": attr_name,
            "last_step": -1,
            "last_value": peek,
            "generator": itertools.chain([peek], generator),
        },
    )


def _wrap_generator_schedule(schedule: Schedule, step, **kwargs) -> float:
    attr_name = schedule.attrs["attr_name"]
    last_step = schedule.attrs["last_step"]
    last_value = schedule.attrs["last_value"]
    generator = schedule.attrs["generator"]

    if step < last_step:
        raise ValueError(
            f"'step' of the generator-based schedule for {attr_name} must not decrease"
        )

    # Ensure that we have a value when we didn't step or when the
    # generator is exhausted.
    value = last_value

    for i in range(step - last_step):
        try:
            value = next(generator)
        except StopIteration:  # schedule exhausted, use last value
            break

    schedule.attrs["last_step"] = step
    schedule.attrs["last_value"] = value

    return value
