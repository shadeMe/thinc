import itertools
from types import GeneratorType
from typing import Any, Generator

from ..schedules import constant, Schedule

from .types import ScheduleT


def convert_to_schedule(schedulable: ScheduleT, name: str) -> Schedule[Any]:
    if isinstance(schedulable, (float, bool, int)):
        return constant(schedulable)
    elif isinstance(schedulable, list):
        value = iter(schedulable)
        return _wrap_generator_as_schedule(name, value)  # type:ignore
    elif isinstance(schedulable, GeneratorType):
        return _wrap_generator_as_schedule(name, schedulable)
    elif isinstance(schedulable, Schedule):
        return schedulable
    else:
        err = f"Invalid schedule for '{name}' ({type(schedulable)})"
        raise ValueError(err)


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
