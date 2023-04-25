from typing import Union, Tuple, List

from ..types import Generator
from ..schedules import Schedule

# Uniquely identifies a parameter tracked by the optimizer.
KeyT = Tuple[int, str]

ScheduleT = Union[float, List[float], Generator, Schedule]
