from typing import NewType, Tuple

from environment.stack_wrapper import LazyFrames

Experience = NewType("Experience", Tuple[LazyFrames, LazyFrames, int, bool, float])
