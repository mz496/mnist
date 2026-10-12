from typing import Callable

from torch import Tensor


# For enforcing project-wide types.

type CrossEntropyLossFunction = Callable[[Tensor, Tensor], Tensor]
