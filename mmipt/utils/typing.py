# Copyright (c) MMIPT. All rights reserved.
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from mmengine.config import ConfigDict
from mmengine.structures import BaseDataElement
from torch import Tensor

ForwardInputs = Tuple[Dict[str, Union[Tensor, str, int]], Tensor]
ForwardResults = Union[Dict[str, Tensor], List[BaseDataElement], Tuple[Tensor],
                       Tensor]

SampleList = Sequence[BaseDataElement]
OptSampleList = Optional[SampleList]
SampleInputs = Union[BaseDataElement, SampleList, None]

NoiseVar = Union[Tensor, Callable, None]
LabelVar = Union[Tensor, Callable, List[int], None]

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

# Type hint of one or more config data
MultiConfig = Union[ConfigType, Sequence[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

# Type hint of Tensor
TensorDict = Dict[str, Tensor]
TensorList = Sequence[Tensor]
MultiTensor = Union[Tensor, TensorDict, TensorList]
