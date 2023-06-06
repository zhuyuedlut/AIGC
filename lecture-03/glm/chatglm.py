import math
import copy
import os
import warnings
import re
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F

from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any