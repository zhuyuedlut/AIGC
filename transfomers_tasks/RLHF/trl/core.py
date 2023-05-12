import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


import collections
import numpy as np

def logprobs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy
