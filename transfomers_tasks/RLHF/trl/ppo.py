__all__ = ['AdaptiveKLController', 'FixedKLController', 'PPOTrainer']

import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random

from transformers import DataCollatorForLanguageModeling
from .core import logprobs_from_logits


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass



class PPOTrainer:
    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
    }

    def __init__(self, model, ref_model, tokenizer, **ppo_params):
        """
        Initialize PPOTrainer

        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            tokenizer (tokenizer): Hugging Face tokenizer
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000
        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)

        self.ref_model = ref_model
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])

        if self.ppo_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                           self.ppo_params['target'],
                                           self.ppo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])


    def step(self, queries, response, scores):
        bs = self.ppo_params['batch_size']
        assert bs == len(queries), f'Batch size ({bs}) does not match number of queries ({len(queries)})'

        timing = dict()
        t0 = time.time()

        response_lengths = [len(r) for r in response]

        t = time.time()
        logprobs, ref_logprobs, values = self.batched_forward_pass(queries, response, response_lengths)


    def batched_forward_pass(self, queries, response):
        bs = self.ppo_params['batch_size']
        fbs = self.ppo_params['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        for i in range(int(bs / fbs)):
            query_batch = queries[i * fbs: (i + 1) * fbs]
            response_batch = response[i * fbs : (i + 1) * fbs]
            input_ids = self.data_collator([torch.cat(q, r) for q, r in zip(query_batch, response_batch)])['input_ids']
            with torch.no_grad():
                logits, _, v = self.model(input_ids)
                ref_logits, _, _ = self.ref_model(input_ids)
            logprobs = logprobs_from_logits(logits, input_ids)
            ref_logprobs = logprobs_from_logits(ref_logits, input_ids)

            for j in range(fbs):
                start = len(query_batch[j]) - 1
                end = start + len(response_batch[j])
                all_values.append(v[j, start:end])           # 生成的tokens的value
                all_logprobs.append(logprobs[j, start:end])  # 生成的tokens的概率
                all_ref_logprobs.append(ref_logprobs[j, start:end])

        return all_logprobs, all_ref_logprobs, all_values