import os
import time
import random

import torch
from rich import print
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer

config = {
    "model_name": 'uer/gpt2-chinese-cluecorpussmall',
    'step': 20000,
    'batch_size': 128,
    'forward_batch_size': 16,
    'ppo_epochs': 4,
    'lr': 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1,
    "gen_len": 16,
    "save_freq": 5,
    'save_dir': 'checkpoints/ppo_sentiment_gpt'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe_device = 0 if torch.cuda.is_available() else -1

prompts = [
    '刚收到货，感觉',
    '这部电影很',
    '说实话，真的很',
    '这次购物总的来说体验很'
]

senti_tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
senti_model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
sentiment_pipe = pipeline('sentiment-analysis', model=senti_model, tokenizer=senti_tokenizer, device=pipe_device)

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.eos_token = gpt2_tokenizer.pad_token
gpt2_model.to(device)
gpt2_model_ref.to(device)

gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}

ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)