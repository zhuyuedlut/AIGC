import torch.cuda
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from transformers import AutoTokenizer, pipeline
from datasets import load_dataset


def build_dataset(config: PPOConfig, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_column({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample['input_ids'] = tokenizer.encode(sample['review'], return_tensors='pt')[:input_size()]
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type='torch')
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


if __name__ == '__main__':
    config = PPOConfig(model_name="lvwerra/gpt2-imdb", learning_rate=1.41e-5)
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

    dataset = build_dataset(config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else 'cpu'

    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    text = "this movie was really bad!!"
    sentiment_pipe(text, **sent_kwargs)

    text = "this movie was really good!!"
    sentiment_pipe(text, **sent_kwargs)
