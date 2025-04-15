# ! pip install -q torch peft bitsandbytes transformers accelerate trl tqdm
import os
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, Dataset
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
)
dataset = load_dataset("gbharti/finance-alpaca")
print(dataset)
n = 100
text_data = []
for point in dataset["train"]:
    text = f"[INST] {point['instruction']} [/INST] {point['output']}"
    text_data.append({'text': text})
text_data = text_data[:n]

text_dataset = Dataset.from_list(text_data)
del(dataset)
print(text_dataset)
BASE_MODEL = "google/gemma-3-1b-pt"
NEW_MODEL = "finance-gemma"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)