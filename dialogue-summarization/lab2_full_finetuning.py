from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

def print_number_of_trainable_model_params(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()

    print("Trainable model parameters: ", trainable_model_params)
    print("All model parameters: ", all_model_params)
    print("Percentage of trainable model parameters: ", (trainable_model_params / all_model_params) * 100)



dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(dataset_name)

checkpoint = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print_number_of_trainable_model_params(model)

index = 200

dialogue = dataset['train'][index]['dialogue']
summary = dataset['train'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs['input_ids'],
        max_new_tokens=200
    )[0],
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example['summary'], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['id', 'dialogue', 'summary', 'topic'])

print(tokenized_dataset)

# tokenized_dataset = tokenized_dataset.filter(lambda example, index: index % 1 == 0, with_indices=True)
# print(tokenized_dataset)

output_dir = f'dialogue-summary-training'

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation']
)

trainer.train()