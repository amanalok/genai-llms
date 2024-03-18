from transformers import AutoModelForSeq2SeqLM, Trainer, AutoTokenizer
from datasets import load_dataset
import evaluate
from peft import PeftModel, PeftConfig
import torch

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
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# print_number_of_trainable_model_params(model)

# full_finetuned_checkpoint = "./dialogue-summary-training/checkpoint-31000"
# model = AutoModelForSeq2SeqLM.from_pretrained(full_finetuned_checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

peft_finetuned_checkpoint = "./peft-dialogue-summary-checkpoint-local"
model = PeftModel.from_pretrained(base_model, peft_finetuned_checkpoint,
                                  torch_dtype=torch.bfloat16, is_trainable=False)
tokenizer = AutoTokenizer.from_pretrained(peft_finetuned_checkpoint)

print_number_of_trainable_model_params(model)

def generate_prediction(batch):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in batch['dialogue']]
    inputs = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt")
    generated_ids = model.generate(input_ids=inputs["input_ids"], max_new_tokens=200)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    batch['pred_summary'] = output
    return batch

# results = dataset['validation'].map(generate_prediction, batched=True)
results = dataset['validation'].select(indices=range(200)).map(generate_prediction, batched=True)

references = results['summary']
predictions = results['pred_summary']

rogue_metric = evaluate.load("rouge")
rogue_output = rogue_metric.compute(predictions=predictions, references=references)

print(rogue_output)