from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

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


checkpoint = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print_number_of_trainable_model_params(model)

def tokenize_function(batch):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in batch['dialogue']]
    batch['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    batch['labels'] = tokenizer(batch['summary'], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return batch

dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(dataset_name)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['id', 'dialogue', 'summary', 'topic'])

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

peft_model = get_peft_model(
    model,
    lora_config
)

print_number_of_trainable_model_params(peft_model)
peft_model.print_trainable_parameters()

output_dir = f"peft-dialogue-summary-training"
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=5,
    weight_decay=0.01
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation']
)

peft_trainer.train()

peft_model_path="./peft-dialogue-summary-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)