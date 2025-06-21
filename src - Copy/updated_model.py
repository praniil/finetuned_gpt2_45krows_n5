import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, EarlyStoppingCallback

torch.cuda.empty_cache()

# Load datasets
dataset_1 = load_dataset("marmikpandya/mental-health")
dataset_2 = load_dataset("fadodr/mental_health_therapy")
dataset_3 = load_dataset("Amod/mental_health_counseling_conversations")
dataset_4 = load_dataset("jkhedri/psychology-dataset")
dataset_5 = load_dataset("samhog/psychology-6k")
dataset_6 = load_dataset("RAJJ18/mental_health_dataset")
dataset_6_selected = dataset_6['train'].shuffle(seed=42).select(range(min(3000, len(dataset_6['train']))))
dataset_7 = load_dataset("Pranilllllll/positive-conversations-dataset-mental-health")

# Standardize column names
dataset_3 = dataset_3.rename_columns({"Context": "input", "Response": "output"})
dataset_4 = dataset_4.rename_columns({"question": "input", "response_j": "output"})
dataset_7 = dataset_7.rename_columns({"Patient's Context": "input", "Psychiatrist's Response": "output"})

# Select relevant columns
datasets = [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6_selected, dataset_7]
datasets = [ds.select_columns(["input", "output"]) for ds in datasets]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    text = [f"<|input|> {inp} <|output|> {out}" for inp, out in zip(examples['input'], examples['output'])]
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

# Tokenize datasets
tokenized_datasets = [ds.map(tokenize_function, batched=True) for ds in datasets]

# Combine datasets
combined_dataset = concatenate_datasets([ds['train'] for ds in tokenized_datasets if 'train' in ds])
train_val_test = combined_dataset.train_test_split(test_size=0.2, seed=42)
val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=42)

final_dataset = {
    'train': train_val_test['train'],
    'validation': val_test['train'],
    'test': val_test['test']
}

# Model setup
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_results",
    eval_strategy="steps",
    save_steps=5000,
    eval_steps=5000,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=1500,
    weight_decay=0.01,
    logging_dir="/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_logs",
    logging_strategy="steps",
    logging_steps=5000,
    learning_rate=5e-5,
    report_to=["tensorboard"],
    fp16=True,
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    log_level="info"
)

# Early Stopping Callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset['train'],
    eval_dataset=final_dataset['validation'],
    tokenizer=tokenizer,
    callbacks=[early_stopping]
)

if __name__ == "__main__":
    checkpoint_path = '/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_results/checkpoint-85000'
    if os.path.exists(checkpoint_path):
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()
    
    # Save model
    model_output_dir = '/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_results/new_model'
    os.makedirs(model_output_dir, exist_ok=True)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
