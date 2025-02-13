from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, GPT2LMHeadModel
from transformers import EarlyStoppingCallback
from datasets import concatenate_datasets
import os
import torch
import numpy as np
from math import exp
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F

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
print("dataset pranil: ", dataset_7['train'][0])

# Standardize column names
dataset_3 = dataset_3.rename_columns({"Context": "input", "Response": "output"})
dataset_4 = dataset_4.rename_columns({"question": "input", "response_j": "output"})
dataset_7 = dataset_7.rename_columns({"Patient's Context": "input", "Psychiatrist's Response": "output"})

# Select relevant columns
datasets = [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6_selected, dataset_7]
datasets = [ds.select_columns(["input", "output"]) for ds in datasets]
print("no of datasets: ", len(datasets))

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
print("length of tokenized datset: ", len(tokenized_datasets))

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

metrics_history = {"loss": [], "perplexity": []}  

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = F.log_softmax(torch.tensor(logits), dim=-1)

    # Compute cross-entropy loss instead of MSE
    loss = F.nll_loss(probs.view(-1, probs.shape[-1]), torch.tensor(labels).view(-1)).item()

    # Compute Perplexity
    perplexity = np.exp(loss) if loss < 300 else float('inf')

    # Append to metrics history
    metrics_history["loss"].append(loss)
    metrics_history["perplexity"].append(perplexity)

    return {"loss": loss, "perplexity": perplexity}


#training argument define
training_args = TrainingArguments(
    output_dir="/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_results",
    evaluation_strategy="epoch", 
    save_strategy="epoch",  # Save checkpoint per epoch
    num_train_epochs=20,
    per_device_train_batch_size=4,  # Reduce batch size if needed
    per_device_eval_batch_size=4,
    warmup_steps=1500,  
    weight_decay=0.01,
    logging_dir="/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_logs",
    logging_strategy="epoch", 
    learning_rate=3e-5,  # Reduce LR for stability
    report_to=["tensorboard"],
    fp16=True,  # Use mixed precision training
    save_total_limit=4, 
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    log_level="info"
)


# Add Early Stopping Callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3  # Stop if validation loss doesn't improve for 3 evaluations
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset['train'],
    eval_dataset=final_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [early_stopping]
)

# Training
# checkpoint_path = '/home/nil/python_projects/gpt2_finetuned_45k_10epochs/results/checkpoint-20000'
# if os.path.exists(checkpoint_path):
#     trainer.train(resume_from_checkpoint=checkpoint_path)
# else:
#     trainer.train()

trainer.train()

# Save model
model_output_dir = '/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_results/new_model'
os.makedirs(model_output_dir, exist_ok=True)
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

# Save metrics history for later plotting
with open('/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_results/metrics_history.json', 'w') as f:
    json.dump(metrics_history, f)
