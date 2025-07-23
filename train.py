import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load Dataset
dataset = load_dataset("json", data_files="self_instruct_uz.json")["train"].train_test_split(test_size=0.05)

# 2. Load Tokenizer
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # required for causal LM padding

# 3. Load Mistral 7B with 4-bit quantization 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# 4. Apply LoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. Prepare dataset 
def format_example(example):
    return f"### Ko'rsatma:\n{example['instruction']}\n### Kirish:\n{example['input']}\n### Natija:\n{example['output']}"

def tokenize_function(example):
    return tokenizer(format_example(example), truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=False)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./mistral-uzbek-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)


trainer.train()

# 9. Save model and tokenizer
model.save_pretrained("./mistral-uzbek-lora")
tokenizer.save_pretrained("./mistral-uzbek-lora")