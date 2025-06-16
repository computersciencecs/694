import os
import torch
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with LoRA on a JSONL dataset.")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--data_path", type=str, default="./listwise.jsonl")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_acc_steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    
    return parser.parse_args()


def load_model_and_tokenizer(model_name, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def compute_token_length_stats(dataset, tokenizer):
    def calculate_lengths(batch):
        lengths = []
        for messages in batch["messages"]:
            inputs = [
                msg["content"][0]["content"]
                for msg in messages if msg["role"] == "user"
            ]
            if len(inputs) > 1:
                inputs = [" ".join(inputs)]
            tokenized = tokenizer(inputs, truncation=False, padding=False)
            lengths.extend([len(seq) for seq in tokenized["input_ids"]])
        return {"lengths": lengths}

    lengths_dataset = dataset.map(calculate_lengths, batched=True)
    all_lengths = lengths_dataset["lengths"]
    print(f"Mean token length: {np.mean(all_lengths)}")
    print(f"Median token length: {np.median(all_lengths)}")
    print(f"Max token length: {np.max(all_lengths)}")
    print(f"90/95/99 percentiles: {np.percentile(all_lengths, [90, 95, 99])}")


def preprocess_dataset(dataset, tokenizer, max_length=2048):
    def preprocess_function(examples):
        all_inputs, all_targets = [], []
        for messages in examples["messages"]:
            inputs = [msg["content"][0]["content"] for msg in messages if msg["role"] == "user"]
            targets = [msg["content"][0]["content"] for msg in messages if msg["role"] == "assistant"]
            all_inputs.extend(inputs)
            all_targets.extend(targets)

        model_inputs = tokenizer(all_inputs, max_length=max_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(all_targets, max_length=max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)


def get_trainer(model, tokenized_dataset, args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        fp16=True,
        learning_rate=args.lr,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )


def train_and_save(trainer, save_path="trained_model"):
    trainer.train()
    trainer.save_model(save_path)
    print(f"✅ Model saved to: {save_path}")


def main():
    args = parse_args()
    os.environ["HF_TOKEN"] = args.token

    print("🚀 Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.token)

    print("📂 Loading dataset...")
    raw_dataset = load_dataset("json", data_files=args.data_path)["train"]

    print("📊 Analyzing token lengths...")
    compute_token_length_stats(raw_dataset, tokenizer)

    print("🧹 Preprocessing dataset...")
    tokenized_dataset = preprocess_dataset(raw_dataset, tokenizer, max_length=args.max_length)

    print("🎯 Starting training...")
    trainer = get_trainer(model, tokenized_dataset, args)
    train_and_save(trainer)


if __name__ == "__main__":
    main()
