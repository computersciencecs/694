# Configure different instruction tuning methods, different LLMs, fine-tune first and then inference or zeroshot through the command line to implement the main program.
import os
import shutil
import subprocess
import argparse
import sys
from config import (
    TaskType, ModelType, TrainingType, 
    TRAIN_CONFIG, INFERENCE_CONFIG, GPT_CONFIG, 
    FILE_PATHS
)

def copy_file_to_directory(source_file, target_dir):
    """Copy files to the target directory"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.copy2(source_file, os.path.join(target_dir, os.path.basename(source_file)))

def run_command(command, cwd=None):
    """Run commands and output directly to the terminal"""
    try:
        # Use subprocess.run to pass output directly to the terminal
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,  # If the command returns a non-zero status code, an exception is thrown.
            text=True,   # Using Text Mode
            stdout=sys.stdout,  # Direct output to the terminal
            stderr=sys.stderr   # Output errors directly to the terminal
        )
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed: {e}")
        return e.returncode

def execute_task(task_type, model_type, training_type, token, adapter_dir, skip_train, openai_api_key, **kwargs):
    """Execute the mission"""
    # Build the target directory path
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), training_type.value)
    
    if task_type == TaskType.FINE_TUNE:
        if model_type in [ModelType.LLAMA_7B, ModelType.LLAMA_8B]:
            if not token:
                raise ValueError("Llama tasks must provide --token")
            # Copy training and inference files
            copy_file_to_directory(FILE_PATHS["train"], target_dir)
            copy_file_to_directory(FILE_PATHS["inference"], target_dir)
            
            # Build training command
            train_cmd = f"""python train.py \
                --token {token} \
                --model_name {model_type.value} \
                --data_path ./{training_type.value}.jsonl \
                --output_dir ./results \
                --epochs {kwargs.get('epochs', TRAIN_CONFIG['epochs'])} \
                --batch_size {kwargs.get('batch_size', TRAIN_CONFIG['batch_size'])} \
                --grad_acc_steps {kwargs.get('grad_acc_steps', TRAIN_CONFIG['grad_acc_steps'])} \
                --lr {kwargs.get('lr', TRAIN_CONFIG['lr'])} \
                --max_length {kwargs.get('max_length', TRAIN_CONFIG['max_length'])}"""
            
            # Build inference command
            inference_cmd = f"""python inference.py \
                --token {token} \
                --model {model_type.value} \
                --data_file ./{training_type.value}test.jsonl \
                --batch_size {kwargs.get('inference_batch_size', INFERENCE_CONFIG['batch_size'])} \
                --tensor_parallel_size {kwargs.get('tensor_parallel_size', INFERENCE_CONFIG['tensor_parallel_size'])} \
                --adapter_dir {adapter_dir}"""
            
            if not skip_train:
                # Execute training
                print("Start training...")
                return_code = run_command(train_cmd, target_dir)
                if return_code != 0:
                    return return_code
            else:
                print("Skip training and go straight to inference...")
            # Execute inference
            print("Start reasoning...")
            return_code = run_command(inference_cmd, target_dir)
            return return_code
    
    elif task_type == TaskType.ZERO_SHOT:
        if model_type == ModelType.GPT_3_5:
            # Copy GPT inference file
            copy_file_to_directory(FILE_PATHS["gptinference"], target_dir)
            
            # Build GPT inference command
            gpt_cmd = f"""python gptinference.py \
                --data_file ./{training_type.value}test.jsonl \
                --batch_size {kwargs.get('gpt_batch_size', GPT_CONFIG['batch_size'])} \
                --training_type {training_type.value}"""
            if openai_api_key:
                gpt_cmd += f" \
                --openai_api_key {openai_api_key}"
            
            # Perform GPT inference
            print("Start GPT reasoning...")
            return_code = run_command(gpt_cmd, target_dir)
            return return_code
        
        elif model_type in [ModelType.LLAMA_7B, ModelType.LLAMA_8B]:
            if not token:
                raise ValueError("Llama tasks must provide --token")
            # Copy zero-shot inference file
            copy_file_to_directory(FILE_PATHS["zeroshot"], target_dir)
            
            # Build zero-shot inference command
            zeroshot_cmd = f"""python zeroshot.py \
                --model {model_type.value} \
                --data_file ./{training_type.value}test.jsonl \
                --batch_size {kwargs.get('inference_batch_size', INFERENCE_CONFIG['batch_size'])} \
                --tensor_parallel_size {kwargs.get('tensor_parallel_size', INFERENCE_CONFIG['tensor_parallel_size'])} \
                --token {token}"""
            
            # Perform zero-shot inference
            print("Starting zero-shot inference...")
            return_code = run_command(zeroshot_cmd, target_dir)
            return return_code

def parse_args():
    parser = argparse.ArgumentParser(description="Perform training or inference tasks")
    
    # Task Type
    parser.add_argument("--task_type", type=str, required=True, 
                      choices=["fine_tune", "zero_shot"],
                      help="Task type: fine_tune or zero_shot")
    
    # Model Type
    parser.add_argument("--model_type", type=str, required=True,
                      choices=["meta-llama/Llama-2-7b-hf", 
                              "meta-llama/Llama-3.1-8B-Instruct",
                              "gpt-3.5-turbo-0125"],
                      help="Model type")
    
    # Training type
    parser.add_argument("--training_type", type=str, required=True,
                      choices=["pointwise", "pairwise", "listwise"],
                      help="Training type：pointwise、pairwise or listwise")
    
    # Token
    parser.add_argument("--token", type=str, required=False, help="HuggingFace token，Only required for Llama missions")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="./results",
                      help="Output Directory")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training rounds")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Training batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=64,
                      help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5,
                      help="Learning Rate")
    parser.add_argument("--max_length", type=int, default=2048,
                      help="Maximum sequence length")
    
    # Inference parameters
    parser.add_argument("--inference_batch_size", type=int, default=80,
                      help="Inference batch size")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                      help="Tensor parallel size")
    parser.add_argument("--gpt_batch_size", type=int, default=10,
                      help="GPT inference batch size")
    
    parser.add_argument("--adapter_dir", type=str, default="./results", help="LoRA adapter directory, support checkpoint")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and go straight to inference")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API Key, only required for GPT tasks")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Convert parameters to enumeration types
    task_type = TaskType(args.task_type)
    model_type = ModelType(args.model_type)
    training_type = TrainingType(args.training_type)
    
    # Execute tasks
    return_code = execute_task(
        task_type=task_type,
        model_type=model_type,
        training_type=training_type,
        token=args.token,
        adapter_dir=args.adapter_dir,
        skip_train=args.skip_train,
        openai_api_key=args.openai_api_key,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_acc_steps=args.grad_acc_steps,
        lr=args.lr,
        max_length=args.max_length,
        inference_batch_size=args.inference_batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpt_batch_size=args.gpt_batch_size
    )
    
    # Determine whether the task is successful based on the return code
    if return_code == 0:
        print("Mission successfully completed")
    else:
        print("Mission Failed") 
