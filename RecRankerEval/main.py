import os
import shutil
import subprocess
import argparse
import sys
from train_and_inferece.config import (
    TaskType, ModelType, TrainingType, 
    TRAIN_CONFIG, INFERENCE_CONFIG, GPT_CONFIG, 
    FILE_PATHS
)

def copy_file_to_directory(source_file, target_dir):
    """复制文件到目标目录"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.copy2(source_file, os.path.join(target_dir, os.path.basename(source_file)))

def run_command(command, cwd=None):
    """运行命令并直接输出到终端"""
    try:
        # 使用subprocess.run，将输出直接传递给终端
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,  # 如果命令返回非零状态码则抛出异常
            text=True,   # 使用文本模式
            stdout=sys.stdout,  # 直接输出到终端
            stderr=sys.stderr   # 直接输出错误到终端
        )
        return 0
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        return e.returncode

def execute_task(task_type, model_type, training_type, token, adapter_dir, skip_train, openai_api_key, **kwargs):
    """执行任务"""
    # 构建目标目录路径
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_and_inferece', training_type.value)
    
    if task_type == TaskType.FINE_TUNE:
        if model_type in [ModelType.LLAMA_7B, ModelType.LLAMA_8B]:
            if not token:
                raise ValueError("Llama 任务必须提供 --token")
            # 复制训练和推理文件
            copy_file_to_directory(FILE_PATHS["train"], target_dir)
            copy_file_to_directory(FILE_PATHS["inference"], target_dir)
            
            # 构建训练命令
            train_cmd = f"""python train.py \
                --token {token} \
                --model_name {model_type.value} \
                --data_path ../{training_type.value}.jsonl \
                --output_dir ./results \
                --epochs {kwargs.get('epochs', TRAIN_CONFIG['epochs'])} \
                --batch_size {kwargs.get('batch_size', TRAIN_CONFIG['batch_size'])} \
                --grad_acc_steps {kwargs.get('grad_acc_steps', TRAIN_CONFIG['grad_acc_steps'])} \
                --lr {kwargs.get('lr', TRAIN_CONFIG['lr'])} \
                --max_length {kwargs.get('max_length', TRAIN_CONFIG['max_length'])}"""
            
            # 构建推理命令
            inference_cmd = f"""python inference.py \
                --token {token} \
                --model {model_type.value} \
                --data_file ../{training_type.value}test.jsonl \
                --batch_size {kwargs.get('inference_batch_size', INFERENCE_CONFIG['batch_size'])} \
                --tensor_parallel_size {kwargs.get('tensor_parallel_size', INFERENCE_CONFIG['tensor_parallel_size'])} \
                --adapter_dir {adapter_dir}"""
            
            if not skip_train:
                # 执行训练
                print("开始训练...")
                return_code = run_command(train_cmd, target_dir)
                if return_code != 0:
                    return return_code
            else:
                print("跳过训练，直接推理...")
            # 执行推理
            print("开始推理...")
            return_code = run_command(inference_cmd, target_dir)
            return return_code
    
    elif task_type == TaskType.ZERO_SHOT:
        if model_type == ModelType.GPT_3_5:
            # 复制GPT推理文件
            copy_file_to_directory(FILE_PATHS["gptinference"], target_dir)
            
            # 构建GPT推理命令
            gpt_cmd = f"""python gptinference.py \
                --data_file ../{training_type.value}test.jsonl \
                --batch_size {kwargs.get('gpt_batch_size', GPT_CONFIG['batch_size'])} \
                --training_type {training_type.value}"""
            if openai_api_key:
                gpt_cmd += f" \
                --openai_api_key {openai_api_key}"
            
            # 执行GPT推理
            print("开始GPT推理...")
            return_code = run_command(gpt_cmd, target_dir)
            return return_code
        
        elif model_type in [ModelType.LLAMA_7B, ModelType.LLAMA_8B]:
            if not token:
                raise ValueError("Llama 任务必须提供 --token")
            # 复制零样本推理文件
            copy_file_to_directory(FILE_PATHS["zeroshot"], target_dir)
            
            # 构建零样本推理命令
            zeroshot_cmd = f"""python zeroshot.py \
                --model {model_type.value} \
                --data_file ../{training_type.value}test.jsonl \
                --batch_size {kwargs.get('inference_batch_size', INFERENCE_CONFIG['batch_size'])} \
                --tensor_parallel_size {kwargs.get('tensor_parallel_size', INFERENCE_CONFIG['tensor_parallel_size'])} \
                --token {token}"""
            
            # 执行零样本推理
            print("开始零样本推理...")
            return_code = run_command(zeroshot_cmd, target_dir)
            return return_code

def parse_args():
    parser = argparse.ArgumentParser(description="执行训练或推理任务")
    
    # 任务类型
    parser.add_argument("--task_type", type=str, required=True, 
                      choices=["fine_tune", "zero_shot"],
                      help="任务类型：fine_tune 或 zero_shot")
    
    # 模型类型
    parser.add_argument("--model_type", type=str, required=True,
                      choices=["meta-llama/Llama-2-7b-hf", 
                              "meta-llama/Llama-3.1-8B-Instruct",
                              "gpt-3.5-turbo-0125"],
                      help="模型类型")
    
    # 训练类型
    parser.add_argument("--training_type", type=str, required=True,
                      choices=["pointwise", "pairwise", "listwise"],
                      help="训练类型：pointwise、pairwise 或 listwise")
    
    # Token
    parser.add_argument("--token", type=str, required=False, help="HuggingFace token，仅Llama任务需要")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./results",
                      help="输出目录")
    parser.add_argument("--epochs", type=int, default=3,
                      help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="训练批次大小")
    parser.add_argument("--grad_acc_steps", type=int, default=64,
                      help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-5,
                      help="学习率")
    parser.add_argument("--max_length", type=int, default=2048,
                      help="最大序列长度")
    
    # 推理参数
    parser.add_argument("--inference_batch_size", type=int, default=80,
                      help="推理批次大小")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                      help="张量并行大小")
    parser.add_argument("--gpt_batch_size", type=int, default=10,
                      help="GPT推理批次大小")
    
    parser.add_argument("--adapter_dir", type=str, default="./results", help="LoRA adapter目录，支持checkpoint")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，直接推理")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API Key，仅GPT任务需要")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 转换参数为枚举类型
    task_type = TaskType(args.task_type)
    model_type = ModelType(args.model_type)
    training_type = TrainingType(args.training_type)
    
    # 执行任务
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
    
    # 根据返回码判断任务是否成功
    if return_code == 0:
        print("任务成功完成")
    else:
        print("任务失败") 