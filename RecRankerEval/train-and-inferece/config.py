from enum import Enum

class TaskType(Enum):
    FINE_TUNE = "fine_tune"
    ZERO_SHOT = "zero_shot"

class ModelType(Enum):
    LLAMA_7B = "meta-llama/Llama-2-7b-hf"
    LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
    GPT_3_5 = "gpt-3.5-turbo-0125"

class TrainingType(Enum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    LISTWISE = "listwise"

# 训练配置
TRAIN_CONFIG = {
    "epochs": 3,
    "batch_size": 2,
    "grad_acc_steps": 64,
    "lr": 2e-5,
    "output_dir": "./results",
    "max_length": 2048
}

# 推理配置
INFERENCE_CONFIG = {
    "batch_size": 80,
    "tensor_parallel_size": 1,
    "temperature": 0.1,
    "top_k": 40,
    "top_p": 0.1,
    "max_tokens": 2048
}

# GPT推理配置
GPT_CONFIG = {
    "batch_size": 10,
    "temperature": 0.1,
    "top_p": 0.1,
    "max_tokens": 2048
}

# 文件路径配置
FILE_PATHS = {
    "train": "train.py",
    "inference": "inference.py",
    "zeroshot": "zeroshot.py",
    "gptinference": "gptinference.py"
} 