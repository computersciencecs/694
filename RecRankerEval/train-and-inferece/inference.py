import argparse
import json
import re
import jsonlines
from fractions import Fraction
from vllm import LLM, SamplingParams
import sys
import os
import torch.distributed as dist

import torch


from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import login
token = "" 
login(token=token, add_to_git_credential=True)
MAX_INT = sys.maxsize


def load_finetuned_model(model_name, adapter_dir):
    # Load the base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the PEFT adapter configuration and apply it to the model
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    
    return model, tokenizer


# Function to load, merge, and save the fine-tuned model with LoRA adapter
def load_and_merge_model(model_name, adapter_dir, merged_model_dir='./merged_model'):
    # Load the base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the adapter and apply it to the model
    model = PeftModel.from_pretrained(model, adapter_dir)
    
    # Merge the adapter weights into the base model
    model = model.merge_and_unload()
    
    # Save the merged model to disk
    model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)
    
    return merged_model_dir


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size + (1 if len(data_list) % batch_size > 0 else 0)
    batch_data = []
    for i in range(n):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(data_list))
        batch_data.append(data_list[start:end])
    return batch_data




def my_test(model_path, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    stop_tokens = []
    sampling_params = SamplingParams(temperature=0.1, top_k=40, top_p=0.1, max_tokens=2048,stop=stop_tokens)  # stop=stop_tokens


    #####model, tokenizer = load_finetuned_model(model, adapter_dir)
    
    ###print('sampleing =====', sampling_params)
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)

    for kkk in ['LightGCN']:  # , 'SASRec','BERT4Rec','CL4SRec''SGL','MF', 'LightGCN', 'SGL',
        data_path = './pointwisetest.jsonl'
        INVALID_ANS = "[invalid]"
        res_ins = []
        res_answers = []
        problem_prompt = (
            "{instruction}"
        )
        with open(data_path, "r+", encoding="utf8") as f:
            for idx, item in enumerate(jsonlines.Reader(f)):
                temp_instr = problem_prompt.format(instruction=item["inst"])
                res_ins.append(temp_instr)
        
      
        print(f"Loaded {len(res_ins)} items for inference")
        print(f"Loaded res_ins for inference")
        
        #print('res_ins', res_ins)
        res_ins = res_ins[start:end]
        res_answers = res_answers[start:end]
        print('lenght ====', len(res_ins))
        batch_res_ins = batch_data(res_ins, batch_size=batch_size)
        result = []
        res_completions = []
        idx = 0
        for prompt in batch_res_ins:
            if isinstance(prompt, list):
                pass
            else:
                prompt = [prompt]
            completions = llm.generate(prompt, sampling_params)
            #completions = llm.generate(prompt)
            for output in completions:
                local_idx = 'INDEX ' + str(idx) + ':'
                prompt = output.prompt
                generated_text = output.outputs[0].text
                #print(generated_text[:10])
                generated_text = generated_text.replace('\n', '').replace('    ', '')
                generated_text = local_idx + generated_text
                res_completions.append(generated_text)
                idx += 1
    
        print(f"Total completions generated: {len(res_completions)}")
        print('res_completions', res_completions[:2])
        def write_list_to_file(string_list, output_file):
            with open(output_file, 'w') as file:
                for item in string_list:
                    file.write(item + '\n')
        import pandas as pd
        df = pd.DataFrame(res_completions)
        #df.to_csv('./res_completionsLightGCN_kg_1_fine.txt', index=None, header=None)
        df.to_csv('./inference.txt', index=None, header=None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='meta-llama/Llama-2-7b-hf')  # model path
    #parser.add_argument("--model", type=str, default='meta-llama/Llama-3.1-8B-Instruct')  # model path
    parser.add_argument("--data_file", type=str,
                        default='/data/path/')  # data path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=80)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size


    parser.add_argument("--adapter_dir", type=str, default='./trained_model', help="Path to the LoRA adapter directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    #my_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size,
               #tensor_parallel_size=args.tensor_parallel_size)

    merged_model_dir = load_and_merge_model(args.model, args.adapter_dir)
    my_test(model_path=merged_model_dir, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size)    
    
    ##my_test(model=args.model, adapter_dir=args.adapter_dir, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size,
               ##tensor_parallel_size=args.tensor_parallel_size)
  
    if dist.is_initialized():
        dist.destroy_process_group()
