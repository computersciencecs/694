import argparse
import jsonlines
import openai
import asyncio
import aiohttp
import os
import sys
import re
import time
from tqdm import tqdm

MAX_INT = sys.maxsize

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size + (1 if len(data_list) % batch_size > 0 else 0)
    return [data_list[i * batch_size:(i + 1) * batch_size] for i in range(n)]

async def generate_response_listwise(session, prompt, idx, progress_bar):
    time.sleep(1)
    try:
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}"},
            json={
                "model": "gpt-3.5-turbo-0125",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.1,
                "max_tokens": 2048
            }
        )
        response_json = await response.json()
        generated_text = response_json["choices"][0]["message"]["content"].strip()
        matches = re.findall(r"\d+\.\s*(.*?)$", generated_text, re.MULTILINE)
        cleaned_output = "\n".join(matches[:5]) if matches else "ERROR: No valid response"
        progress_bar.update(1)
        return f"INDEX {idx}:\n{cleaned_output}"
    except Exception as e:
        progress_bar.update(1)
        return f"INDEX {idx}: ERROR - {str(e)}"

async def generate_response_pairwise(session, prompt, idx, progress_bar):
    time.sleep(1)
    try:
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}"},
            json={
                "model": "gpt-3.5-turbo-0125",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.1,
                "max_tokens": 50
            }
        )
        response_json = await response.json()
        generated_text = response_json["choices"][0]["message"]["content"].strip()
        print(f"🔍 [DEBUG] INDEX {idx} - Raw Response ({len(generated_text)} chars): {repr(generated_text)}")
        progress_bar.update(1)
        return f"INDEX {idx}:\n{generated_text}"
    except Exception as e:
        progress_bar.update(1)
        return f"INDEX {idx}: ERROR - {str(e)}"

async def generate_response_pointwise(session, prompt, idx, progress_bar):
    time.sleep(1)
    try:
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}"},
            json={
                "model": "gpt-3.5-turbo-0125",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.1,
                "max_tokens": 10
            }
        )
        response_json = await response.json()
        generated_text = response_json["choices"][0]["message"]["content"].strip()
        print(f"🔍 [DEBUG] INDEX {idx} - Raw Response ({len(generated_text)} chars): {repr(generated_text)}")
        match = re.search(r"\b([1-5])\b", generated_text)
        cleaned_output = match.group(1) if match else "ERROR: No valid response"
        progress_bar.update(1)
        return f"INDEX {idx}:\n{cleaned_output}"
    except Exception as e:
        progress_bar.update(1)
        return f"INDEX {idx}: ERROR - {str(e)}"

async def process_batch(batch_res_ins, start_idx, progress_bar, mode):
    async with aiohttp.ClientSession() as session:
        if mode == "listwise":
            tasks = [generate_response_listwise(session, prompt, start_idx + i, progress_bar) for i, prompt in enumerate(batch_res_ins)]
        elif mode == "pairwise":
            tasks = [generate_response_pairwise(session, prompt, start_idx + i, progress_bar) for i, prompt in enumerate(batch_res_ins)]
        elif mode == "pointwise":
            tasks = [generate_response_pointwise(session, prompt, start_idx + i, progress_bar) for i, prompt in enumerate(batch_res_ins)]
        else:
            raise ValueError(f"Unknown training_type: {mode}")
        return await asyncio.gather(*tasks)

def run_inference(data_path, start, end, batch_size, mode):
    res_ins = []
    with open(data_path, "r", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            res_ins.append(item["inst"])
    res_ins = res_ins[start:end]
    print(f"📌 Loaded {len(res_ins)} items for inference.")
    with tqdm(total=len(res_ins), desc="🚀 Processing", unit="query") as progress_bar:
        batch_res_ins = batch_data(res_ins, batch_size=batch_size)
        result = []
        for batch_idx, batch in enumerate(batch_res_ins):
            print(f"🟢 Processing batch {batch_idx + 1}/{len(batch_res_ins)}...")
            responses = asyncio.run(process_batch(batch, batch_idx * batch_size, progress_bar, mode))
            result.extend(responses)
    output_file = "./res_completions_gpt35.txt"
    with open(output_file, 'w', encoding="utf-8") as file:
        file.write("\n".join(result))
    print(f"✅ 推理完成！结果已保存到 {output_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="数据路径")
    parser.add_argument("--start", type=int, default=0, help="开始索引")
    parser.add_argument("--end", type=int, default=MAX_INT, help="结束索引")
    parser.add_argument("--batch_size", type=int, default=10, help="批量大小")
    parser.add_argument("--training_type", type=str, required=True, choices=["listwise", "pairwise", "pointwise"], help="推理类型")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API Key，可选，优先级高于环境变量")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 优先使用命令行参数，否则用环境变量
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请通过 --openai_api_key 或环境变量 OPENAI_API_KEY 提供 OpenAI API Key！")
    openai.api_key = api_key
    run_inference(data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, mode=args.training_type)
