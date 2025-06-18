#This file is used to merge the prompts of the forward and reverse pairwise methods for subsequent inference.
import json


file1 = "pairwisetest1.jsonl"
file2 = "pairwise_invtest2.jsonl"

output_file = "pairwisetest.jsonl"


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))  
    return data


def write_jsonl(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


data1 = read_jsonl(file1)
data2 = read_jsonl(file2)


merged_data = data1 + data2


write_jsonl(output_file, merged_data)

print(f"Merge completed, output file: {output_file}")
