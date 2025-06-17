# This file is used to remove the hint in the training prompt of the pointwise method.
import json


input_file = "pointwise.jsonl"
output_file = "pointwise_processed.jsonl"


marker_start = "Hint: Another recommender"
marker_end = "\n\nPlease only"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        
        data = json.loads(line)
        
        for message in data.get("messages", []):
            if message.get("role") == "user":
                
                for part in message.get("content", []):
                    if part.get("type") == "text":
                        text = part.get("content", "")
                        
                        if marker_start in text and marker_end in text:
                            start_idx = text.find(marker_start)
                            end_idx = text.find(marker_end)
                            
                            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                                
                                new_text = text[:start_idx] + text[end_idx:]
                                part["content"] = new_text
        
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"✅ Processing completed, results saved to {output_file}")
