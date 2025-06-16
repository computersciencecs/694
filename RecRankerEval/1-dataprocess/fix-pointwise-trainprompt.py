import json

# Input and output file paths
input_file = "pointwise.jsonl"
output_file = "pointwise_processed.jsonl"

# Define two markers
marker_start = "Hint: Another recommender"
marker_end = "\n\nPlease only"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
for line in fin:
line = line.strip()
if not line:
continue
# Parse JSON lines
data = json.loads(line)
# Traverse all messages, mainly processing messages with role "user"
for message in data.get("messages", []):
if message.get("role") == "user":
# Assume that the message content is stored in the "content" field, and the field is a list, each element contains "type" and "content"
for part in message.get("content", []):
if part.get("type") == "text":
text = part.get("content", "")
# Check if both markers are present
if marker_start in text and marker_end in text:
start_idx = text.find(marker_start)
end_idx = text.find(marker_end)
# If both markers exist and marker_start appears before marker_end
if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
# Delete the part from marker_start to the point before marker_end appears (keep marker_end and the content after it)
new_text = text[:start_idx] + text[end_idx:]
part["content"] = new_text
# Write the modified JSON object, one per line
fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"✅ Processing completed, results saved to {output_file}")