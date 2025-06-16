import jsonlines
import re

# Input and output file paths
input_file = "pointwisetest.jsonl"
output_file = "cleanedpointwisetest.jsonl"

# Count the number of modified lines
modified_count = 0
total_count = 0

# Process JSONL file
with jsonlines.open(input_file, mode="r") as reader, jsonlines.open(output_file, mode="w") as writer:
for obj in reader:
total_count += 1
original_inst = obj["inst"]

# Use regular expressions to match the content between "Hint: Another recommender system" and "\n\nPlease only"
cleaned_inst, num_subs = re.subn(
r'Hint: Another recommender system.*?(?=\n\nPlease only)', # Delete only the content from Hint to "Please only"
'',
original_inst,
flags=re.DOTALL
)

# Count the number of modifications
if num_subs > 0:
modified_count += 1
obj["inst"] = cleaned_inst # Update content

# Write to a new JSONL file
writer.write(obj)

# Output modification status
print(f"✅ Processing completed! Total number of lines: {total_count}, number of modified lines: {modified_count}")
print(f"📁 The result has been saved to {output_file}")