# This file is used to remove the hint in the training prompt of the pointwise method.
import jsonlines
import re

input_file = "pointwisetest.jsonl"
output_file = "pointwisetestfix.jsonl"

modified_count = 0
total_count = 0


with jsonlines.open(input_file, mode="r") as reader, jsonlines.open(output_file, mode="w") as writer:
    for obj in reader:
        total_count += 1
        original_inst = obj["inst"]


        cleaned_inst, num_subs = re.subn(
            r'Hint: Another recommender system.*?(?=\n\nPlease only)',  
            '',
            original_inst,
            flags=re.DOTALL
        )

        
        if num_subs > 0:
            modified_count += 1
            obj["inst"] = cleaned_inst  

        
        writer.write(obj)


# Output modification status
print(f"✅ Processing completed! Total number of lines: {total_count}, number of modified lines: {modified_count}")
print(f"📁 The result has been saved to {output_file}")
