

import os
import json
import time
import re
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from openai import OpenAI
# === Load API Key ===
client = OpenAI( 
  api_key=""
)

# === Config ===
model = "gpt-4o"
num_samples = 2000
batch_size = 5
output_file = "make-gpt-4o.jsonl"
max_retries = 3

# === Uzbek Prompt ===
with open("prompt.txt", "r", encoding="utf-8") as f:
    instruction_prompt = f.read()


# === Utility: Escape newlines in JSON strings ===
def escape_newlines_in_json_strings(text):
    # Replace newlines inside double quotes with \\n
    def replacer(match):
        s = match.group(0)
        return s.replace('\n', '\\n').replace('\r', '\\r')
    return re.sub(r'\"(.*?)\"', replacer, text, flags=re.DOTALL)

# === Robust JSON Array Extractor ===
def extract_and_clean_json_array(text):
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        array_text = match.group(1)
        cleaned = escape_newlines_in_json_strings(array_text)
        try:
            return json.loads(cleaned)
        except Exception as e:
            print("⚠️ JSON extraction error after cleaning:", e)
    else:
        print("⚠️ No JSON array found in the output.")
    return []

# === Generation Loop ===
all_examples = 0
with open(output_file, "w", encoding="utf-8") as fout:
    for batch_num in tqdm(range(num_samples // batch_size)):
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Siz o‘zbek tilida so‘zlovchi model yordamchisiz."},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0.9,
                    max_tokens=2048,
                    top_p=0.95
                )

                content = response.choices[0].message.content.strip()
                print(f"\n--- Batch {batch_num+1} Raw Output ---\n{content}\n-----------------\n")  # Debug print

                samples = extract_and_clean_json_array(content)
                print(f"Parsed {len(samples)} samples.")  # Debug print

                for sample in samples:
                    if all(k in sample for k in ["instruction", "input", "output"]):
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        all_examples += 1
                    else:
                        print("⚠️ Skipped incomplete sample:", sample)  # Debug print

                time.sleep(1)
                break  # Success, move to next batch
            except Exception as e:
                print(f"⚠️ Error in batch {batch_num+1}, attempt {attempt+1}: {e}")
                time.sleep(5)
        else:
            print(f"❌ Failed to process batch {batch_num+1} after {max_retries} attempts.")

print(f"✅ Done. Total examples generated: {all_examples}")
