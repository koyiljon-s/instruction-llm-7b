import json

def convert_jsonl_to_json(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data.append(json.loads(line))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Usage:
convert_jsonl_to_json('make-gpt-4o.jsonl', 'make-gpt-4o.json')
