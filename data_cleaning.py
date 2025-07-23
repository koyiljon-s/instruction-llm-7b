import json

def remove_duplicate_instructions(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten the list of lists into a single list of dictionaries
    flat_data = [sample for sublist in data for sample in sublist]

    seen = set()
    unique_data = []
    duplicate_count = 0  # 🔢 Counter

    for sample in flat_data:
        if not isinstance(sample, dict):
            continue  # Skip non-dictionaries
        instruction = sample.get('instruction', '').strip()
        if instruction in seen:
            duplicate_count += 1  # Increment counter when duplicate found
            continue
        seen.add(instruction)
        unique_data.append(sample)

    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=4, ensure_ascii=False)

    # Show stats
    total_samples = len(flat_data)
    unique_samples = len(unique_data)
    print(f"➡️  Total samples: {total_samples}")
    print(f"✅ Unique samples: {unique_samples}")
    print(f"❌ Duplicates removed: {duplicate_count}")

# Call the function
remove_duplicate_instructions('self-instruct-uz.json', 'self_instruct_uz.json')
