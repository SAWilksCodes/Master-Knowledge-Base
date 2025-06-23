import json
import os
from pathlib import Path
import argparse
import shutil
from collections import defaultdict

def extract_models_from_conversation(convo):
    """Extract all AI models used in a conversation."""
    models = set()

    # Extract from mapping structure (ChatGPT format)
    if 'mapping' in convo:
        for node_id, node in convo['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']
                # Check for model in metadata
                if 'metadata' in msg and 'model_slug' in msg['metadata']:
                    model = msg['metadata']['model_slug']
                    if model:
                        models.add(model)
                # Check for author role
                if 'author' in msg and 'role' in msg['author']:
                    if msg['author']['role'] == 'assistant':
                        # Try to get model from metadata
                        if 'metadata' in msg and 'model_slug' in msg['metadata']:
                            model = msg['metadata']['model_slug']
                            if model:
                                models.add(model)

    # Extract from messages array (alternative format)
    elif 'messages' in convo:
        for msg in convo['messages']:
            # Check for model field
            if 'model' in msg and msg['model']:
                models.add(msg['model'])
            # Check metadata
            if 'metadata' in msg and 'model' in msg['metadata']:
                models.add(msg['metadata']['model'])
            # Check for model_slug
            if 'metadata' in msg and 'model_slug' in msg['metadata']:
                models.add(msg['metadata']['model_slug'])

    # Check conversation-level metadata
    if 'model' in convo:
        models.add(convo['model'])
    if 'model_slug' in convo:
        models.add(convo['model_slug'])

    # Clean up model names
    cleaned_models = set()
    for model in models:
        if model:
            # Normalize model names
            model_lower = model.lower()
            if 'gpt-4' in model_lower:
                if 'turbo' in model_lower:
                    cleaned_models.add('gpt-4-turbo')
                elif 'o' in model_lower or '4o' in model_lower:
                    cleaned_models.add('gpt-4o')
                else:
                    cleaned_models.add('gpt-4')
            elif 'gpt-3.5' in model_lower:
                cleaned_models.add('gpt-3.5-turbo')
            elif 'claude' in model_lower:
                if '3' in model_lower:
                    if 'opus' in model_lower:
                        cleaned_models.add('claude-3-opus')
                    elif 'sonnet' in model_lower:
                        cleaned_models.add('claude-3-sonnet')
                    elif 'haiku' in model_lower:
                        cleaned_models.add('claude-3-haiku')
                    else:
                        cleaned_models.add('claude-3')
                elif '2' in model_lower:
                    cleaned_models.add('claude-2')
                else:
                    cleaned_models.add('claude')
            elif 'o1' in model_lower:
                if 'mini' in model_lower:
                    cleaned_models.add('o1-mini')
                elif 'preview' in model_lower:
                    cleaned_models.add('o1-preview')
                else:
                    cleaned_models.add('o1')
            elif 'text-' in model_lower and 'davinci' in model_lower:
                cleaned_models.add('text-davinci')
            else:
                # Keep original if no pattern matches
                cleaned_models.add(model)

    return list(cleaned_models) if cleaned_models else ['unknown_model']

def split_by_model(input_dir, output_dir, copy_files=True):
    """Split conversations by AI model used."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Clear output directory if it exists
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob('*.json'))
    print(f"🤖 Processing {len(json_files)} conversations by model...")

    stats = defaultdict(int)
    multi_model_count = 0
    processed_count = 0
    no_model_count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                convo = json.load(f)

            models = extract_models_from_conversation(convo)

            if 'unknown_model' in models:
                no_model_count += 1

            # Handle multi-model conversations
            if len(models) > 1:
                multi_model_count += 1

            for model in models:
                # Create model folder
                model_folder = output_path / model
                model_folder.mkdir(parents=True, exist_ok=True)

                # Update stats
                stats[model] += 1

                # Copy or move file
                dest_file = model_folder / json_file.name
                if copy_files or len(models) > 1:
                    shutil.copy2(json_file, dest_file)
                else:
                    shutil.move(str(json_file), str(dest_file))
                    break  # Only move once if not copying

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"   Processed {processed_count}/{len(json_files)} files...")

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Print summary
    print(f"\n✅ Model-based split complete!")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"\n📊 Conversations by model:")

    # Sort by count (descending)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)

    for model, count in sorted_stats:
        print(f"   {model}: {count} conversations")

    if multi_model_count > 0:
        print(f"\n🔄 {multi_model_count} conversations used multiple models")

    if no_model_count > 0:
        print(f"\n❓ {no_model_count} conversations had no identifiable model")

    total_categorized = sum(stats.values())
    print(f"\n📈 Total categorizations: {total_categorized}")
    print(f"📄 Total files processed: {processed_count}")

def main():
    parser = argparse.ArgumentParser(
        description="Split conversations by AI model used"
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for organized files")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")

    args = parser.parse_args()
    split_by_model(args.input_dir, args.output_dir, copy_files=not args.move)

if __name__ == "__main__":
    main()
