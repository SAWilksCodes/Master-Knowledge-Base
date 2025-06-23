#!/usr/bin/env python3
"""
Generate Conversation IDs
Creates unique IDs for each conversation based on existing filenames.
This is the foundation for all other ID generation.
"""

import json
import csv
import os
from pathlib import Path
import argparse
from datetime import datetime
import hashlib

def generate_conversation_id(filename, conversation_data):
    """Generate a unique conversation ID from filename and metadata."""
    # Extract base info from filename
    base_name = Path(filename).stem

    # Get conversation metadata
    title = conversation_data.get('title', 'Untitled')
    create_time = conversation_data.get('create_time', 0)
    conversation_id = conversation_data.get('conversation_id', '')

    # Create a hash for uniqueness
    hash_input = f"{base_name}_{title}_{create_time}"
    hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    # Format: CONV_YYYYMMDD_HASH
    if create_time:
        try:
            dt = datetime.fromtimestamp(create_time)
            date_str = dt.strftime('%Y%m%d')
        except:
            date_str = '00000000'
    else:
        date_str = '00000000'

    return f"CONV_{date_str}_{hash_id}"

def process_conversations(input_dir, output_file):
    """Process all conversations and generate IDs."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob('*.json'))

    print(f"🔍 Processing {len(json_files)} conversations for ID generation...")

    conversation_ids = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)

            # Generate conversation ID
            conv_id = generate_conversation_id(json_file.name, conversation_data)

            # Extract metadata
            title = conversation_data.get('title', 'Untitled')
            create_time = conversation_data.get('create_time', 0)

            # Format date
            date_str = 'Unknown'
            if create_time:
                try:
                    dt = datetime.fromtimestamp(create_time)
                    date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass

            conversation_ids.append({
                'conversation_id': conv_id,
                'filename': json_file.name,
                'title': title,
                'create_time': create_time,
                'date': date_str,
                'original_conversation_id': conversation_data.get('conversation_id', '')
            })

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Sort by date
    conversation_ids.sort(key=lambda x: x['create_time'], reverse=True)

    # Write CSV
    output_path = Path(output_file)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['conversation_id', 'filename', 'title', 'create_time', 'date', 'original_conversation_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(conversation_ids)

    # Create summary
    summary_path = output_path.with_suffix('.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Conversation ID Generation Summary\n\n")
        f.write(f"Total conversations processed: {len(conversation_ids)}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `CONV_YYYYMMDD_HASH`\n")
        f.write("- YYYYMMDD: Date of conversation\n")
        f.write("- HASH: 8-character MD5 hash of filename + title + timestamp\n\n")

        f.write("## Recent Conversations\n\n")
        for conv in conversation_ids[:10]:
            f.write(f"- **{conv['conversation_id']}**: {conv['title']} ({conv['date']})\n")

    print(f"\n✅ Conversation ID generation complete!")
    print(f"📄 CSV file: {output_path.absolute()}")
    print(f"📄 Summary: {summary_path.absolute()}")
    print(f"📊 Generated {len(conversation_ids)} conversation IDs")

    return conversation_ids

def main():
    parser = argparse.ArgumentParser(description="Generate unique conversation IDs")
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-o", "--output", default="conversation_ids.csv", help="Output CSV file")

    args = parser.parse_args()
    process_conversations(args.input_dir, args.output)

if __name__ == "__main__":
    main()
