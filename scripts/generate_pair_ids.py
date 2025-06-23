#!/usr/bin/env python3
"""
Generate Pair IDs
Creates unique IDs for each user/assistant exchange pair.
Builds on conversation IDs to create hierarchical relationships.
"""

import json
import csv
import os
from pathlib import Path
import argparse
from datetime import datetime
import hashlib

def extract_message_pairs(conversation_data):
    """Extract user/assistant message pairs from conversation."""
    pairs = []

    if 'mapping' in conversation_data:
        # Build message tree
        messages = {}
        for node_id, node in conversation_data['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']
                if 'author' in msg and 'content' in msg:
                    messages[node_id] = {
                        'id': node_id,
                        'author': msg['author'].get('role', 'unknown'),
                        'content': msg['content'],
                        'create_time': msg.get('create_time', 0),
                        'parent': node.get('parent'),
                        'children': node.get('children', [])
                    }

        # Find user/assistant pairs
        for msg_id, msg in messages.items():
            if msg['author'] == 'user':
                # Look for assistant response
                for child_id in msg.get('children', []):
                    if child_id in messages and messages[child_id]['author'] == 'assistant':
                        pairs.append({
                            'user_message_id': msg_id,
                            'assistant_message_id': child_id,
                            'user_content': msg['content'],
                            'assistant_content': messages[child_id]['content'],
                            'timestamp': msg['create_time'],
                            'parent_message_id': msg.get('parent')
                        })

    return pairs

def generate_pair_id(conversation_id, pair_index, user_content, assistant_content):
    """Generate unique pair ID."""
    # Create hash from content
    content_hash = hashlib.md5(
        f"{user_content[:100]}_{assistant_content[:100]}".encode()
    ).hexdigest()[:6]

    # Format: PAIR_CONVID_INDEX_HASH
    return f"PAIR_{conversation_id}_{pair_index:04d}_{content_hash}"

def process_conversations(input_dir, conversation_ids_file, output_file):
    """Process conversations and generate pair IDs."""
    input_path = Path(input_dir)

    # Load conversation IDs
    conversation_ids = {}
    with open(conversation_ids_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conversation_ids[row['filename']] = row['conversation_id']

    json_files = list(input_path.glob('*.json'))
    print(f"🔍 Processing {len(json_files)} conversations for pair ID generation...")

    all_pairs = []
    pair_counter = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)

            conversation_id = conversation_ids.get(json_file.name, f"CONV_{json_file.stem}")

            # Extract pairs
            pairs = extract_message_pairs(conversation_data)

            for i, pair in enumerate(pairs):
                # Generate pair ID
                pair_id = generate_pair_id(conversation_id, i,
                                         str(pair['user_content']),
                                         str(pair['assistant_content']))

                # Extract text content
                user_text = ""
                assistant_text = ""

                if isinstance(pair['user_content'], dict) and 'parts' in pair['user_content']:
                    user_text = ' '.join([str(part) for part in pair['user_content']['parts'] if isinstance(part, str)])

                if isinstance(pair['assistant_content'], dict) and 'parts' in pair['assistant_content']:
                    assistant_text = ' '.join([str(part) for part in pair['assistant_content']['parts'] if isinstance(part, str)])

                # Format timestamp
                timestamp = pair['timestamp']
                date_str = 'Unknown'
                if timestamp:
                    try:
                        dt = datetime.fromtimestamp(timestamp)
                        date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass

                all_pairs.append({
                    'pair_id': pair_id,
                    'conversation_id': conversation_id,
                    'filename': json_file.name,
                    'pair_index': i,
                    'user_message_id': pair['user_message_id'],
                    'assistant_message_id': pair['assistant_message_id'],
                    'user_text': user_text[:200] + '...' if len(user_text) > 200 else user_text,
                    'assistant_text': assistant_text[:200] + '...' if len(assistant_text) > 200 else assistant_text,
                    'timestamp': timestamp,
                    'date': date_str,
                    'parent_message_id': pair['parent_message_id']
                })

                pair_counter += 1

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Sort by timestamp
    all_pairs.sort(key=lambda x: x['timestamp'], reverse=True)

    # Write CSV
    output_path = Path(output_file)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['pair_id', 'conversation_id', 'filename', 'pair_index',
                     'user_message_id', 'assistant_message_id', 'user_text',
                     'assistant_text', 'timestamp', 'date', 'parent_message_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_pairs)

    # Create summary
    summary_path = output_path.with_suffix('.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Pair ID Generation Summary\n\n")
        f.write(f"Total pairs generated: {len(all_pairs)}\n")
        f.write(f"Total conversations processed: {len(json_files)}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `PAIR_CONVID_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- INDEX: Pair index within conversation\n")
        f.write("- HASH: 6-character hash of user/assistant content\n\n")

        f.write("## Recent Pairs\n\n")
        for pair in all_pairs[:10]:
            f.write(f"- **{pair['pair_id']}**: {pair['user_text'][:50]}... ({pair['date']})\n")

    print(f"\n✅ Pair ID generation complete!")
    print(f"📄 CSV file: {output_path.absolute()}")
    print(f"📄 Summary: {summary_path.absolute()}")
    print(f"📊 Generated {len(all_pairs)} pair IDs")

    return all_pairs

def main():
    parser = argparse.ArgumentParser(description="Generate unique pair IDs for user/assistant exchanges")
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-c", "--conversation-ids", required=True, help="CSV file with conversation IDs")
    parser.add_argument("-o", "--output", default="pair_ids.csv", help="Output CSV file")

    args = parser.parse_args()
    process_conversations(args.input_dir, args.conversation_ids, args.output)

if __name__ == "__main__":
    main()
