#!/usr/bin/env python3
"""
Generate Code Block IDs
Creates unique IDs for each code block found in conversations.
You have 14,611 code blocks - this will create a comprehensive code index.
"""

import json
import csv
import os
from pathlib import Path
import argparse
import hashlib
import re
from datetime import datetime

def extract_code_blocks(conversation_data):
    """Extract all code blocks from conversation."""
    code_blocks = []

    if 'mapping' in conversation_data:
        for node_id, node in conversation_data['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']
                if 'content' in msg:
                    content = msg['content']
                    if isinstance(content, dict) and 'parts' in content:
                        for part in content['parts']:
                            if isinstance(part, str):
                                # Look for code blocks (```language ... ```)
                                code_pattern = r'```(\w+)?\n(.*?)```'
                                matches = re.findall(code_pattern, part, re.DOTALL)

                                for lang, code in matches:
                                    code_blocks.append({
                                        'message_id': node_id,
                                        'language': lang.strip() if lang else 'plaintext',
                                        'code': code.strip(),
                                        'author': msg.get('author', {}).get('role', 'unknown'),
                                        'create_time': msg.get('create_time', 0)
                                    })

    return code_blocks

def generate_code_block_id(conversation_id, block_index, language, code_hash):
    """Generate unique code block ID."""
    # Format: CODE_CONVID_INDEX_LANG_HASH
    return f"CODE_{conversation_id}_{block_index:04d}_{language.upper()}_{code_hash}"

def analyze_code_complexity(code, language):
    """Analyze code complexity and characteristics."""
    lines = code.split('\n')
    line_count = len(lines)

    # Count different elements
    function_count = len(re.findall(r'\b(def|function|fn)\b', code, re.IGNORECASE))
    class_count = len(re.findall(r'\bclass\b', code, re.IGNORECASE))
    import_count = len(re.findall(r'\b(import|from|require|include)\b', code, re.IGNORECASE))
    comment_count = len(re.findall(r'#|//|/\*|\*/|<!--|-->', code))

    # Estimate complexity
    complexity = 'simple'
    if line_count > 50 or function_count > 5 or class_count > 2:
        complexity = 'complex'
    elif line_count > 20 or function_count > 2:
        complexity = 'medium'

    return {
        'line_count': line_count,
        'function_count': function_count,
        'class_count': class_count,
        'import_count': import_count,
        'comment_count': comment_count,
        'complexity': complexity
    }

def process_conversations(input_dir, conversation_ids_file, output_file):
    """Process conversations and generate code block IDs."""
    input_path = Path(input_dir)

    # Load conversation IDs
    conversation_ids = {}
    with open(conversation_ids_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conversation_ids[row['filename']] = row['conversation_id']

    json_files = list(input_path.glob('*.json'))
    print(f"🔍 Processing {len(json_files)} conversations for code block ID generation...")

    all_code_blocks = []
    language_stats = {}
    total_blocks = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)

            conversation_id = conversation_ids.get(json_file.name, f"CONV_{json_file.stem}")
            title = conversation_data.get('title', 'Untitled')

            # Extract code blocks
            code_blocks = extract_code_blocks(conversation_data)

            for i, block in enumerate(code_blocks):
                # Generate code block ID
                code_hash = hashlib.md5(block['code'].encode()).hexdigest()[:6]
                code_block_id = generate_code_block_id(conversation_id, i, block['language'], code_hash)

                # Analyze code complexity
                analysis = analyze_code_complexity(block['code'], block['language'])

                # Format timestamp
                timestamp = block['create_time']
                date_str = 'Unknown'
                if timestamp:
                    try:
                        dt = datetime.fromtimestamp(timestamp)
                        date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass

                # Update language stats
                lang = block['language']
                if lang not in language_stats:
                    language_stats[lang] = 0
                language_stats[lang] += 1

                all_code_blocks.append({
                    'code_block_id': code_block_id,
                    'conversation_id': conversation_id,
                    'filename': json_file.name,
                    'title': title,
                    'block_index': i,
                    'message_id': block['message_id'],
                    'language': lang,
                    'line_count': analysis['line_count'],
                    'function_count': analysis['function_count'],
                    'class_count': analysis['class_count'],
                    'import_count': analysis['import_count'],
                    'comment_count': analysis['comment_count'],
                    'complexity': analysis['complexity'],
                    'author': block['author'],
                    'timestamp': timestamp,
                    'date': date_str,
                    'code_preview': block['code'][:200] + '...' if len(block['code']) > 200 else block['code']
                })

                total_blocks += 1

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Sort by timestamp
    all_code_blocks.sort(key=lambda x: x['timestamp'], reverse=True)

    # Write CSV
    output_path = Path(output_file)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['code_block_id', 'conversation_id', 'filename', 'title', 'block_index',
                     'message_id', 'language', 'line_count', 'function_count', 'class_count',
                     'import_count', 'comment_count', 'complexity', 'author', 'timestamp',
                     'date', 'code_preview']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_code_blocks)

    # Create summary
    summary_path = output_path.with_suffix('.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Code Block ID Generation Summary\n\n")
        f.write(f"Total code blocks generated: {len(all_code_blocks)}\n")
        f.write(f"Total conversations processed: {len(json_files)}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `CODE_CONVID_INDEX_LANG_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- INDEX: Code block index within conversation\n")
        f.write("- LANG: Programming language\n")
        f.write("- HASH: 6-character hash of code content\n\n")

        f.write("## Language Distribution\n\n")
        for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_blocks) * 100
            f.write(f"- **{lang}**: {count} blocks ({percentage:.1f}%)\n")

        f.write("\n## Complexity Distribution\n\n")
        complexity_stats = {}
        for block in all_code_blocks:
            comp = block['complexity']
            complexity_stats[comp] = complexity_stats.get(comp, 0) + 1

        for comp, count in sorted(complexity_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_blocks) * 100
            f.write(f"- **{comp}**: {count} blocks ({percentage:.1f}%)\n")

        f.write("\n## Recent Code Blocks\n\n")
        for block in all_code_blocks[:10]:
            f.write(f"- **{block['code_block_id']}**: {block['language']} ({block['line_count']} lines) - {block['title']} ({block['date']})\n")

    print(f"\n✅ Code Block ID generation complete!")
    print(f"📄 CSV file: {output_path.absolute()}")
    print(f"📄 Summary: {summary_path.absolute()}")
    print(f"📊 Generated {len(all_code_blocks)} code block IDs")
    print(f"📊 Languages found: {len(language_stats)}")

    return all_code_blocks

def main():
    parser = argparse.ArgumentParser(description="Generate unique code block IDs")
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-c", "--conversation-ids", required=True, help="CSV file with conversation IDs")
    parser.add_argument("-o", "--output", default="code_block_ids.csv", help="Output CSV file")

    args = parser.parse_args()
    process_conversations(args.input_dir, args.conversation_ids, args.output)

if __name__ == "__main__":
    main()
