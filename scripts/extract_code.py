import sys
import io

# Fix Unicode issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json
import os
from pathlib import Path
import argparse
import re
from datetime import datetime

def extract_code_blocks_from_text(text):
    """Extract code blocks from markdown text."""
    code_blocks = []

    # Match code blocks with optional language
    # Matches ```language\ncode\n``` or ```\ncode\n```
    pattern = r'```(?P<lang>\w+)?\n(?P<code>.*?)```'

    for match in re.finditer(pattern, text, re.DOTALL):
        language = match.group('lang') or 'plaintext'
        code = match.group('code').strip()

        if code:  # Skip empty code blocks
            code_blocks.append({
                'language': language,
                'code': code
            })

    # Also match inline code if it looks substantial (multi-line)
    inline_pattern = r'`([^`]+)`'
    for match in re.finditer(inline_pattern, text):
        code = match.group(1)
        if '\n' in code and len(code) > 50:  # Multi-line inline code
            code_blocks.append({
                'language': 'plaintext',
                'code': code
            })

    return code_blocks

def extract_code_from_conversation(convo):
    """Extract all code blocks from a conversation."""
    all_code_blocks = []
    title = convo.get('title', 'Untitled')
    convo_id = convo.get('conversation_id', 'unknown')

    # Get timestamp
    timestamp = None
    for field in ['create_time', 'created_at', 'timestamp']:
        if field in convo and convo[field]:
            timestamp = convo[field]
            break

    # Extract from mapping structure
    if 'mapping' in convo:
        for node_id, node in convo['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']

                # Get role
                role = 'unknown'
                if 'author' in msg and isinstance(msg['author'], dict):
                    role = msg['author'].get('role', 'unknown')

                # Extract text content
                if 'content' in msg:
                    content = msg['content']
                    text = ''

                    if isinstance(content, dict) and 'parts' in content:
                        for part in content['parts']:
                            if isinstance(part, str):
                                text += part + '\n'
                    elif isinstance(content, str):
                        text = content

                    # Extract code blocks
                    code_blocks = extract_code_blocks_from_text(text)

                    for block in code_blocks:
                        block['conversation_title'] = title
                        block['conversation_id'] = convo_id
                        block['role'] = role
                        block['timestamp'] = timestamp
                        all_code_blocks.append(block)

    return all_code_blocks

def save_code_blocks(code_blocks, output_file, format='markdown'):
    """Save extracted code blocks to file."""
    output_path = Path(output_file)

    if format == 'markdown':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Extracted Code Blocks\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total code blocks: {len(code_blocks)}\n\n")
            f.write("---\n\n")

            current_convo = None

            for i, block in enumerate(code_blocks, 1):
                # Add conversation header if new conversation
                if block['conversation_title'] != current_convo:
                    current_convo = block['conversation_title']
                    f.write(f"\n## 📁 {current_convo}\n\n")

                f.write(f"### Code Block {i} ({block['language']})\n")
                f.write(f"*From: {block['role']}*\n\n")
                f.write(f"```{block['language']}\n")
                f.write(block['code'])
                if not block['code'].endswith('\n'):
                    f.write('\n')
                f.write("```\n\n")

    elif format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(code_blocks, f, ensure_ascii=False, indent=2)

    elif format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for block in code_blocks:
                f.write(json.dumps(block, ensure_ascii=False) + '\n')

def extract_code_from_directory(input_dir, output_file, format='markdown', min_length=10):
    """Extract all code blocks from conversations in a directory."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob('*.json'))

    print(f"💻 Extracting code from {len(json_files)} conversations...")

    all_code_blocks = []
    stats = {
        'conversations_with_code': 0,
        'languages': {}
    }

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                convo = json.load(f)

            code_blocks = extract_code_from_conversation(convo)

            # Filter by minimum length
            code_blocks = [b for b in code_blocks if len(b['code'].strip()) >= min_length]

            if code_blocks:
                stats['conversations_with_code'] += 1
                all_code_blocks.extend(code_blocks)

                # Update language stats
                for block in code_blocks:
                    lang = block['language']
                    stats['languages'][lang] = stats['languages'].get(lang, 0) + 1

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Save code blocks
    save_code_blocks(all_code_blocks, output_file, format)

    # Print summary
    print(f"\n✅ Code extraction complete!")
    print(f"📄 Output file: {Path(output_file).absolute()}")
    print(f"\n📊 Statistics:")
    print(f"   Total code blocks: {len(all_code_blocks)}")
    print(f"   Conversations with code: {stats['conversations_with_code']}")
    print(f"\n📚 Languages found:")

    for lang in sorted(stats['languages'].keys()):
        print(f"   {lang}: {stats['languages'][lang]} blocks")

def main():
    parser = argparse.ArgumentParser(
        description="Extract all code blocks from conversations"
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("-f", "--format", choices=['markdown', 'json', 'jsonl'],
                       default='markdown', help="Output format")
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum code length to include (default: 10 chars)")

    args = parser.parse_args()
    extract_code_from_directory(args.input_dir, args.output, args.format, args.min_length)

if __name__ == "__main__":
    main()
