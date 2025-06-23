import sys
import io

# Fix Unicode issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json
import os
import argparse
from pathlib import Path
from datetime import datetime

def extract_text_content(content):
    """
    Extract text from different content formats in OpenAI exports.
    Handles both old and new message formats.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        # Handle content with parts structure
        if 'parts' in content:
            parts = content['parts']
            if isinstance(parts, list):
                text_parts = []
                for part in parts:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                return '\n'.join(text_parts)
        # Handle content with text key
        elif 'text' in content:
            return content['text']

    if isinstance(content, list):
        # New format: content is a list of parts
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get('type') == 'text' and 'text' in part:
                    text_parts.append(part['text'])
                elif 'text' in part:
                    text_parts.append(part['text'])
        return '\n'.join(text_parts)

    return str(content)

def extract_message_pairs(conversation):
    """
    Extract user/assistant message pairs from a conversation.
    Returns a list of dictionaries with 'user' and 'assistant' keys.
    """
    pairs = []

    # Find the messages in the conversation structure
    messages = []

    # Check common locations for messages
    if 'messages' in conversation:
        messages = conversation['messages']
    elif 'mapping' in conversation:
        # Handle the mapping structure (common in newer exports)
        mapping = conversation['mapping']
        # Reconstruct linear message flow from the mapping
        messages = reconstruct_from_mapping(mapping)
    elif isinstance(conversation, list):
        messages = conversation

    # Process messages into pairs
    current_user_message = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        # Get role and content
        role = msg.get('role', '').lower()
        content = msg.get('content', '')

        # Skip system messages
        if role == 'system':
            continue

        # Extract text content
        text = extract_text_content(content)

        # Skip empty messages
        if not text or text.strip() == '':
            continue

        # Handle user messages
        if role == 'user':
            if current_user_message and not pairs:
                # Edge case: conversation starts with multiple user messages
                pairs.append({
                    'user': current_user_message,
                    'assistant': '[No response]'
                })
            current_user_message = text

        # Handle assistant messages
        elif role in ['assistant', 'model', 'chatgpt']:
            if current_user_message:
                pairs.append({
                    'user': current_user_message,
                    'assistant': text
                })
                current_user_message = None
            else:
                # Assistant message without user message
                pairs.append({
                    'user': '[Previous context]',
                    'assistant': text
                })

    # Handle trailing user message
    if current_user_message:
        pairs.append({
            'user': current_user_message,
            'assistant': '[No response]'
        })

    return pairs

def reconstruct_from_mapping(mapping):
    """
    Reconstruct linear message flow from OpenAI's mapping structure.
    """
    messages = []

    # Find root node
    root_id = None
    for node_id, node in mapping.items():
        if node.get('parent') is None:
            root_id = node_id
            break

    if not root_id:
        return messages

    # Traverse the tree
    current_id = root_id
    visited = set()

    while current_id:
        if current_id in visited:
            break
        visited.add(current_id)

        node = mapping.get(current_id, {})

        # Extract message if present
        if 'message' in node and node['message']:
            msg = node['message']

            # Handle new structure where role is in author.role
            if 'author' in msg and isinstance(msg.get('author'), dict):
                role = msg['author'].get('role', '')
                # Create a normalized message structure
                normalized_msg = {
                    'role': role,
                    'content': msg.get('content', '')
                }
                messages.append(normalized_msg)
            elif msg.get('content'):
                # Fallback for older format
                messages.append(msg)

        # Move to next node (first child)
        children = node.get('children', [])
        current_id = children[0] if children else None

    return messages

def process_file(input_file, output_format='jsonl'):
    """
    Process a single conversation file and extract message pairs.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        conversation = json.load(f)

    pairs = extract_message_pairs(conversation)

    if output_format == 'jsonl':
        # Return pairs as JSONL strings
        return [json.dumps(pair, ensure_ascii=False) for pair in pairs]
    elif output_format == 'markdown':
        # Return pairs as markdown
        lines = []
        for i, pair in enumerate(pairs, 1):
            lines.append(f"## Exchange {i}\n")
            lines.append(f"**User:** {pair['user']}\n")
            lines.append(f"**Assistant:** {pair['assistant']}\n")
        return lines
    else:
        # Return raw pairs
        return pairs

def process_directory(input_dir, output_file, output_format='jsonl', verbose=False):
    """
    Process all JSON files in a directory and combine into single output.
    """
    input_path = Path(input_dir)
    json_files = list(input_path.glob('*.json'))

    print(f"📂 Found {len(json_files)} conversation files to process")

    all_pairs = []
    successful = 0
    failed = 0

    for i, json_file in enumerate(json_files):
        try:
            if verbose:
                print(f"Processing {json_file.name}...")

            pairs = process_file(json_file, output_format='raw')

            # Add metadata to each pair
            for pair in pairs:
                pair['source_file'] = json_file.name
                pair['conversation_index'] = i

            all_pairs.extend(pairs)
            successful += 1

            if not verbose and successful % 10 == 0:
                print(f"   Progress: {successful}/{len(json_files)} files processed...")

        except Exception as e:
            failed += 1
            if verbose:
                print(f"❌ Error processing {json_file.name}: {str(e)}")

    # Write output
    print(f"\n💾 Writing {len(all_pairs)} message pairs to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        if output_format == 'jsonl':
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        elif output_format == 'json':
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)
        elif output_format == 'markdown':
            for i, pair in enumerate(all_pairs, 1):
                f.write(f"## Exchange {i}\n")
                f.write(f"**Source:** {pair['source_file']}\n")
                f.write(f"**User:** {pair['user']}\n")
                f.write(f"**Assistant:** {pair['assistant']}\n\n")

    print(f"\n✅ Processing complete!")
    print(f"   ✓ Successfully processed: {successful} files")
    print(f"   ✓ Total message pairs: {len(all_pairs)}")
    if failed > 0:
        print(f"   ⚠️  Failed to process: {failed} files")

def main():
    parser = argparse.ArgumentParser(
        description="Extract user/assistant message pairs from OpenAI conversation exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python extract_pairs.py -i conversation.json -o pairs.jsonl

  # Process all files in a directory
  python extract_pairs.py -d ./split_chats -o all_pairs.jsonl

  # Output as markdown for readability
  python extract_pairs.py -d ./split_chats -o all_pairs.md -f markdown

  # Output as regular JSON array
  python extract_pairs.py -d ./split_chats -o all_pairs.json -f json
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input-file",
        help="Process a single conversation JSON file"
    )
    input_group.add_argument(
        "-d", "--input-dir",
        help="Process all JSON files in a directory"
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "-f", "--format",
        choices=['jsonl', 'json', 'markdown'],
        default='jsonl',
        help="Output format (default: jsonl)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress"
    )

    args = parser.parse_args()

    if args.input_file:
        # Process single file
        print(f"📄 Processing {args.input_file}")
        output_lines = process_file(args.input_file, args.format)

        with open(args.output, 'w', encoding='utf-8') as f:
            if args.format == 'markdown':
                f.write('\n'.join(output_lines))
            else:
                f.write('\n'.join(output_lines))

        print(f"✅ Saved {len(output_lines)} message pairs to {args.output}")

    else:
        # Process directory
        process_directory(args.input_dir, args.output, args.format, args.verbose)

if __name__ == "__main__":
    main()
