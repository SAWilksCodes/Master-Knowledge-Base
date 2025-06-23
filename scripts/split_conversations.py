import sys
import io

# Fix Unicode issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json
import os
import argparse
from datetime import datetime
from pathlib import Path

def sanitize_filename(title, max_length=60):
    """
    Sanitize a string to be safe for use as a filename.
    """
    # Replace problematic characters
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    # Remove multiple underscores and trim
    safe_title = "_".join(safe_title.split())
    return safe_title[:max_length]

def get_conversation_metadata(convo):
    """
    Extract metadata from a conversation object.
    """
    # Get title
    title = convo.get('title', 'Untitled_Conversation')

    # Try to get timestamp from various possible locations
    timestamp = None
    for key in ['create_time', 'created_at', 'timestamp']:
        if key in convo:
            timestamp = convo[key]
            break

    # If we have a timestamp, try to parse it
    date_prefix = ""
    if timestamp:
        try:
            # Handle both Unix timestamps and ISO format
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            date_prefix = dt.strftime("%Y-%m-%d_")
        except:
            pass

    return title, date_prefix

def extract_text_content(content):
    """Extract text from various content formats."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
        return '\n'.join(text_parts)

    if isinstance(content, dict) and 'text' in content:
        return content['text']

    return str(content)

def reconstruct_from_mapping(mapping):
    """Reconstruct linear message flow from mapping structure."""
    messages = []

    # Find root
    root_id = None
    for node_id, node in mapping.items():
        if node.get('parent') is None:
            root_id = node_id
            break

    if not root_id:
        return messages

    # Traverse tree
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id, {})

        if 'message' in node and node['message']:
            messages.append(node['message'])

        children = node.get('children', [])
        current_id = children[0] if children else None

    return messages

def convert_to_markdown(convo):
    """
    Convert a conversation JSON to markdown format.
    """
    lines = []

    # Add title
    title = convo.get('title', 'Untitled Conversation')
    lines.append(f"# {title}\n")

    # Add metadata if available
    if 'create_time' in convo or 'created_at' in convo:
        timestamp = convo.get('create_time', convo.get('created_at'))
        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            lines.append(f"**Date:** {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except:
            pass

    lines.append("\n---\n")

    # Extract messages
    messages = []

    # Handle different conversation structures
    if 'messages' in convo:
        messages = convo['messages']
    elif 'mapping' in convo:
        # Reconstruct from mapping
        messages = reconstruct_from_mapping(convo['mapping'])

    # Format messages
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get('role', '').title()
        content = msg.get('content', '')

        # Extract text from content
        text = extract_text_content(content)

        if text and text.strip():
            lines.append(f"\n## {role}\n")
            lines.append(f"{text}\n")

    return '\n'.join(lines)

def split_conversations(input_file, output_dir, output_format='json', verbose=False):
    """
    Splits a large OpenAI conversations.json export into individual files.
    Can output as JSON or Markdown format.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Reading {input_file}...")

    # Open and parse the JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return

    # Handle different possible structures
    conversations = []

    # Check if it's a list directly
    if isinstance(data, list):
        conversations = data
        if verbose:
            print("✓ Found conversations as root list")

    # Check for common keys that might contain conversations
    elif isinstance(data, dict):
        for key in ['conversations', 'chats', 'messages']:
            if key in data:
                conversations = data[key]
                if verbose:
                    print(f"✓ Found conversations under '{key}' key")
                break

        # If still empty, maybe the dict itself is keyed by conversation IDs
        if not conversations and all(isinstance(v, dict) for v in data.values()):
            conversations = list(data.values())
            if verbose:
                print("✓ Found conversations as dictionary values")

    if not conversations:
        print("❌ Could not find conversations in the JSON structure")
        print("   Keys found:", list(data.keys()) if isinstance(data, dict) else "N/A (list)")
        return

    print(f"📊 Found {len(conversations)} conversations to split")

    # Process each conversation
    successful = 0
    failed = 0

    for idx, convo in enumerate(conversations):
        try:
            # Skip if not a proper conversation object
            if not isinstance(convo, dict):
                if verbose:
                    print(f"⚠️  Skipping item {idx}: not a dictionary")
                failed += 1
                continue

            # Get metadata
            title, date_prefix = get_conversation_metadata(convo)
            safe_title = sanitize_filename(title)

            # Create filename with index for guaranteed uniqueness
            extension = 'md' if output_format == 'markdown' else 'json'
            filename = f"{idx:04d}_{date_prefix}{safe_title}.{extension}"
            filepath = output_path / filename

            # Write the conversation
            with open(filepath, 'w', encoding='utf-8') as out_f:
                if output_format == 'markdown':
                    markdown_content = convert_to_markdown(convo)
                    out_f.write(markdown_content)
                else:
                    json.dump(convo, out_f, ensure_ascii=False, indent=2)

            successful += 1

            if verbose:
                print(f"✓ Saved: {filename}")
            elif successful % 100 == 0:
                print(f"   Progress: {successful}/{len(conversations)} conversations saved...")

        except Exception as e:
            failed += 1
            if verbose:
                print(f"❌ Error processing conversation {idx}: {str(e)}")

    # Summary
    print(f"\n✅ Split complete!")
    print(f"   📁 Output directory: {output_path.absolute()}")
    print(f"   ✓ Successfully saved: {successful} conversations")
    if failed > 0:
        print(f"   ⚠️  Failed to process: {failed} items")

def main():
    parser = argparse.ArgumentParser(
        description="Split an OpenAI conversations.json export into individual JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_conversations.py -i conversations.json -o ./split_chats
  python split_conversations.py -i ~/Downloads/conversations.json -o ./AI_Chat_Exports/Raw -v
  python split_conversations.py -i conversations.json -o ./split_markdown -f markdown
        """
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to conversations.json file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Directory where split files will be saved"
    )
    parser.add_argument(
        "-f", "--format",
        choices=['json', 'markdown'],
        default='json',
        help="Output format for split files (default: json)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress for each conversation"
    )

    args = parser.parse_args()

    # Run the splitter
    split_conversations(args.input, args.output_dir, args.format, args.verbose)

if __name__ == "__main__":
    main()
