import sys
import io

# Fix Unicode issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json
import os
from pathlib import Path
import argparse
import shutil

def count_messages(convo):
    """Count the number of messages in a conversation."""
    count = 0

    # Count from mapping structure
    if 'mapping' in convo:
        for node_id, node in convo['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']
                # Skip system messages
                if 'author' in msg and isinstance(msg['author'], dict):
                    role = msg['author'].get('role', '')
                    if role != 'system':
                        count += 1

    # Count from messages array
    elif 'messages' in convo:
        for msg in convo['messages']:
            if msg.get('role') != 'system':
                count += 1

    return count

def get_conversation_length_stats(convo):
    """Get detailed statistics about conversation length."""
    stats = {
        'total_messages': count_messages(convo),
        'user_messages': 0,
        'assistant_messages': 0,
        'total_chars': 0,
        'code_blocks': 0
    }

    # Analyze mapping structure
    if 'mapping' in convo:
        for node_id, node in convo['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']

                # Get role
                if 'author' in msg and isinstance(msg['author'], dict):
                    role = msg['author'].get('role', '')
                    if role == 'user':
                        stats['user_messages'] += 1
                    elif role == 'assistant':
                        stats['assistant_messages'] += 1

                # Count content length
                if 'content' in msg:
                    content = msg['content']
                    if isinstance(content, dict) and 'parts' in content:
                        for part in content['parts']:
                            if isinstance(part, str):
                                stats['total_chars'] += len(part)
                                # Count code blocks
                                stats['code_blocks'] += part.count('```')

    return stats

def extract_long_conversations(input_dir, output_dir, min_messages=20, copy_files=True):
    """Extract conversations with at least min_messages exchanges."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob('*.json'))
    print(f"📚 Analyzing {len(json_files)} conversations for length...")

    long_convos = []
    length_distribution = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                convo = json.load(f)

            stats = get_conversation_length_stats(convo)
            message_count = stats['total_messages']

            # Track distribution
            bucket = (message_count // 10) * 10  # Round to nearest 10
            length_distribution[bucket] = length_distribution.get(bucket, 0) + 1

            # Check if it's a long conversation
            if message_count >= min_messages:
                long_convos.append({
                    'file': json_file,
                    'title': convo.get('title', 'Untitled'),
                    'stats': stats
                })

                # Copy or move file
                dest_file = output_path / json_file.name
                if copy_files:
                    shutil.copy2(json_file, dest_file)
                else:
                    shutil.move(str(json_file), str(dest_file))

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Sort by length
    long_convos.sort(key=lambda x: x['stats']['total_messages'], reverse=True)

    # Create summary report
    report_path = output_path / 'long_conversations_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Long Conversations Report\n\n")
        f.write(f"Conversations with {min_messages}+ messages: {len(long_convos)}\n\n")

        f.write("## Top 10 Longest Conversations\n\n")
        for i, convo in enumerate(long_convos[:10], 1):
            stats = convo['stats']
            f.write(f"{i}. **{convo['title']}**\n")
            f.write(f"   - Total messages: {stats['total_messages']}\n")
            f.write(f"   - User messages: {stats['user_messages']}\n")
            f.write(f"   - Assistant messages: {stats['assistant_messages']}\n")
            f.write(f"   - Total characters: {stats['total_chars']:,}\n")
            f.write(f"   - Code blocks: {stats['code_blocks']}\n\n")

        f.write("## Length Distribution\n\n")
        for bucket in sorted(length_distribution.keys()):
            count = length_distribution[bucket]
            f.write(f"- {bucket}-{bucket+9} messages: {count} conversations\n")

    # Print summary
    print(f"\n✅ Long conversation extraction complete!")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📊 Found {len(long_convos)} conversations with {min_messages}+ messages")
    print(f"📄 Report saved to: {report_path.absolute()}")

    if long_convos:
        print(f"\n🏆 Longest conversation: '{long_convos[0]['title']}' with {long_convos[0]['stats']['total_messages']} messages")

def main():
    parser = argparse.ArgumentParser(
        description="Extract conversations with many messages"
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for long conversations")
    parser.add_argument("--min-messages", type=int, default=20,
                       help="Minimum number of messages (default: 20)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")

    args = parser.parse_args()
    extract_long_conversations(args.input_dir, args.output_dir, args.min_messages,
                             copy_files=not args.move)

if __name__ == "__main__":
    main()
