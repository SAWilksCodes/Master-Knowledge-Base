import json
import csv
import os
from pathlib import Path
import argparse
from datetime import datetime
import re

def extract_first_user_message(convo):
    """Extract the first user message from a conversation."""
    # Try mapping structure
    if 'mapping' in convo:
        # Find all user messages
        user_messages = []

        for node_id, node in convo['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']
                if 'author' in msg and isinstance(msg['author'], dict):
                    if msg['author'].get('role') == 'user':
                        # Get timestamp if available
                        timestamp = msg.get('create_time', 0)

                        # Extract text
                        text = ''
                        if 'content' in msg:
                            content = msg['content']
                            if isinstance(content, dict) and 'parts' in content:
                                for part in content['parts']:
                                    if isinstance(part, str):
                                        text = part
                                        break

                        if text:
                            user_messages.append((timestamp, text))

        # Sort by timestamp and return first
        if user_messages:
            user_messages.sort(key=lambda x: x[0])
            return user_messages[0][1][:200]  # First 200 chars

    # Try messages array
    elif 'messages' in convo:
        for msg in convo['messages']:
            if msg.get('role') == 'user' and msg.get('content'):
                content = msg['content']
                if isinstance(content, str):
                    return content[:200]

    return "No user message found"

def get_conversation_tags(convo):
    """Extract tags/topics from conversation."""
    tags = set()

    # Get title
    title = convo.get('title', '').lower()

    # Extract text for analysis
    text_sample = title + ' ' + extract_first_user_message(convo).lower()

    # Simple tag detection
    tag_patterns = {
        'python': r'\bpython\b',
        'javascript': r'\b(javascript|js|node)\b',
        'code': r'\b(code|coding|program|debug|error|function|class)\b',
        'data': r'\b(data|analysis|pandas|csv|excel)\b',
        'ai': r'\b(ai|machine learning|neural|gpt|llm)\b',
        'writing': r'\b(write|writing|essay|article|content)\b',
        'math': r'\b(math|equation|calculate|formula)\b',
        'api': r'\b(api|endpoint|request|http)\b',
        'web': r'\b(web|website|html|css|react)\b',
        'help': r'\b(help|how to|explain|what is)\b'
    }

    for tag, pattern in tag_patterns.items():
        if re.search(pattern, text_sample):
            tags.add(tag)

    return list(tags)

def count_messages(convo):
    """Count messages in conversation."""
    count = 0

    if 'mapping' in convo:
        for node_id, node in convo['mapping'].items():
            if 'message' in node and node['message']:
                if 'author' in node['message']:
                    count += 1
    elif 'messages' in convo:
        count = len(convo['messages'])

    return count

def create_index(input_dir, output_file):
    """Create a CSV index of all conversations."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob('*.json'))

    print(f"🔍 Creating index for {len(json_files)} conversations...")

    rows = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                convo = json.load(f)

            # Extract metadata
            convo_id = convo.get('conversation_id', json_file.stem)
            title = convo.get('title', 'Untitled')

            # Get timestamp
            timestamp = None
            for field in ['create_time', 'created_at', 'timestamp']:
                if field in convo and convo[field]:
                    timestamp = convo[field]
                    break

            # Format date
            date_str = 'Unknown'
            if timestamp:
                try:
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                    else:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass

            # Get other metadata
            message_count = count_messages(convo)
            first_message = extract_first_user_message(convo)
            tags = ', '.join(get_conversation_tags(convo))
            model = convo.get('default_model_slug', 'unknown')

            rows.append({
                'filename': json_file.name,
                'conversation_id': convo_id,
                'title': title,
                'date': date_str,
                'message_count': message_count,
                'first_user_message': first_message,
                'tags': tags,
                'model': model
            })

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Sort by date
    rows.sort(key=lambda x: x['date'], reverse=True)

    # Write CSV
    output_path = Path(output_file)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'conversation_id', 'title', 'date',
                     'message_count', 'first_user_message', 'tags', 'model']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    # Also create a markdown summary
    summary_path = output_path.with_suffix('.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Conversation Index Summary\n\n")
        f.write(f"Total conversations: {len(rows)}\n\n")

        # Tag statistics
        all_tags = {}
        for row in rows:
            if row['tags']:
                for tag in row['tags'].split(', '):
                    all_tags[tag] = all_tags.get(tag, 0) + 1

        f.write("## Most Common Topics\n\n")
        for tag, count in sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"- {tag}: {count} conversations\n")

        # Model statistics
        model_stats = {}
        for row in rows:
            model = row['model']
            model_stats[model] = model_stats.get(model, 0) + 1

        f.write("\n## Models Used\n\n")
        for model, count in sorted(model_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {model}: {count} conversations\n")

        # Recent conversations
        f.write("\n## 10 Most Recent Conversations\n\n")
        for row in rows[:10]:
            f.write(f"- **{row['title']}** ({row['date']})\n")
            f.write(f"  - Messages: {row['message_count']}\n")
            f.write(f"  - Tags: {row['tags'] or 'none'}\n\n")

    print(f"\n✅ Index creation complete!")
    print(f"📄 CSV index: {output_path.absolute()}")
    print(f"📄 Summary report: {summary_path.absolute()}")
    print(f"📊 Indexed {len(rows)} conversations")

def main():
    parser = argparse.ArgumentParser(
        description="Create a searchable CSV index of all conversations"
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-o", "--output", default="conversation_index.csv",
                       help="Output CSV file (default: conversation_index.csv)")

    args = parser.parse_args()
    create_index(args.input_dir, args.output)

if __name__ == "__main__":
    main()
