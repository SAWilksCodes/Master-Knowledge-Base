import json
import os
from pathlib import Path
import argparse
import re
from collections import defaultdict

def is_likely_question(text):
    """Determine if text is likely a question."""
    text = text.strip()

    # Check for question marks
    if '?' in text:
        return True

    # Check for question patterns
    question_patterns = [
        r'^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did)\b',
        r'^(explain|describe|tell me|show me|help me|teach me)\b',
        r'^(please|pls).*(explain|tell|show|help|teach|describe)',
    ]

    text_lower = text.lower()
    for pattern in question_patterns:
        if re.match(pattern, text_lower):
            return True

    return False

def is_quality_response(text, min_length=100):
    """Check if response is substantial enough for training."""
    if len(text) < min_length:
        return False

    # Check for substantive content (not just acknowledgments)
    short_responses = [
        r'^(yes|no|okay|ok|sure|thanks|thank you|sorry)[\.\!]?$',
        r'^i (don\'t|cannot|can\'t) .{0,20}$',
    ]

    text_lower = text.lower().strip()
    for pattern in short_responses:
        if re.match(pattern, text_lower):
            return False

    return True

def extract_messages_from_mapping(mapping):
    """Extract messages from ChatGPT mapping structure in order."""
    messages = []

    # Find root node
    root_id = None
    for node_id, node in mapping.items():
        if node.get('parent') is None:
            root_id = node_id
            break

    if not root_id:
        return messages

    # Build a tree structure for traversal
    children_map = defaultdict(list)
    for node_id, node in mapping.items():
        parent = node.get('parent')
        if parent:
            children_map[parent].append(node_id)

    # DFS traversal to get messages in order
    def traverse(node_id, path=[]):
        if node_id not in mapping:
            return

        node = mapping[node_id]

        # Extract message if present
        if 'message' in node and node['message']:
            msg = node['message']

            # Get role
            role = 'unknown'
            if 'author' in msg and isinstance(msg['author'], dict):
                role = msg['author'].get('role', 'unknown')

            # Extract text content
            text = ''
            if 'content' in msg:
                content = msg['content']
                if isinstance(content, dict) and 'parts' in content:
                    parts = content['parts']
                    if parts and isinstance(parts[0], str):
                        text = parts[0]
                elif isinstance(content, str):
                    text = content

            if text and role in ['user', 'assistant']:
                messages.append({
                    'role': role,
                    'text': text.strip(),
                    'node_id': node_id
                })

        # Traverse children
        for child_id in children_map.get(node_id, []):
            traverse(child_id, path + [node_id])

    traverse(root_id)
    return messages

def extract_qa_from_conversation(convo):
    """Extract high-quality Q&A pairs from a conversation."""
    qa_pairs = []
    title = convo.get('title', 'Untitled')

    # Extract messages based on structure
    messages = []

    if 'mapping' in convo:
        # ChatGPT export format
        messages = extract_messages_from_mapping(convo['mapping'])

    elif 'messages' in convo:
        # Alternative format with messages array
        for msg in convo['messages']:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            if isinstance(content, list) and content:
                # Handle content as array
                text = content[0] if isinstance(content[0], str) else str(content[0])
            elif isinstance(content, str):
                text = content
            else:
                continue

            if text and role in ['user', 'assistant']:
                messages.append({
                    'role': role,
                    'text': text.strip()
                })

    # Process messages into Q&A pairs
    i = 0
    while i < len(messages) - 1:
        current = messages[i]

        # Look for user message
        if current['role'] == 'user':
            # Find the next assistant response
            j = i + 1
            while j < len(messages) and messages[j]['role'] != 'assistant':
                j += 1

            if j < len(messages):
                next_msg = messages[j]

                # Check quality
                if (is_likely_question(current['text']) and
                    is_quality_response(next_msg['text'])):

                    qa_pairs.append({
                        'question': current['text'],
                        'answer': next_msg['text'],
                        'conversation_title': title,
                        'metadata': {
                            'source': 'chatgpt_export',
                            'question_length': len(current['text']),
                            'answer_length': len(next_msg['text'])
                        }
                    })

                i = j  # Skip to after the assistant response
            else:
                i += 1
        else:
            i += 1

    return qa_pairs

def save_qa_pairs(qa_pairs, output_file, format='jsonl'):
    """Save Q&A pairs in specified format."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    elif format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    elif format == 'alpaca':
        # Alpaca format for fine-tuning
        alpaca_data = []
        for pair in qa_pairs:
            alpaca_data.append({
                'instruction': pair['question'],
                'input': '',
                'output': pair['answer']
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    elif format == 'markdown':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Extracted Q&A Pairs for Training\n\n")
            f.write(f"Total pairs: {len(qa_pairs)}\n\n---\n\n")

            for i, pair in enumerate(qa_pairs, 1):
                f.write(f"## Q&A Pair {i}\n\n")
                f.write(f"**Source**: {pair['conversation_title']}\n\n")
                f.write(f"### Question:\n{pair['question']}\n\n")
                f.write(f"### Answer:\n{pair['answer']}\n\n")
                f.write("---\n\n")

def extract_qa_from_directory(input_dir, output_file, format='jsonl', min_answer_length=100):
    """Extract Q&A pairs from all conversations in directory."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob('*.json'))

    print(f"🎯 Extracting Q&A pairs from {len(json_files)} conversations...")

    all_qa_pairs = []
    stats = {
        'conversations_with_qa': 0,
        'total_questions': 0,
        'avg_question_length': 0,
        'avg_answer_length': 0,
        'conversations_processed': 0
    }

    question_lengths = []
    answer_lengths = []

    for idx, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                convo = json.load(f)

            qa_pairs = extract_qa_from_conversation(convo)

            # Filter by answer length
            qa_pairs = [p for p in qa_pairs if p['metadata']['answer_length'] >= min_answer_length]

            if qa_pairs:
                stats['conversations_with_qa'] += 1
                all_qa_pairs.extend(qa_pairs)

                # Collect lengths for statistics
                for pair in qa_pairs:
                    question_lengths.append(pair['metadata']['question_length'])
                    answer_lengths.append(pair['metadata']['answer_length'])

            stats['conversations_processed'] += 1

            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{len(json_files)} files...")

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Calculate statistics
    stats['total_questions'] = len(all_qa_pairs)
    if question_lengths:
        stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)

    # Save Q&A pairs
    if all_qa_pairs:
        save_qa_pairs(all_qa_pairs, output_file, format)

        print(f"\n✅ Q&A extraction complete!")
        print(f"📁 Output file: {Path(output_file).absolute()}")
        print(f"\n📊 Statistics:")
        print(f"   Total conversations processed: {stats['conversations_processed']}")
        print(f"   Conversations with Q&A pairs: {stats['conversations_with_qa']}")
        print(f"   Total Q&A pairs extracted: {stats['total_questions']}")
        print(f"   Average question length: {stats['avg_question_length']:.0f} chars")
        print(f"   Average answer length: {stats['avg_answer_length']:.0f} chars")
    else:
        print("\n❌ No Q&A pairs found!")

    return all_qa_pairs

def main():
    parser = argparse.ArgumentParser(
        description="Extract Q&A pairs from ChatGPT conversations for training"
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-o", "--output-file", required=True, help="Output file for Q&A pairs")
    parser.add_argument("-f", "--format", choices=['jsonl', 'json', 'alpaca', 'markdown'],
                       default='jsonl', help="Output format (default: jsonl)")
    parser.add_argument("--min-answer-length", type=int, default=100,
                       help="Minimum answer length to include (default: 100)")

    args = parser.parse_args()
    extract_qa_from_directory(
        args.input_dir,
        args.output_file,
        args.format,
        args.min_answer_length
    )

if __name__ == "__main__":
    main()
