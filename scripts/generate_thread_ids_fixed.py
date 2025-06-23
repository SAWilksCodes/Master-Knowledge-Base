#!/usr/bin/env python3
"""
Generate Thread IDs for ChatGPT Conversations (Fixed Version)

This script identifies conversation threading patterns and assigns unique IDs.
Threads are identified through message relationships, topic continuity, and temporal patterns.
Fixed to handle the actual JSON structure with messages in mapping object.
"""

import json
import csv
import os
import re
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Thread patterns and indicators
THREAD_PATTERNS = {
    'continuation': [
        r'\b(continue|next|following|subsequent|proceed|carry on|go on)\b',
        r'\b(and then|after that|meanwhile|meanwhile|in the meantime)\b',
        r'\b(previously|earlier|before|last time|as mentioned|as discussed)\b'
    ],
    'clarification': [
        r'\b(clarify|explain|elaborate|expand|detail|specify|define)\b',
        r'\b(what do you mean|can you explain|I don\'t understand|confused)\b',
        r'\b(question|ask|inquiry|wonder|curious|unsure|uncertain)\b'
    ],
    'correction': [
        r'\b(correct|fix|error|mistake|wrong|incorrect|not right)\b',
        r'\b(actually|in fact|to clarify|let me correct|that\'s not right)\b',
        r'\b(update|revise|modify|change|adjust|amend|rectify)\b'
    ],
    'reference': [
        r'\b(refer|reference|mention|cite|quote|as you said|you mentioned)\b',
        r'\b(earlier|above|below|previously|in the previous message)\b',
        r'\b(this|that|it|the|what we discussed|our conversation)\b'
    ],
    'development': [
        r'\b(develop|build|create|implement|design|construct|establish)\b',
        r'\b(progress|advance|evolve|grow|expand|extend|enhance)\b',
        r'\b(improve|optimize|refine|polish|perfect|finalize)\b'
    ],
    'problem_solving': [
        r'\b(problem|issue|challenge|difficulty|obstacle|trouble|bug)\b',
        r'\b(solve|resolve|fix|address|tackle|overcome|deal with)\b',
        r'\b(solution|approach|method|strategy|technique|workaround)\b'
    ],
    'decision_making': [
        r'\b(decide|choose|select|pick|determine|settle|resolve)\b',
        r'\b(option|choice|alternative|possibility|consider|evaluate)\b',
        r'\b(decision|conclusion|outcome|result|final|definitive)\b'
    ],
    'collaboration': [
        r'\b(collaborate|work together|team|partnership|cooperation)\b',
        r'\b(share|exchange|discuss|brainstorm|ideate|plan together)\b',
        r'\b(we|us|our|together|joint|mutual|collective)\b'
    ],
    'iteration': [
        r'\b(iterate|repeat|cycle|loop|version|revision|draft)\b',
        r'\b(improve|refine|polish|enhance|optimize|perfect)\b',
        r'\b(iteration|version|revision|draft|attempt|try again)\b'
    ],
    'integration': [
        r'\b(integrate|connect|link|combine|merge|unite|join)\b',
        r'\b(api|interface|connection|bridge|gateway|middleware)\b',
        r'\b(system|platform|ecosystem|architecture|framework)\b'
    ],
    'testing': [
        r'\b(test|trial|experiment|validate|verify|check|examine)\b',
        r'\b(debug|troubleshoot|diagnose|investigate|analyze)\b',
        r'\b(result|outcome|finding|discovery|insight|observation)\b'
    ],
    'deployment': [
        r'\b(deploy|launch|release|publish|go live|production)\b',
        r'\b(server|hosting|cloud|infrastructure|environment)\b',
        r'\b(rollout|release|version|update|maintenance)\b'
    ],
    'documentation': [
        r'\b(document|record|note|write|describe|explain|detail)\b',
        r'\b(guide|manual|tutorial|instruction|procedure|process)\b',
        r'\b(readme|wiki|help|support|knowledge base|documentation)\b'
    ],
    'review': [
        r'\b(review|examine|assess|evaluate|analyze|inspect)\b',
        r'\b(feedback|comment|suggestion|recommendation|advice)\b',
        r'\b(quality|performance|efficiency|effectiveness|success)\b'
    ]
}

THREAD_INDICATORS = [
    r'\b(thread|conversation|discussion|dialogue|exchange|interaction)\b',
    r'\b(continue|follow|proceed|next|previous|earlier|later)\b',
    r'\b(context|background|history|timeline|sequence|order)\b'
]

def extract_threads(text: str) -> List[Dict[str, Any]]:
    """Extract thread patterns from text using pattern matching."""
    threads = []
    text_lower = text.lower()

    # Check for thread indicators
    has_indicators = any(re.search(pattern, text_lower) for pattern in THREAD_INDICATORS)

    # Check each thread category
    for category, patterns in THREAD_PATTERNS.items():
        matches = []
        for pattern in patterns:
            found = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in found:
                matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': pattern
                })

        if matches:
            # Calculate thread strength based on matches and context
            strength = len(matches)
            if has_indicators:
                strength += 2

            # Extract surrounding context
            context_start = max(0, min(matches[0]['start'] - 50, len(text)))
            context_end = min(len(text), max(matches[-1]['end'] + 50, len(text)))
            context = text[context_start:context_end].strip()

            threads.append({
                'category': category,
                'matches': matches,
                'strength': strength,
                'context': context,
                'indicators_present': has_indicators
            })

    return threads

def generate_thread_id(conversation_id: str, thread: Dict[str, Any], index: int) -> str:
    """Generate unique thread ID."""
    # Create hash from conversation ID, category, and context
    hash_input = f"{conversation_id}_{thread['category']}_{thread['context'][:50]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"THREAD_{conversation_id}_{thread['category'].upper()}_{index:03d}_{hash_value}"

def extract_message_content(message_data: Dict[str, Any]) -> str:
    """Extract text content from message data structure."""
    if not message_data or 'message' not in message_data:
        return ""

    message = message_data['message']
    if not message or 'content' not in message:
        return ""

    content = message['content']
    if isinstance(content, str):
        return content
    elif isinstance(content, dict) and 'parts' in content:
        # Handle content.parts array structure
        parts = content['parts']
        if isinstance(parts, list):
            return ' '.join(str(part) for part in parts if part)
        else:
            return str(parts)
    else:
        return str(content)

def analyze_message_relationships(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze message relationships and threading patterns."""
    relationships = []

    # Create a map of message IDs to their data
    messages = {}
    for msg_id, msg_data in mapping.items():
        if msg_data and 'message' in msg_data and msg_data['message']:
            messages[msg_id] = msg_data

    # Analyze parent-child relationships
    for msg_id, msg_data in messages.items():
        parent_id = msg_data.get('parent')
        children = msg_data.get('children', [])

        if parent_id and parent_id in messages:
            relationships.append({
                'type': 'parent_child',
                'child_id': msg_id,
                'parent_id': parent_id,
                'relationship_strength': 1
            })

        for child_id in children:
            if child_id in messages:
                relationships.append({
                    'type': 'child_parent',
                    'child_id': child_id,
                    'parent_id': msg_id,
                    'relationship_strength': 1
                })

    return relationships

def process_conversation(file_path: str, conversation_id: str) -> List[Dict[str, Any]]:
    """Process a single conversation file for threads."""
    threads = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle the actual JSON structure with mapping object
        mapping = data.get('mapping', {})
        if not mapping:
            return threads

        # Analyze message relationships
        relationships = analyze_message_relationships(mapping)

        # Process each message in the mapping
        for message_id, message_data in mapping.items():
            if not message_data or 'message' not in message_data:
                continue

            message = message_data['message']
            if not message:
                continue

            # Extract content from the message
            content = extract_message_content(message_data)
            if not content or not isinstance(content, str):
                continue

            # Extract threads from message content
            message_threads = extract_threads(content)

            for i, thread in enumerate(message_threads):
                thread_id = generate_thread_id(conversation_id, thread, len(threads))

                # Get message metadata
                author = message.get('author', {})
                role = author.get('role', 'unknown') if author else 'unknown'
                create_time = message.get('create_time', '')
                timestamp = datetime.fromtimestamp(create_time).isoformat() if create_time else ''

                # Find related messages
                related_messages = []
                for rel in relationships:
                    if rel['child_id'] == message_id or rel['parent_id'] == message_id:
                        related_messages.append(rel)

                threads.append({
                    'thread_id': thread_id,
                    'conversation_id': conversation_id,
                    'category': thread['category'],
                    'strength': thread['strength'],
                    'context': thread['context'],
                    'indicators_present': thread['indicators_present'],
                    'message_id': message_id,
                    'role': role,
                    'timestamp': timestamp,
                    'related_messages': len(related_messages),
                    'file_path': file_path
                })

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return threads

def load_conversation_ids(csv_path: str) -> Dict[str, str]:
    """Load conversation ID mappings from CSV."""
    conversation_ids = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                conversation_ids[row['filename']] = row['conversation_id']
    except Exception as e:
        print(f"Error loading conversation IDs: {e}")
    return conversation_ids

def main():
    parser = argparse.ArgumentParser(description='Generate Thread IDs for ChatGPT conversations')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file for thread IDs')

    args = parser.parse_args()

    print("🔍 Processing conversations for thread ID generation...")

    # Load conversation IDs
    conversation_ids = load_conversation_ids(args.conversation_ids)

    all_threads = []
    processed_count = 0

    # Process each JSON file
    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input, filename)
            conversation_id = conversation_ids.get(filename, f"UNKNOWN_{filename}")

            threads = process_conversation(file_path, conversation_id)
            all_threads.extend(threads)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} conversations...")

    # Write results to CSV
    if all_threads:
        fieldnames = [
            'thread_id', 'conversation_id', 'category', 'strength',
            'context', 'indicators_present', 'message_id', 'role',
            'timestamp', 'related_messages', 'file_path'
        ]

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_threads)

        # Generate summary
        summary_path = args.output.replace('.csv', '.md')
        generate_summary(all_threads, summary_path, processed_count)

        print(f"\n✅ Thread ID generation complete!")
        print(f"📄 CSV file: {os.path.abspath(args.output)}")
        print(f"📄 Summary: {os.path.abspath(summary_path)}")
        print(f"📊 Generated {len(all_threads)} thread IDs")
    else:
        print("❌ No threads found in conversations")

def generate_summary(threads: List[Dict[str, Any]], output_path: str, total_conversations: int):
    """Generate summary markdown file."""
    if not threads:
        return

    # Count by category
    category_counts = defaultdict(int)
    strength_distribution = defaultdict(int)

    for thread in threads:
        category_counts[thread['category']] += 1
        strength_distribution[thread['strength']] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Thread ID Generation Summary\n\n")
        f.write(f"Total threads generated: {len(threads)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `THREAD_CONVID_CATEGORY_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- CATEGORY: Thread category\n")
        f.write("- INDEX: Thread index within conversation\n")
        f.write("- HASH: 6-character hash of context\n\n")

        f.write("## Category Distribution\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(threads)) * 100
            f.write(f"- **{category.replace('_', ' ').title()}**: {count} threads ({percentage:.1f}%)\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(threads)) * 100
            f.write(f"- **Strength {strength}**: {count} threads ({percentage:.1f}%)\n")

        f.write("\n## Recent Threads\n\n")
        recent_threads = sorted(threads, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        for thread in recent_threads:
            f.write(f"- **{thread['thread_id']}**: {thread['category']} - {thread['context'][:100]}...\n")

if __name__ == "__main__":
    main()
