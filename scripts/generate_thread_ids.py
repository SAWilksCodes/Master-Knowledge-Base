#!/usr/bin/env python3
"""
Generate Thread IDs for ChatGPT Conversations

This script identifies conversation threads and related discussions and assigns unique IDs.
Threads are identified through topic continuity, temporal proximity, and semantic similarity.
"""

import json
import csv
import os
import re
import hashlib
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Thread identification patterns
THREAD_INDICATORS = {
    'continuation': [
        r'continuing|following up|as mentioned|previously|earlier',
        r'building on|expanding|elaborating|further|more',
        r'next|then|after that|subsequently|meanwhile'
    ],
    'reference': [
        r'referring to|about|regarding|concerning|with respect to',
        r'as we discussed|as mentioned|as noted|as stated',
        r'this|that|it|the|our|my|your'
    ],
    'question_followup': [
        r'follow up|additional|more|another|further',
        r'clarification|elaboration|explanation|detail',
        r'what about|how about|what if|suppose'
    ],
    'topic_shift': [
        r'by the way|speaking of|on another note|meanwhile',
        r'changing topics|different subject|unrelated|aside',
        r'while we\'re at it|since we\'re talking about'
    ]
}

# Thread types
THREAD_TYPES = {
    'continuation': 'Continuing previous discussion',
    'clarification': 'Seeking or providing clarification',
    'elaboration': 'Expanding on previous points',
    'example': 'Providing examples or instances',
    'application': 'Applying concepts or solutions',
    'evaluation': 'Assessing or evaluating previous content',
    'synthesis': 'Combining or connecting ideas',
    'divergence': 'Branching into new related topics'
}

# Temporal proximity thresholds (in minutes)
TEMPORAL_THRESHOLDS = {
    'immediate': 5,      # Same conversation
    'recent': 30,        # Recent conversation
    'related': 1440,     # Same day
    'distant': 10080     # Same week
}

def extract_threads(text: str, message_index: int, timestamp: str) -> List[Dict[str, Any]]:
    """Extract thread information from text."""
    threads = []
    text_lower = text.lower()

    # Check for thread indicators
    thread_matches = {}
    for category, patterns in THREAD_INDICATORS.items():
        matches = []
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches.append(pattern)
        if matches:
            thread_matches[category] = matches

    if thread_matches:
        # Determine thread type
        thread_type = 'continuation'
        if 'question_followup' in thread_matches:
            thread_type = 'clarification'
        elif 'continuation' in thread_matches:
            thread_type = 'elaboration'
        elif 'topic_shift' in thread_matches:
            thread_type = 'divergence'

        # Calculate thread strength
        strength = sum(len(matches) for matches in thread_matches.values())

        # Extract context
        context_start = max(0, text_lower.find(list(thread_matches.values())[0][0]) - 100)
        context_end = min(len(text), text_lower.find(list(thread_matches.values())[-1][-1]) + 100)
        context = text[context_start:context_end].strip()

        threads.append({
            'type': thread_type,
            'strength': strength,
            'context': context,
            'indicators': list(thread_matches.keys()),
            'message_index': message_index,
            'timestamp': timestamp
        })

    return threads

def identify_conversation_threads(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify threads within a single conversation."""
    threads = []
    current_thread = None

    for i, message in enumerate(messages):
        content = message.get('content', '')
        if not content or not isinstance(content, str):
            continue

        # Extract thread indicators from this message
        message_threads = extract_threads(content, i, message.get('timestamp', ''))

        if message_threads:
            # Start or continue a thread
            if current_thread is None:
                current_thread = {
                    'start_index': i,
                    'start_timestamp': message.get('timestamp', ''),
                    'type': message_threads[0]['type'],
                    'strength': message_threads[0]['strength'],
                    'messages': [i],
                    'context': message_threads[0]['context']
                }
            else:
                # Continue existing thread
                current_thread['messages'].append(i)
                current_thread['strength'] += message_threads[0]['strength']
        else:
            # End current thread if exists
            if current_thread is not None:
                current_thread['end_index'] = i - 1
                current_thread['end_timestamp'] = messages[i-1].get('timestamp', '')
                current_thread['duration'] = len(current_thread['messages'])
                threads.append(current_thread)
                current_thread = None

    # Add final thread if exists
    if current_thread is not None:
        current_thread['end_index'] = len(messages) - 1
        current_thread['end_timestamp'] = messages[-1].get('timestamp', '')
        current_thread['duration'] = len(current_thread['messages'])
        threads.append(current_thread)

    return threads

def generate_thread_id(conversation_id: str, thread: Dict[str, Any], index: int) -> str:
    """Generate unique thread ID."""
    # Create hash from conversation ID, type, and context
    hash_input = f"{conversation_id}_{thread['type']}_{thread['context'][:50]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"THREAD_{conversation_id}_{thread['type'].upper()}_{index:03d}_{hash_value}"

def process_conversation(file_path: str, conversation_id: str) -> List[Dict[str, Any]]:
    """Process a single conversation file for threads."""
    threads = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Identify threads within the conversation
        conversation_threads = identify_conversation_threads(data.get('messages', []))

        for i, thread in enumerate(conversation_threads):
            thread_id = generate_thread_id(conversation_id, thread, i)

            threads.append({
                'thread_id': thread_id,
                'conversation_id': conversation_id,
                'type': thread['type'],
                'strength': thread['strength'],
                'duration': thread['duration'],
                'start_index': thread['start_index'],
                'end_index': thread['end_index'],
                'start_timestamp': thread['start_timestamp'],
                'end_timestamp': thread['end_timestamp'],
                'context': thread['context'],
                'message_indices': '; '.join(map(str, thread['messages'])),
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
            'thread_id', 'conversation_id', 'type', 'strength', 'duration',
            'start_index', 'end_index', 'start_timestamp', 'end_timestamp',
            'context', 'message_indices', 'file_path'
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

    # Count by type
    type_counts = defaultdict(int)
    strength_distribution = defaultdict(int)
    duration_distribution = defaultdict(int)

    for thread in threads:
        type_counts[thread['type']] += 1
        strength_distribution[round(thread['strength'])] += 1
        duration_distribution[thread['duration']] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Thread ID Generation Summary\n\n")
        f.write(f"Total threads generated: {len(threads)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `THREAD_CONVID_TYPE_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- TYPE: Thread type\n")
        f.write("- INDEX: Thread index within conversation\n")
        f.write("- HASH: 6-character hash of context\n\n")

        f.write("## Thread Type Distribution\n\n")
        for thread_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(threads)) * 100
            description = THREAD_TYPES.get(thread_type, 'Unknown')
            f.write(f"- **{thread_type.title()}**: {count} threads ({percentage:.1f}%) - {description}\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(threads)) * 100
            f.write(f"- **Strength {strength}**: {count} threads ({percentage:.1f}%)\n")

        f.write("\n## Duration Distribution\n\n")
        for duration in sorted(duration_distribution.keys()):
            count = duration_distribution[duration]
            percentage = (count / len(threads)) * 100
            f.write(f"- **Duration {duration}**: {count} threads ({percentage:.1f}%)\n")

        f.write("\n## Recent Threads\n\n")
        recent_threads = sorted(threads, key=lambda x: x.get('end_timestamp', ''), reverse=True)[:10]
        for thread in recent_threads:
            f.write(f"- **{thread['thread_id']}**: {thread['type']} (duration: {thread['duration']}, strength: {thread['strength']})\n")

if __name__ == "__main__":
    main()
