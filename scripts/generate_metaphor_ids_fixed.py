#!/usr/bin/env python3
"""
Generate Metaphor IDs for ChatGPT Conversations (Fixed Version)

This script identifies metaphorical language patterns in conversations and assigns unique IDs.
Metaphors are identified through pattern matching and linguistic analysis.
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

# Metaphor patterns and indicators
METAPHOR_PATTERNS = {
    'journey': [
        r'\b(journey|path|road|travel|walk|step|move forward|progress|destination)\b',
        r'\b(start|begin|continue|reach|arrive|complete|finish)\b'
    ],
    'building': [
        r'\b(build|construct|create|foundation|structure|framework|architecture)\b',
        r'\b(lay the groundwork|build upon|construct|assemble|put together)\b'
    ],
    'growth': [
        r'\b(grow|develop|evolve|mature|bloom|flourish|thrive|expand)\b',
        r'\b(seed|plant|root|branch|fruit|harvest|cultivate)\b'
    ],
    'light': [
        r'\b(light|illuminate|bright|shine|glow|spark|ignite|enlighten)\b',
        r'\b(dark|shadow|obscure|clarify|reveal|expose)\b'
    ],
    'water': [
        r'\b(flow|stream|river|ocean|wave|tide|current|dive|swim)\b',
        r'\b(pour|flood|drain|sink|float|navigate)\b'
    ],
    'machine': [
        r'\b(machine|engine|mechanism|gear|cog|process|system|automate)\b',
        r'\b(break down|fix|repair|maintain|optimize|efficiency)\b'
    ],
    'war': [
        r'\b(battle|fight|war|attack|defend|victory|defeat|strategy)\b',
        r'\b(weapon|armor|shield|target|enemy|ally|triumph)\b'
    ],
    'game': [
        r'\b(game|play|win|lose|score|level|challenge|competition)\b',
        r'\b(player|team|strategy|move|turn|round|match)\b'
    ],
    'food': [
        r'\b(cook|recipe|ingredient|taste|flavor|spice|nourish|digest)\b',
        r'\b(feast|hunger|appetite|satisfy|consume|prepare)\b'
    ],
    'nature': [
        r'\b(nature|wild|natural|organic|raw|pure|untamed|wilderness)\b',
        r'\b(forest|mountain|river|sky|earth|wind|storm)\b'
    ]
}

METAPHOR_INDICATORS = [
    r'\b(like|as if|as though|similar to|resembles|reminds me of)\b',
    r'\b(metaphorically|figuratively|symbolically|literally)\b',
    r'\b(imagine|picture|visualize|think of it as)\b'
]

def extract_metaphors(text: str) -> List[Dict[str, Any]]:
    """Extract metaphors from text using pattern matching."""
    metaphors = []
    text_lower = text.lower()

    # Check for metaphor indicators
    has_indicators = any(re.search(pattern, text_lower) for pattern in METAPHOR_INDICATORS)

    # Check each metaphor category
    for category, patterns in METAPHOR_PATTERNS.items():
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
            # Calculate metaphor strength based on matches and context
            strength = len(matches)
            if has_indicators:
                strength += 2

            # Extract surrounding context
            context_start = max(0, min(matches[0]['start'] - 50, len(text)))
            context_end = min(len(text), max(matches[-1]['end'] + 50, len(text)))
            context = text[context_start:context_end].strip()

            metaphors.append({
                'category': category,
                'matches': matches,
                'strength': strength,
                'context': context,
                'indicators_present': has_indicators
            })

    return metaphors

def generate_metaphor_id(conversation_id: str, metaphor: Dict[str, Any], index: int) -> str:
    """Generate unique metaphor ID."""
    # Create hash from conversation ID, category, and context
    hash_input = f"{conversation_id}_{metaphor['category']}_{metaphor['context'][:50]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"METAPHOR_{conversation_id}_{metaphor['category'].upper()}_{index:03d}_{hash_value}"

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

def process_conversation(file_path: str, conversation_id: str) -> List[Dict[str, Any]]:
    """Process a single conversation file for metaphors."""
    metaphors = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle the actual JSON structure with mapping object
        mapping = data.get('mapping', {})
        if not mapping:
            return metaphors

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

            # Extract metaphors from message content
            message_metaphors = extract_metaphors(content)

            for i, metaphor in enumerate(message_metaphors):
                metaphor_id = generate_metaphor_id(conversation_id, metaphor, len(metaphors))

                # Get message metadata
                author = message.get('author', {})
                role = author.get('role', 'unknown') if author else 'unknown'
                create_time = message.get('create_time', '')
                timestamp = datetime.fromtimestamp(create_time).isoformat() if create_time else ''

                metaphors.append({
                    'metaphor_id': metaphor_id,
                    'conversation_id': conversation_id,
                    'category': metaphor['category'],
                    'strength': metaphor['strength'],
                    'context': metaphor['context'],
                    'indicators_present': metaphor['indicators_present'],
                    'message_id': message_id,
                    'role': role,
                    'timestamp': timestamp,
                    'file_path': file_path
                })

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return metaphors

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
    parser = argparse.ArgumentParser(description='Generate Metaphor IDs for ChatGPT conversations')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file for metaphor IDs')

    args = parser.parse_args()

    print("🔍 Processing conversations for metaphor ID generation...")

    # Load conversation IDs
    conversation_ids = load_conversation_ids(args.conversation_ids)

    all_metaphors = []
    processed_count = 0

    # Process each JSON file
    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input, filename)
            conversation_id = conversation_ids.get(filename, f"UNKNOWN_{filename}")

            metaphors = process_conversation(file_path, conversation_id)
            all_metaphors.extend(metaphors)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} conversations...")

    # Write results to CSV
    if all_metaphors:
        fieldnames = [
            'metaphor_id', 'conversation_id', 'category', 'strength',
            'context', 'indicators_present', 'message_id', 'role',
            'timestamp', 'file_path'
        ]

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metaphors)

        # Generate summary
        summary_path = args.output.replace('.csv', '.md')
        generate_summary(all_metaphors, summary_path, processed_count)

        print(f"\n✅ Metaphor ID generation complete!")
        print(f"📄 CSV file: {os.path.abspath(args.output)}")
        print(f"📄 Summary: {os.path.abspath(summary_path)}")
        print(f"📊 Generated {len(all_metaphors)} metaphor IDs")
    else:
        print("❌ No metaphors found in conversations")

def generate_summary(metaphors: List[Dict[str, Any]], output_path: str, total_conversations: int):
    """Generate summary markdown file."""
    if not metaphors:
        return

    # Count by category
    category_counts = defaultdict(int)
    strength_distribution = defaultdict(int)

    for metaphor in metaphors:
        category_counts[metaphor['category']] += 1
        strength_distribution[metaphor['strength']] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Metaphor ID Generation Summary\n\n")
        f.write(f"Total metaphors generated: {len(metaphors)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `METAPHOR_CONVID_CATEGORY_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- CATEGORY: Metaphor category\n")
        f.write("- INDEX: Metaphor index within conversation\n")
        f.write("- HASH: 6-character hash of context\n\n")

        f.write("## Category Distribution\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(metaphors)) * 100
            f.write(f"- **{category.title()}**: {count} metaphors ({percentage:.1f}%)\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(metaphors)) * 100
            f.write(f"- **Strength {strength}**: {count} metaphors ({percentage:.1f}%)\n")

        f.write("\n## Recent Metaphors\n\n")
        recent_metaphors = sorted(metaphors, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        for metaphor in recent_metaphors:
            f.write(f"- **{metaphor['metaphor_id']}**: {metaphor['category']} - {metaphor['context'][:100]}...\n")

if __name__ == "__main__":
    main()
