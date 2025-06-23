#!/usr/bin/env python3
"""
Generate Concept IDs for ChatGPT Conversations (Fixed Version)

This script identifies conceptual language patterns in conversations and assigns unique IDs.
Concepts are identified through pattern matching and linguistic analysis.
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

# Concept patterns and indicators
CONCEPT_PATTERNS = {
    'abstraction': [
        r'\b(abstract|concept|idea|theory|principle|framework|model|paradigm)\b',
        r'\b(philosophy|ideology|doctrine|belief|perspective|viewpoint)\b'
    ],
    'system': [
        r'\b(system|structure|organization|hierarchy|network|ecosystem)\b',
        r'\b(architecture|infrastructure|platform|foundation|base)\b'
    ],
    'process': [
        r'\b(process|procedure|method|approach|strategy|technique)\b',
        r'\b(workflow|pipeline|sequence|order|steps|stages)\b'
    ],
    'data': [
        r'\b(data|information|knowledge|insight|analysis|metrics)\b',
        r'\b(dataset|database|repository|storage|retrieval)\b'
    ],
    'interaction': [
        r'\b(interaction|communication|collaboration|cooperation|partnership)\b',
        r'\b(relationship|connection|link|bridge|interface)\b'
    ],
    'transformation': [
        r'\b(transform|change|evolve|adapt|modify|convert)\b',
        r'\b(transition|migration|upgrade|enhancement|improvement)\b'
    ],
    'quality': [
        r'\b(quality|excellence|precision|accuracy|reliability)\b',
        r'\b(performance|efficiency|effectiveness|optimization)\b'
    ],
    'scale': [
        r'\b(scale|size|scope|magnitude|proportion|dimension)\b',
        r'\b(growth|expansion|scaling|amplification|multiplication)\b'
    ],
    'time': [
        r'\b(time|duration|period|cycle|frequency|timing)\b',
        r'\b(schedule|timeline|deadline|milestone|phase)\b'
    ],
    'space': [
        r'\b(space|location|position|area|region|domain)\b',
        r'\b(environment|context|setting|atmosphere|surroundings)\b'
    ]
}

CONCEPT_INDICATORS = [
    r'\b(conceptually|theoretically|fundamentally|essentially)\b',
    r'\b(represents|embodies|signifies|denotes|implies)\b',
    r'\b(underlying|core|central|key|primary)\b'
]

def extract_concepts(text: str) -> List[Dict[str, Any]]:
    """Extract concepts from text using pattern matching."""
    concepts = []
    text_lower = text.lower()

    # Check for concept indicators
    has_indicators = any(re.search(pattern, text_lower) for pattern in CONCEPT_INDICATORS)

    # Check each concept category
    for category, patterns in CONCEPT_PATTERNS.items():
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
            # Calculate concept strength based on matches and context
            strength = len(matches)
            if has_indicators:
                strength += 2

            # Extract surrounding context
            context_start = max(0, min(matches[0]['start'] - 50, len(text)))
            context_end = min(len(text), max(matches[-1]['end'] + 50, len(text)))
            context = text[context_start:context_end].strip()

            concepts.append({
                'category': category,
                'matches': matches,
                'strength': strength,
                'context': context,
                'indicators_present': has_indicators
            })

    return concepts

def generate_concept_id(conversation_id: str, concept: Dict[str, Any], index: int) -> str:
    """Generate unique concept ID."""
    # Create hash from conversation ID, category, and context
    hash_input = f"{conversation_id}_{concept['category']}_{concept['context'][:50]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"CONCEPT_{conversation_id}_{concept['category'].upper()}_{index:03d}_{hash_value}"

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
    """Process a single conversation file for concepts."""
    concepts = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle the actual JSON structure with mapping object
        mapping = data.get('mapping', {})
        if not mapping:
            return concepts

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

            # Extract concepts from message content
            message_concepts = extract_concepts(content)

            for i, concept in enumerate(message_concepts):
                concept_id = generate_concept_id(conversation_id, concept, len(concepts))

                # Get message metadata
                author = message.get('author', {})
                role = author.get('role', 'unknown') if author else 'unknown'
                create_time = message.get('create_time', '')
                timestamp = datetime.fromtimestamp(create_time).isoformat() if create_time else ''

                concepts.append({
                    'concept_id': concept_id,
                    'conversation_id': conversation_id,
                    'category': concept['category'],
                    'strength': concept['strength'],
                    'context': concept['context'],
                    'indicators_present': concept['indicators_present'],
                    'message_id': message_id,
                    'role': role,
                    'timestamp': timestamp,
                    'file_path': file_path
                })

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return concepts

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
    parser = argparse.ArgumentParser(description='Generate Concept IDs for ChatGPT conversations')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file for concept IDs')

    args = parser.parse_args()

    print("🔍 Processing conversations for concept ID generation...")

    # Load conversation IDs
    conversation_ids = load_conversation_ids(args.conversation_ids)

    all_concepts = []
    processed_count = 0

    # Process each JSON file
    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input, filename)
            conversation_id = conversation_ids.get(filename, f"UNKNOWN_{filename}")

            concepts = process_conversation(file_path, conversation_id)
            all_concepts.extend(concepts)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} conversations...")

    # Write results to CSV
    if all_concepts:
        fieldnames = [
            'concept_id', 'conversation_id', 'category', 'strength',
            'context', 'indicators_present', 'message_id', 'role',
            'timestamp', 'file_path'
        ]

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_concepts)

        # Generate summary
        summary_path = args.output.replace('.csv', '.md')
        generate_summary(all_concepts, summary_path, processed_count)

        print(f"\n✅ Concept ID generation complete!")
        print(f"📄 CSV file: {os.path.abspath(args.output)}")
        print(f"📄 Summary: {os.path.abspath(summary_path)}")
        print(f"📊 Generated {len(all_concepts)} concept IDs")
    else:
        print("❌ No concepts found in conversations")

def generate_summary(concepts: List[Dict[str, Any]], output_path: str, total_conversations: int):
    """Generate summary markdown file."""
    if not concepts:
        return

    # Count by category
    category_counts = defaultdict(int)
    strength_distribution = defaultdict(int)

    for concept in concepts:
        category_counts[concept['category']] += 1
        strength_distribution[concept['strength']] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Concept ID Generation Summary\n\n")
        f.write(f"Total concepts generated: {len(concepts)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `CONCEPT_CONVID_CATEGORY_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- CATEGORY: Concept category\n")
        f.write("- INDEX: Concept index within conversation\n")
        f.write("- HASH: 6-character hash of context\n\n")

        f.write("## Category Distribution\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(concepts)) * 100
            f.write(f"- **{category.title()}**: {count} concepts ({percentage:.1f}%)\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(concepts)) * 100
            f.write(f"- **Strength {strength}**: {count} concepts ({percentage:.1f}%)\n")

        f.write("\n## Recent Concepts\n\n")
        recent_concepts = sorted(concepts, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        for concept in recent_concepts:
            f.write(f"- **{concept['concept_id']}**: {concept['category']} - {concept['context'][:100]}...\n")

if __name__ == "__main__":
    main()
