#!/usr/bin/env python3
"""
Resumable Semantic Anchor System for ChatGPT Conversations

This script creates semantic anchors with resumable processing:
1. Checks existing output to skip already processed files
2. Processes files in batches with memory safety
3. Can be stopped and restarted without losing progress
4. Writes results incrementally to avoid memory issues
"""

import json
import csv
import os
import re
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter

# Safe, non-backtracking semantic patterns
SEMANTIC_PATTERNS = {
    'concept': [
        r'\b(?:concept|idea|notion|principle|theory|framework|model|approach)\b',
        r'\b(?:methodology|strategy|technique|method|process|procedure)\b',
        r'\b(?:paradigm|philosophy|doctrine|belief|understanding|perspective)\b'
    ],
    'entity': [
        r'\b(?:project|system|application|platform|tool|software|hardware)\b',
        r'\b(?:company|organization|team|group|department|division)\b',
        r'\b(?:person|individual|user|developer|engineer|architect)\b'
    ],
    'technology': [
        r'\b(?:api|sdk|library|framework|language|database|server)\b',
        r'\b(?:algorithm|protocol|standard|specification|interface)\b',
        r'\b(?:cloud|ai|ml|nlp|blockchain)\b'
    ],
    'domain': [
        r'\b(?:business|finance|healthcare|education|entertainment|science)\b',
        r'\b(?:engineering|design|marketing|sales|operations|research)\b',
        r'\b(?:development|production|testing|deployment|maintenance)\b'
    ],
    'problem': [
        r'\b(?:issue|challenge|problem|obstacle|difficulty|complication)\b',
        r'\b(?:bug|error|failure|crash|performance|scalability)\b',
        r'\b(?:security|privacy|compliance|regulation|requirement)\b'
    ],
    'solution': [
        r'\b(?:solution|fix|workaround|patch|update|improvement)\b',
        r'\b(?:optimization|enhancement|refinement|upgrade|migration)\b',
        r'\b(?:implementation|deployment|integration|configuration)\b'
    ]
}

# Safe technical patterns
TECH_PATTERNS = [
    r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',  # camelCase
    r'\b[a-z]+(?:_[a-z]+)*\b',  # snake_case
    r'\b[A-Z]+(?:_[A-Z]+)*\b',  # UPPER_SNAKE
    r'\b[A-Z]{2,}\b'  # ACRONYMS
]

# Safe code patterns
CODE_PATTERNS = [
    r'`([^`]{1,200})`',  # Inline code with length limit
    r'\b\w+\.(?:py|js|ts|java|cpp|c|h|json|yaml|yml|md|txt)\b',  # File extensions
    r'\b[A-Z][a-zA-Z0-9]*\.(?:py|js|ts|java|cpp|c|h)\b'  # Class files
]

def extract_semantic_anchors(text: str) -> List[Dict[str, Any]]:
    """Extract semantic anchors from text using pattern matching."""
    anchors = []
    text_lower = text.lower()

    # Pattern-based extraction
    for category, patterns in SEMANTIC_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context (limited)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                anchors.append({
                    'type': category,
                    'text': match.group(),
                    'label': category,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'source': 'pattern',
                    'context': context
                })

    # Extract technical terms
    for pattern in TECH_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            anchors.append({
                'type': 'technology',
                'text': match.group(),
                'label': 'technical_term',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.6,
                'source': 'technical_pattern'
            })

    # Extract code references
    for pattern in CODE_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            anchors.append({
                'type': 'code',
                'text': match.group(),
                'label': 'code_reference',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.7,
                'source': 'code_pattern'
            })

    return anchors

def generate_anchor_id(anchor: Dict[str, Any], index: int) -> str:
    """Generate unique anchor ID."""
    hash_input = f"{anchor['text']}_{anchor['type']}_{index}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    return f"ANCHOR_{anchor['type'].upper()}_{index:06d}_{hash_value}"

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
        parts = content['parts']
        if isinstance(parts, list):
            return ' '.join(str(part) for part in parts if part)
        else:
            return str(parts)
    else:
        return str(content)

def get_already_processed_files(csv_path: str) -> Set[str]:
    """Get set of conversation IDs that have already been processed."""
    processed = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed.add(row['conversation_id'])
        except Exception as e:
            print(f"⚠️  Warning: Could not read existing CSV: {e}")
    return processed

def process_conversation_streaming(file_path: str, conversation_id: str, csv_writer, max_file_size_mb: int = 10) -> int:
    """Process a single conversation file and write anchors directly to CSV."""
    anchor_count = 0

    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_file_size_mb:
            print(f"⚠️  Skipping large file: {os.path.basename(file_path)} ({file_size_mb:.1f}MB)")
            return 0

        print(f"Processing: {os.path.basename(file_path)} ({file_size_mb:.1f}MB)")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        mapping = data.get('mapping', {})
        if not mapping:
            return 0

        # Process each message with limits
        message_count = 0
        max_messages = 1000  # Limit messages per file

        for message_id, message_data in mapping.items():
            if message_count >= max_messages:
                print(f"⚠️  Message limit reached for {os.path.basename(file_path)}")
                break

            if not message_data or 'message' not in message_data:
                continue

            message = message_data['message']
            if not message:
                continue

            # Extract content
            content = extract_message_content(message_data)
            if not content or not isinstance(content, str):
                continue

            # Skip very long messages
            if len(content) > 50000:  # 50KB limit per message
                print(f"⚠️  Skipping long message ({len(content)} chars)")
                continue

            # Extract anchors from message content
            message_anchors = extract_semantic_anchors(content)

            # Write anchors directly to CSV
            for i, anchor in enumerate(message_anchors):
                anchor_id = generate_anchor_id(anchor, anchor_count)

                csv_writer.writerow({
                    'anchor_id': anchor_id,
                    'conversation_id': conversation_id,
                    'message_id': message_id,
                    'type': anchor['type'],
                    'text': anchor['text'],
                    'label': anchor['label'],
                    'confidence': anchor['confidence'],
                    'source': anchor['source'],
                    'cluster_id': -1,  # No clustering in streaming mode
                    'context': anchor.get('context', ''),
                    'start': anchor['start'],
                    'end': anchor['end'],
                    'file_path': file_path
                })
                anchor_count += 1

            message_count += 1

        print(f"✅ Generated {anchor_count} anchors from {message_count} messages")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

    return anchor_count

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
    parser = argparse.ArgumentParser(description='Generate Semantic Anchors for ChatGPT conversations (Resumable)')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output directory for semantic anchor results')
    parser.add_argument('--max-file-size', type=int, default=10, help='Maximum file size in MB to process')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of files to process before progress update')

    args = parser.parse_args()

    print("🧠 Processing conversations for semantic anchor generation (Resumable Mode)...")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load conversation IDs
    conversation_ids = load_conversation_ids(args.conversation_ids)

    # Define output files
    anchors_csv = os.path.join(args.output, 'semantic_anchors.csv')

    # Check for already processed files
    already_processed = get_already_processed_files(anchors_csv)
    print(f"📊 Found {len(already_processed)} already processed conversations")

    # Prepare CSV writer
    fieldnames = [
        'anchor_id', 'conversation_id', 'message_id', 'type', 'text',
        'label', 'confidence', 'source', 'cluster_id', 'context',
        'start', 'end', 'file_path'
    ]

    # Open CSV in append mode if it exists, write mode if not
    mode = 'a' if os.path.exists(anchors_csv) else 'w'
    with open(anchors_csv, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if creating new file
        if mode == 'w':
            writer.writeheader()

        # Process each JSON file
        processed_count = 0
        skipped_count = 0
        total_anchors = 0

        for filename in os.listdir(args.input):
            if filename.endswith('.json'):
                file_path = os.path.join(args.input, filename)
                conversation_id = conversation_ids.get(filename, f"UNKNOWN_{filename}")

                # Skip if already processed
                if conversation_id in already_processed:
                    print(f"⏭️  Skipping already processed: {filename}")
                    skipped_count += 1
                    continue

                # Process the file
                anchors_generated = process_conversation_streaming(
                    file_path, conversation_id, writer, args.max_file_size
                )

                if anchors_generated > 0:
                    processed_count += 1
                    total_anchors += anchors_generated
                else:
                    skipped_count += 1

                # Progress update every batch_size files
                if (processed_count + skipped_count) % args.batch_size == 0:
                    print(f"📈 Progress: {processed_count} processed, {skipped_count} skipped, {total_anchors} total anchors")

        print(f"\n✅ Processing complete!")
        print(f"📊 Files processed: {processed_count}")
        print(f"📊 Files skipped: {skipped_count}")
        print(f"📊 Total anchors generated: {total_anchors}")
        print(f"📄 Results saved to: {os.path.abspath(anchors_csv)}")

if __name__ == "__main__":
    main()
