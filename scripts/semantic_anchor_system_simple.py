#!/usr/bin/env python3
"""
Simple Semantic Anchor System for ChatGPT Conversations (Streaming & Resumable)

This script creates semantic anchors from conversations using a memory-efficient
streaming approach. It writes anchors to a CSV file as they are found,
preventing memory overload.

It is RESUMABLE. If stopped and restarted, it will skip already-processed files.
"""

import json
import csv
import os
import re
import hashlib
import argparse
from typing import List, Dict, Any, Set

# Semantic anchor patterns (simplified) - These are fine
SEMANTIC_PATTERNS = {
    'concept': [
        r'\b(concept|idea|notion|principle|theory|framework|model|approach)\b',
        r'\b(methodology|strategy|technique|method|process|procedure)\b',
        r'\b(paradigm|philosophy|doctrine|belief|understanding|perspective)\b'
    ],
    'entity': [
        r'\b(project|system|application|platform|tool|software|hardware)\b',
        r'\b(company|organization|team|group|department|division)\b',
        r'\b(person|individual|user|developer|engineer|architect)\b'
    ],
    'technology': [
        r'\b(api|sdk|library|framework|language|database|server)\b',
        r'\b(algorithm|protocol|standard|specification|interface)\b',
        r'\b(cloud|ai|ml|nlp|computer vision|blockchain)\b'
    ],
    'domain': [
        r'\b(business|finance|healthcare|education|entertainment|science)\b',
        r'\b(engineering|design|marketing|sales|operations|research)\b',
        r'\b(development|production|testing|deployment|maintenance)\b'
    ],
    'problem': [
        r'\b(issue|challenge|problem|obstacle|difficulty|complication)\b',
        r'\b(bug|error|failure|crash|performance|scalability)\b',
        r'\b(security|privacy|compliance|regulation|requirement)\b'
    ],
    'solution': [
        r'\b(solution|fix|workaround|patch|update|improvement)\b',
        r'\b(optimization|enhancement|refinement|upgrade|migration)\b',
        r'\b(implementation|deployment|integration|configuration)\b'
    ]
}

def extract_semantic_anchors(text: str) -> List[Dict[str, Any]]:
    """Extract semantic anchors from text using pattern matching."""
    anchors = []
    text_lower = text.lower()

    # Pattern-based extraction
    for category, patterns in SEMANTIC_PATTERNS.items():
        for pattern in patterns:
            try:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
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
            except re.error:
                # Ignore regex errors on malformed text
                continue
    return anchors

def generate_anchor_id(anchor: Dict[str, Any], conv_id: str, msg_id: str, index: int) -> str:
    """Generate a more unique anchor ID to prevent collisions."""
    hash_input = f"{conv_id}_{msg_id}_{anchor['text']}_{anchor['type']}_{index}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    return f"ANCHOR_{anchor['type'].upper()}_{hash_value}"

def extract_message_content(message_data: Dict[str, Any]) -> str:
    """Extract text content from message data structure."""
    if not message_data or 'message' not in message_data:
        return ""
    message = message_data.get('message', {})
    if not message or 'content' not in message:
        return ""
    content = message['content']
    if 'parts' in content and isinstance(content['parts'], list):
        return ' '.join(str(part) for part in content['parts'] if isinstance(part, str))
    return ""

def process_conversation_streaming(file_path: str, conversation_id: str, anchors_writer: csv.DictWriter) -> int:
    """
    Process a single conversation file and write anchors directly to the CSV writer.
    This function is designed to have a low and stable memory footprint.
    """
    anchors_found_in_file = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f" -> Skipping corrupted JSON in {os.path.basename(file_path)}")
                return 0

        mapping = data.get('mapping', {})
        if not mapping:
            return 0

        # Process each message one by one
        for message_id, message_data in mapping.items():
            content = extract_message_content(message_data)
            if not content:
                continue

            # This list is small and only for the current message
            message_anchors = extract_semantic_anchors(content)

            # Write anchors for this message immediately to the file
            for i, anchor in enumerate(message_anchors):
                anchor.update({
                    'conversation_id': conversation_id,
                    'message_id': message_id,
                    'file_path': os.path.basename(file_path),
                    'cluster_id': -1,  # Clustering is disabled for now
                })
                anchor['anchor_id'] = generate_anchor_id(anchor, conversation_id, message_id, i)

                clean_anchor = {k: v for k, v in anchor.items() if k in anchors_writer.fieldnames}
                anchors_writer.writerow(clean_anchor)
                anchors_found_in_file += 1

    except Exception as e:
        print(f" -> UNEXPECTED ERROR in {os.path.basename(file_path)}: {e}")

    return anchors_found_in_file

def load_conversation_ids(csv_path: str) -> Dict[str, str]:
    """Load conversation ID mappings from CSV."""
    conversation_ids = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use os.path.basename to match just the filename
                filename_key = os.path.basename(row['filename'])
                conversation_ids[filename_key] = row['conversation_id']
    except FileNotFoundError:
        print(f"Error: Conversation ID file not found at {csv_path}")
        return {}
    except Exception as e:
        print(f"Error loading conversation IDs: {e}")
    return conversation_ids

def get_processed_files(csv_path: str) -> Set[str]:
    """Read the output CSV to find which files have already been processed."""
    if not os.path.exists(csv_path):
        return set()

    processed = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Check if file is empty
            if os.fstat(f.fileno()).st_size == 0:
                return set()

            reader = csv.DictReader(f)
            for row in reader:
                if 'file_path' in row:
                    processed.add(row['file_path'])
    except Exception as e:
        print(f"Warning: Could not read existing anchors file to check progress: {e}")

    return processed

def main():
    parser = argparse.ArgumentParser(description='Generate Semantic Anchors for ChatGPT conversations')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output directory for semantic anchor results')
    args = parser.parse_args()

    print("🧠 Starting Streaming & Resumable Semantic Anchor Generation...")
    os.makedirs(args.output, exist_ok=True)

    conversation_ids = load_conversation_ids(args.conversation_ids)
    if not conversation_ids:
        print("Could not load conversation IDs. Aborting.")
        return

    anchors_csv_path = os.path.join(args.output, 'semantic_anchors.csv')
    anchor_fieldnames = [
        'anchor_id', 'conversation_id', 'message_id', 'type', 'text',
        'label', 'confidence', 'source', 'cluster_id', 'context',
        'start', 'end', 'file_path'
    ]

    # --- RESUMABILITY ---
    processed_files = get_processed_files(anchors_csv_path)
    if processed_files:
        print(f"Found {len(processed_files)} already-processed files. Resuming...")

    is_new_file = not os.path.exists(anchors_csv_path) or not processed_files

    total_anchors_generated = 0
    processed_count = 0

    file_list = sorted([f for f in os.listdir(args.input) if f.endswith('.json')])
    total_files = len(file_list)
    print(f"Found {total_files} total conversation files.")

    # --- APPEND MODE ---
    with open(anchors_csv_path, 'a', newline='', encoding='utf-8') as anchors_file:
        anchors_writer = csv.DictWriter(anchors_file, fieldnames=anchor_fieldnames)

        # Write header only if it's a new file
        if is_new_file:
            anchors_writer.writeheader()

        for filename in file_list:
            base_filename = os.path.basename(filename)

            # --- SKIP PROCESSED FILES ---
            if base_filename in processed_files:
                continue

            file_path = os.path.join(args.input, filename)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            conversation_id = conversation_ids.get(base_filename, f"UNKNOWN_{base_filename}")

            print(f"[{processed_count + len(processed_files) + 1}/{total_files}] Processing {filename} ({file_size_mb:.2f} MB)...", end='', flush=True)

            anchors_found = process_conversation_streaming(file_path, conversation_id, anchors_writer)

            print(f" found {anchors_found} anchors.")
            total_anchors_generated += anchors_found
            processed_count += 1

    print("\n✅ Semantic anchor generation complete!")
    print(f"📄 Anchors CSV is at: {os.path.abspath(anchors_csv_path)}")
    print(f"📊 Added {total_anchors_generated} new semantic anchors from {processed_count} new conversations.")
    print("\nNOTE: Clustering and relationship analysis were skipped to ensure stability. This can be run as a separate step on the generated CSV file.")

if __name__ == "__main__":
    main()
