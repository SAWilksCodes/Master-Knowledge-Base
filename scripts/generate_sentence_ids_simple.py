#!/usr/bin/env python3
import json
import csv
import os
import re
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
import string

def clean_text(text):
    if not text or not isinstance(text, str):
        return ''
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    text = re.sub(r'([,.!?])\s*([,.!?])', r'\1', text)
    if text and not text[-1] in '.!?':
        text += '.'
    return text

def extract_sentences(text):
    """Simple sentence extraction using regex patterns"""
    if not text or not isinstance(text, str):
        return []

    text = clean_text(text)

    # Simple sentence splitting on periods, exclamation marks, and question marks
    # This is more basic than NLTK but should work for most cases
    sentences = re.split(r'[.!?]+', text)

    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Filter out very short fragments
            cleaned_sentences.append(sentence)

    return cleaned_sentences

def extract_words(text):
    """Simple word extraction using regex"""
    if not text or not isinstance(text, str):
        return []

    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    return words

def generate_sentence_id(sentence):
    hash_value = hashlib.md5(sentence.lower().encode()).hexdigest()[:8]
    return f'SENT_{hash_value}'

def generate_word_id(word):
    hash_value = hashlib.md5(word.lower().encode()).hexdigest()[:6]
    return f'WORD_{hash_value}'

def extract_text_from_content(content):
    """Extract text from ChatGPT export content structure"""
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        # Handle ChatGPT export format: content.parts[0] contains the text
        if 'parts' in content and isinstance(content['parts'], list):
            if len(content['parts']) > 0:
                part = content['parts'][0]
                if isinstance(part, str):
                    return part
                elif isinstance(part, dict) and 'text' in part:
                    return part['text']
        # Fallback for other formats
        elif 'text' in content:
            return content['text']
        elif 'content' in content:
            return extract_text_from_content(content['content'])
    return ''

def process_conversation(file_path, conversation_id, sentence_map, word_map):
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        messages = []
        if 'mapping' in data:
            for msg_id, msg_data in data['mapping'].items():
                if 'message' in msg_data and msg_data['message']:
                    message = msg_data['message']
                    content = message.get('content', {})

                    # Extract text from content using the correct structure
                    text = extract_text_from_content(content)

                    if text and text.strip():
                        messages.append({
                            'content': text,
                            'role': message.get('author', {}).get('role', 'unknown'),
                            'timestamp': message.get('create_time', ''),
                            'message_id': msg_id
                        })

        for message in messages:
            content = message.get('content', '')
            if not content:
                continue

            message_sentences = extract_sentences(content)

            for sentence in message_sentences:
                sentence_id = generate_sentence_id(sentence)

                if sentence_id not in sentence_map:
                    words = extract_words(sentence)
                    sentence_map[sentence_id] = {
                        'sentence_id': sentence_id,
                        'sentence': sentence,
                        'word_count': len(words),
                        'char_count': len(sentence),
                        'conversation_count': 1,
                        'first_seen': conversation_id
                    }

                    # Process words
                    for word in words:
                        if len(word) < 2:
                            continue
                        word_id = generate_word_id(word)

                        if word_id not in word_map:
                            word_map[word_id] = {
                                'word_id': word_id,
                                'word': word,
                                'frequency': 1,
                                'first_seen': conversation_id
                            }
                        else:
                            word_map[word_id]['frequency'] += 1
                else:
                    sentence_map[sentence_id]['conversation_count'] += 1

                sentences.append({
                    'sentence_id': sentence_id,
                    'conversation_id': conversation_id,
                    'message_role': message.get('role', 'unknown'),
                    'message_timestamp': message.get('timestamp', ''),
                    'message_id': message.get('message_id', ''),
                    'file_path': file_path
                })

    except Exception as e:
        print(f'Error processing {file_path}: {e}')

    return sentences

def load_conversation_ids(csv_path):
    conversation_ids = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                conversation_ids[row['filename']] = row['conversation_id']
    except Exception as e:
        print(f'Error loading conversation IDs: {e}')
    return conversation_ids

def main():
    parser = argparse.ArgumentParser(description='Generate Canonical Sentence IDs for ChatGPT conversations (SIMPLE VERSION)')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output directory for sentence and word IDs')

    args = parser.parse_args()

    print('🔍 Processing conversations for sentence and word ID generation (SIMPLE VERSION)...')

    os.makedirs(args.output, exist_ok=True)

    conversation_ids = load_conversation_ids(args.conversation_ids)

    sentence_map = {}
    word_map = {}
    all_sentence_occurrences = []

    processed_count = 0
    total_files = len([f for f in os.listdir(args.input) if f.endswith('.json')])

    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input, filename)
            conversation_id = conversation_ids.get(filename, f'UNKNOWN_{filename}')

            sentence_occurrences = process_conversation(file_path, conversation_id, sentence_map, word_map)
            all_sentence_occurrences.extend(sentence_occurrences)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f'Processed {processed_count}/{total_files} conversations...')

    print(f'✅ Processed all {processed_count} conversations!')

    # Initialize output file paths
    sentences_output = os.path.join(args.output, 'sentences_simple.csv')
    occurrences_output = os.path.join(args.output, 'sentence_occurrences_simple.csv')
    words_output = os.path.join(args.output, 'words_simple.csv')
    report_output = os.path.join(args.output, 'sentence_word_analysis_report_simple.json')

    # Write sentences
    if sentence_map:
        fieldnames = ['sentence_id', 'sentence', 'word_count', 'char_count', 'conversation_count', 'first_seen']

        with open(sentences_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sentence_data in sentence_map.values():
                row = {
                    'sentence_id': sentence_data['sentence_id'],
                    'sentence': sentence_data['sentence'],
                    'word_count': sentence_data['word_count'],
                    'char_count': sentence_data['char_count'],
                    'conversation_count': sentence_data['conversation_count'],
                    'first_seen': sentence_data['first_seen']
                }
                writer.writerow(row)

    # Write sentence occurrences
    if all_sentence_occurrences:
        fieldnames = ['sentence_id', 'conversation_id', 'message_role', 'message_timestamp', 'message_id', 'file_path']

        with open(occurrences_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_sentence_occurrences)

    # Write words
    if word_map:
        fieldnames = ['word_id', 'word', 'frequency', 'first_seen']

        with open(words_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(word_map.values())

    # Generate analysis report
    report_data = {
        'processing_date': datetime.now().isoformat(),
        'total_files_processed': processed_count,
        'total_sentences_found': len(all_sentence_occurrences),
        'unique_sentences': len(sentence_map),
        'total_words_found': sum(word_data['frequency'] for word_data in word_map.values()),
        'unique_words': len(word_map),
        'deduplication_ratio_sentences': round(1 - (len(sentence_map) / max(len(all_sentence_occurrences), 1)), 3),
        'deduplication_ratio_words': round(1 - (len(word_map) / max(sum(word_data['frequency'] for word_data in word_map.values()), 1)), 3),
        'top_10_sentences': sorted(sentence_map.values(), key=lambda x: x['conversation_count'], reverse=True)[:10],
        'top_10_words': sorted(word_map.values(), key=lambda x: x['frequency'], reverse=True)[:10]
    }

    with open(report_output, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f'\n✅ Sentence and Word ID generation complete!')
    print(f'📄 Sentences: {os.path.abspath(sentences_output)}')
    print(f'📄 Occurrences: {os.path.abspath(occurrences_output)}')
    print(f'📄 Words: {os.path.abspath(words_output)}')
    print(f'📊 Report: {os.path.abspath(report_output)}')
    print(f'📊 Generated {len(sentence_map)} unique sentences')
    print(f'📊 Generated {len(word_map)} unique words')
    print(f'📊 Found {len(all_sentence_occurrences)} sentence occurrences')

if __name__ == '__main__':
    main()
