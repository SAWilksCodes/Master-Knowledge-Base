#!/usr/bin/env python3
"""
Generate Concept IDs for ChatGPT Conversations

This script identifies key concepts and ideas in conversations and assigns unique IDs.
Concepts are identified through keyword extraction, frequency analysis, and semantic grouping.
"""

import json
import csv
import os
import re
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Concept categories and keywords
CONCEPT_CATEGORIES = {
    'technology': [
        'ai', 'machine learning', 'neural network', 'algorithm', 'software', 'hardware',
        'programming', 'coding', 'development', 'database', 'api', 'framework',
        'automation', 'robotics', 'blockchain', 'cloud', 'cybersecurity'
    ],
    'business': [
        'strategy', 'marketing', 'sales', 'revenue', 'profit', 'business model',
        'startup', 'entrepreneurship', 'management', 'leadership', 'team',
        'product', 'service', 'customer', 'market', 'competition'
    ],
    'science': [
        'research', 'experiment', 'hypothesis', 'theory', 'data', 'analysis',
        'methodology', 'study', 'investigation', 'discovery', 'innovation',
        'physics', 'chemistry', 'biology', 'mathematics', 'statistics'
    ],
    'education': [
        'learning', 'teaching', 'education', 'course', 'tutorial', 'study',
        'knowledge', 'skill', 'expertise', 'training', 'curriculum',
        'academic', 'scholarly', 'research', 'degree', 'certification'
    ],
    'creative': [
        'design', 'art', 'creativity', 'imagination', 'inspiration', 'style',
        'aesthetic', 'visual', 'graphic', 'creative process', 'artistic',
        'composition', 'color', 'form', 'texture', 'pattern'
    ],
    'health': [
        'health', 'wellness', 'fitness', 'nutrition', 'medical', 'therapy',
        'mental health', 'physical', 'exercise', 'diet', 'lifestyle',
        'prevention', 'treatment', 'recovery', 'wellbeing'
    ],
    'finance': [
        'money', 'finance', 'investment', 'budget', 'financial', 'economy',
        'banking', 'credit', 'debt', 'savings', 'retirement', 'insurance',
        'trading', 'portfolio', 'asset', 'liability'
    ],
    'social': [
        'community', 'society', 'culture', 'social', 'relationship', 'communication',
        'network', 'collaboration', 'teamwork', 'partnership', 'interaction',
        'social media', 'connection', 'engagement', 'participation'
    ],
    'environment': [
        'environment', 'sustainability', 'climate', 'ecology', 'nature',
        'conservation', 'green', 'renewable', 'energy', 'pollution',
        'biodiversity', 'ecosystem', 'carbon', 'recycling'
    ],
    'philosophy': [
        'philosophy', 'ethics', 'morality', 'values', 'belief', 'principle',
        'wisdom', 'knowledge', 'truth', 'reality', 'existence', 'meaning',
        'purpose', 'consciousness', 'mind', 'spirituality'
    ]
}

# Concept extraction patterns
CONCEPT_PATTERNS = {
    'definition': r'\b(define|definition|means|refers to|is a|are)\b',
    'explanation': r'\b(explain|describe|elaborate|clarify|detail|break down)\b',
    'example': r'\b(example|instance|case|scenario|illustration|demonstration)\b',
    'comparison': r'\b(compare|contrast|similar|different|versus|vs|like|unlike)\b',
    'process': r'\b(process|procedure|method|approach|technique|strategy)\b',
    'problem': r'\b(problem|issue|challenge|difficulty|obstacle|barrier)\b',
    'solution': r'\b(solution|answer|resolve|fix|address|overcome)\b'
}

def extract_concepts(text: str) -> List[Dict[str, Any]]:
    """Extract concepts from text using keyword analysis and pattern matching."""
    concepts = []
    text_lower = text.lower()

    # Tokenize and clean text
    sentences = sent_tokenize(text)
    words = word_tokenize(text_lower)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]

    # Count word frequencies
    word_freq = Counter(words)

    # Check each concept category
    for category, keywords in CONCEPT_CATEGORIES.items():
        matches = []
        for keyword in keywords:
            if keyword in text_lower:
                matches.append(keyword)

        if matches:
            # Calculate concept strength
            strength = len(matches)

            # Check for concept patterns
            pattern_matches = []
            for pattern_name, pattern in CONCEPT_PATTERNS.items():
                if re.search(pattern, text_lower):
                    pattern_matches.append(pattern_name)

            if pattern_matches:
                strength += len(pattern_matches)

            # Extract context around concept mentions
            context_sentences = []
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in matches):
                    context_sentences.append(sentence.strip())

            context = ' '.join(context_sentences[:3])  # Limit to 3 sentences

            concepts.append({
                'category': category,
                'keywords': matches,
                'strength': strength,
                'context': context,
                'patterns': pattern_matches,
                'word_frequency': dict(word_freq.most_common(10))
            })

    return concepts

def generate_concept_id(conversation_id: str, concept: Dict[str, Any], index: int) -> str:
    """Generate unique concept ID."""
    # Create hash from conversation ID, category, and keywords
    hash_input = f"{conversation_id}_{concept['category']}_{'_'.join(concept['keywords'][:3])}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"CONCEPT_{conversation_id}_{concept['category'].upper()}_{index:03d}_{hash_value}"

def process_conversation(file_path: str, conversation_id: str) -> List[Dict[str, Any]]:
    """Process a single conversation file for concepts."""
    concepts = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process each message
        for message in data.get('messages', []):
            content = message.get('content', '')
            if not content or not isinstance(content, str):
                continue

            # Extract concepts from message content
            message_concepts = extract_concepts(content)

            for i, concept in enumerate(message_concepts):
                concept_id = generate_concept_id(conversation_id, concept, len(concepts))

                concepts.append({
                    'concept_id': concept_id,
                    'conversation_id': conversation_id,
                    'category': concept['category'],
                    'keywords': '; '.join(concept['keywords']),
                    'strength': concept['strength'],
                    'context': concept['context'],
                    'patterns': '; '.join(concept['patterns']),
                    'message_index': data['messages'].index(message),
                    'role': message.get('role', 'unknown'),
                    'timestamp': message.get('timestamp', ''),
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
            'concept_id', 'conversation_id', 'category', 'keywords', 'strength',
            'context', 'patterns', 'message_index', 'role', 'timestamp', 'file_path'
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
    pattern_counts = defaultdict(int)

    for concept in concepts:
        category_counts[concept['category']] += 1
        strength_distribution[concept['strength']] += 1

        if concept['patterns']:
            for pattern in concept['patterns'].split('; '):
                pattern_counts[pattern] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Concept ID Generation Summary\n\n")
        f.write(f"Total concepts generated: {len(concepts)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `CONCEPT_CONVID_CATEGORY_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- CATEGORY: Concept category\n")
        f.write("- INDEX: Concept index within conversation\n")
        f.write("- HASH: 6-character hash of keywords\n\n")

        f.write("## Category Distribution\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(concepts)) * 100
            f.write(f"- **{category.title()}**: {count} concepts ({percentage:.1f}%)\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(concepts)) * 100
            f.write(f"- **Strength {strength}**: {count} concepts ({percentage:.1f}%)\n")

        f.write("\n## Pattern Distribution\n\n")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(concepts)) * 100
            f.write(f"- **{pattern.title()}**: {count} concepts ({percentage:.1f}%)\n")

        f.write("\n## Recent Concepts\n\n")
        recent_concepts = sorted(concepts, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        for concept in recent_concepts:
            f.write(f"- **{concept['concept_id']}**: {concept['category']} - {concept['keywords'][:100]}...\n")

if __name__ == "__main__":
    main()
