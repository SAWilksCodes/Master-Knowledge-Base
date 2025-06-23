#!/usr/bin/env python3
"""
Enhanced Semantic Anchor System for ChatGPT Conversations

This script creates intelligent semantic anchors from conversations by:
1. Identifying key concepts, entities, and knowledge points
2. Creating semantic clusters and relationships
3. Generating anchor embeddings for intelligent retrieval
4. Building a knowledge graph structure
"""

import json
import csv
import os
import re
import hashlib
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️  spaCy model not found. Installing...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Semantic anchor patterns and categories
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

def extract_semantic_anchors(text: str, doc_id: str) -> List[Dict[str, Any]]:
    """Extract semantic anchors from text using NLP and pattern matching."""
    anchors = []

    # Process with spaCy
    doc = nlp(text)

    # Extract named entities
    for ent in doc.ents:
        anchors.append({
            'type': 'entity',
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'confidence': 0.9,
            'source': 'ner'
        })

    # Extract noun phrases and key concepts
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= 2:  # Multi-word phrases
            anchors.append({
                'type': 'concept',
                'text': chunk.text,
                'label': 'noun_phrase',
                'start': chunk.start_char,
                'end': chunk.end_char,
                'confidence': 0.7,
                'source': 'noun_chunk'
            })

    # Pattern-based extraction
    text_lower = text.lower()
    for category, patterns in SEMANTIC_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context
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

    # Extract technical terms (camelCase, snake_case, etc.)
    tech_patterns = [
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',  # camelCase
        r'\b[a-z]+(?:_[a-z]+)*\b',  # snake_case
        r'\b[A-Z]+(?:_[A-Z]+)*\b',  # UPPER_SNAKE
        r'\b[A-Z]{2,}\b'  # ACRONYMS
    ]

    for pattern in tech_patterns:
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

    return anchors

def create_anchor_embedding(anchor: Dict[str, Any], text: str) -> str:
    """Create a semantic embedding representation for an anchor."""
    # Create a rich representation including context
    context_start = max(0, anchor['start'] - 200)
    context_end = min(len(text), anchor['end'] + 200)
    context = text[context_start:context_end].strip()

    # Combine anchor text with context and metadata
    embedding_text = f"{anchor['text']} {anchor['type']} {anchor['label']} {context}"
    return embedding_text

def cluster_anchors(anchors: List[Dict[str, Any]], texts: List[str]) -> List[Dict[str, Any]]:
    """Cluster similar anchors using TF-IDF and DBSCAN."""
    if not anchors:
        return []

    # Create embeddings for clustering
    embeddings = []
    for anchor in anchors:
        text_idx = anchor.get('text_idx', 0)
        text = texts[text_idx] if text_idx < len(texts) else ""
        embedding_text = create_anchor_embedding(anchor, text)
        embeddings.append(embedding_text)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(embeddings)

        # Clustering with DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(tfidf_matrix)

        # Assign cluster labels
        for i, anchor in enumerate(anchors):
            anchor['cluster_id'] = int(clustering.labels_[i])

    except Exception as e:
        print(f"Clustering error: {e}")
        # Fallback: assign unique cluster IDs
        for i, anchor in enumerate(anchors):
            anchor['cluster_id'] = i

    return anchors

def generate_anchor_id(anchor: Dict[str, Any], index: int) -> str:
    """Generate unique anchor ID."""
    # Create hash from anchor text and type
    hash_input = f"{anchor['text']}_{anchor['type']}_{anchor.get('cluster_id', index)}"
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

def process_conversation(file_path: str, conversation_id: str) -> List[Dict[str, Any]]:
    """Process a single conversation file for semantic anchors."""
    anchors = []
    texts = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        mapping = data.get('mapping', {})
        if not mapping:
            return anchors

        # Process each message
        for message_id, message_data in mapping.items():
            if not message_data or 'message' not in message_data:
                continue

            message = message_data['message']
            if not message:
                continue

            # Extract content
            content = extract_message_content(message_data)
            if not content or not isinstance(content, str):
                continue

            texts.append(content)
            text_idx = len(texts) - 1

            # Extract anchors from message content
            message_anchors = extract_semantic_anchors(content, message_id)

            # Add metadata
            for anchor in message_anchors:
                anchor.update({
                    'conversation_id': conversation_id,
                    'message_id': message_id,
                    'text_idx': text_idx,
                    'file_path': file_path
                })

            anchors.extend(message_anchors)

        # Cluster anchors
        anchors = cluster_anchors(anchors, texts)

        # Generate IDs
        for i, anchor in enumerate(anchors):
            anchor['anchor_id'] = generate_anchor_id(anchor, i)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return anchors

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

def analyze_anchor_relationships(anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze relationships between anchors."""
    relationships = []

    # Group anchors by cluster
    clusters = defaultdict(list)
    for anchor in anchors:
        cluster_id = anchor.get('cluster_id', -1)
        clusters[cluster_id].append(anchor)

    # Find relationships within clusters
    for cluster_id, cluster_anchors in clusters.items():
        if len(cluster_anchors) > 1:
            for i, anchor1 in enumerate(cluster_anchors):
                for j, anchor2 in enumerate(cluster_anchors[i+1:], i+1):
                    # Calculate similarity
                    similarity = 0.0
                    if anchor1['text'].lower() == anchor2['text'].lower():
                        similarity = 1.0
                    elif anchor1['type'] == anchor2['type']:
                        similarity = 0.5

                    if similarity > 0:
                        relationships.append({
                            'anchor1_id': anchor1['anchor_id'],
                            'anchor2_id': anchor2['anchor_id'],
                            'relationship_type': 'similarity',
                            'strength': similarity,
                            'cluster_id': cluster_id
                        })

    return relationships

def main():
    parser = argparse.ArgumentParser(description='Generate Semantic Anchors for ChatGPT conversations')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output directory for semantic anchor results')

    args = parser.parse_args()

    print("🧠 Processing conversations for semantic anchor generation...")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load conversation IDs
    conversation_ids = load_conversation_ids(args.conversation_ids)

    all_anchors = []
    all_relationships = []
    processed_count = 0

    # Process each JSON file
    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input, filename)
            conversation_id = conversation_ids.get(filename, f"UNKNOWN_{filename}")

            anchors = process_conversation(file_path, conversation_id)
            all_anchors.extend(anchors)

            # Analyze relationships
            relationships = analyze_anchor_relationships(anchors)
            all_relationships.extend(relationships)

            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} conversations...")

    # Write results
    if all_anchors:
        # Write anchors CSV
        anchors_csv = os.path.join(args.output, 'semantic_anchors.csv')
        fieldnames = [
            'anchor_id', 'conversation_id', 'message_id', 'type', 'text',
            'label', 'confidence', 'source', 'cluster_id', 'context',
            'start', 'end', 'file_path'
        ]

        with open(anchors_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for anchor in all_anchors:
                # Clean anchor for CSV
                clean_anchor = {k: v for k, v in anchor.items() if k in fieldnames}
                writer.writerow(clean_anchor)

        # Write relationships CSV
        if all_relationships:
            relationships_csv = os.path.join(args.output, 'anchor_relationships.csv')
            rel_fieldnames = ['anchor1_id', 'anchor2_id', 'relationship_type', 'strength', 'cluster_id']

            with open(relationships_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rel_fieldnames)
                writer.writeheader()
                writer.writerows(all_relationships)

        # Generate summary
        summary_path = os.path.join(args.output, 'semantic_anchor_summary.md')
        generate_summary(all_anchors, all_relationships, summary_path, processed_count)

        print(f"\n✅ Semantic anchor generation complete!")
        print(f"📄 Anchors CSV: {os.path.abspath(anchors_csv)}")
        if all_relationships:
            print(f"📄 Relationships CSV: {os.path.abspath(relationships_csv)}")
        print(f"📄 Summary: {os.path.abspath(summary_path)}")
        print(f"📊 Generated {len(all_anchors)} semantic anchors")
        print(f"📊 Generated {len(all_relationships)} anchor relationships")
    else:
        print("❌ No semantic anchors found in conversations")

def generate_summary(anchors: List[Dict[str, Any]], relationships: List[Dict[str, Any]],
                    output_path: str, total_conversations: int):
    """Generate summary markdown file."""
    if not anchors:
        return

    # Count by type
    type_counts = Counter(anchor['type'] for anchor in anchors)
    source_counts = Counter(anchor['source'] for anchor in anchors)
    cluster_counts = Counter(anchor.get('cluster_id', -1) for anchor in anchors)

    # Find most common anchors
    text_counts = Counter(anchor['text'].lower() for anchor in anchors)
    most_common = text_counts.most_common(20)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Semantic Anchor Generation Summary\n\n")
        f.write(f"Total anchors generated: {len(anchors)}\n")
        f.write(f"Total relationships: {len(relationships)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `ANCHOR_TYPE_INDEX_HASH`\n")
        f.write("- TYPE: Anchor type (concept, entity, technology, etc.)\n")
        f.write("- INDEX: Sequential index\n")
        f.write("- HASH: 8-character hash of anchor content\n\n")

        f.write("## Anchor Type Distribution\n\n")
        for anchor_type, count in type_counts.most_common():
            percentage = (count / len(anchors)) * 100
            f.write(f"- **{anchor_type.title()}**: {count} anchors ({percentage:.1f}%)\n")

        f.write("\n## Source Distribution\n\n")
        for source, count in source_counts.most_common():
            percentage = (count / len(anchors)) * 100
            f.write(f"- **{source.title()}**: {count} anchors ({percentage:.1f}%)\n")

        f.write(f"\n## Clustering Results\n\n")
        f.write(f"Total clusters: {len(cluster_counts)}\n")
        f.write(f"Clustered anchors: {sum(count for cluster_id, count in cluster_counts.items() if cluster_id >= 0)}\n")
        f.write(f"Unclustered anchors: {cluster_counts.get(-1, 0)}\n\n")

        f.write("## Most Common Anchors\n\n")
        for text, count in most_common:
            f.write(f"- **{text}**: {count} occurrences\n")

        f.write("\n## Recent Anchors\n\n")
        recent_anchors = anchors[-10:]  # Last 10 anchors
        for anchor in recent_anchors:
            context_preview = anchor.get('context', '')[:100] + '...' if anchor.get('context', '') else 'No context'
            f.write(f"- **{anchor['anchor_id']}**: {anchor['text']} ({anchor['type']}) - {context_preview}\n")

if __name__ == "__main__":
    main()
