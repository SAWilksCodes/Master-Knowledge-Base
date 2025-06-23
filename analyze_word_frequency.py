#!/usr/bin/env python3
import json
import csv
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import re
import itertools

# Basic POS patterns (simplified without external dependencies)
POS_PATTERNS = {
    'noun': [r'\b\w+ing\b', r'\b\w+tion\b', r'\b\w+ness\b', r'\b\w+ment\b', r'\b\w+ity\b'],
    'verb': [r'\b\w+ed\b', r'\b\w+ing\b', r'\b\w+ize\b', r'\b\w+ify\b'],
    'adjective': [r'\b\w+ful\b', r'\b\w+less\b', r'\b\w+able\b', r'\b\w+ive\b'],
    'adverb': [r'\b\w+ly\b']
}

# Emotional tone keywords
EMOTIONAL_TONE = {
    'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'perfect', 'brilliant', 'outstanding'],
    'negative': ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrating', 'annoying', 'worst', 'failed', 'broken'],
    'neutral': ['okay', 'fine', 'normal', 'standard', 'regular', 'typical', 'average', 'usual', 'common', 'basic'],
    'technical': ['function', 'method', 'class', 'variable', 'parameter', 'algorithm', 'data', 'system', 'process', 'structure'],
    'question': ['how', 'what', 'why', 'when', 'where', 'which', 'who', 'can', 'could', 'would', 'should', 'might']
}

def load_word_data(csv_path):
    """Load word frequency data from CSV"""
    words = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                words.append({
                    'word_id': row['word_id'],
                    'word': row['word'],
                    'frequency': int(row['frequency']),
                    'first_seen': row['first_seen']
                })
    except Exception as e:
        print(f'Error loading word data: {e}')
    return words

def classify_pos(word):
    """Simple POS classification based on patterns"""
    word_lower = word.lower()

    for pos, patterns in POS_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, word_lower):
                return pos

    # Common word classifications
    if word_lower in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
        return 'determiner'
    elif word_lower in ['and', 'or', 'but', 'so', 'because', 'if', 'when', 'while']:
        return 'conjunction'
    elif word_lower in ['in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'of']:
        return 'preposition'
    elif word_lower in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']:
        return 'pronoun'
    elif word_lower in ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had']:
        return 'auxiliary'
    else:
        return 'unknown'

def classify_emotional_tone(word):
    """Classify word's emotional tone"""
    word_lower = word.lower()

    for tone, words in EMOTIONAL_TONE.items():
        if word_lower in words:
            return tone

    return 'neutral'

def generate_co_occurrence_graph(words, sentence_occurrences_path, top_n=500):
    """Generate co-occurrence graph from sentence data"""
    print('📊 Generating co-occurrence graph...')

    # Get top N words
    top_words = sorted(words, key=lambda x: x['frequency'], reverse=True)[:top_n]
    top_word_set = {w['word'] for w in top_words}

    co_occurrences = defaultdict(int)

    try:
        with open(sentence_occurrences_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            processed = 0

            for row in reader:
                # This is a simplified approach - in reality we'd need to parse sentences
                # For now, we'll create a basic co-occurrence based on word proximity
                processed += 1
                if processed % 10000 == 0:
                    print(f'Processed {processed} sentence occurrences...')

                # Skip for now - would need actual sentence text to do proper co-occurrence
                pass

    except Exception as e:
        print(f'Error processing sentence occurrences: {e}')

    return dict(co_occurrences)

def analyze_top_words(words, top_n=500):
    """Analyze top N words for POS and emotional tone"""
    print(f'🔍 Analyzing top {top_n} words...')

    top_words = sorted(words, key=lambda x: x['frequency'], reverse=True)[:top_n]

    analyzed_words = []
    pos_stats = Counter()
    tone_stats = Counter()

    for word_data in top_words:
        word = word_data['word']
        pos = classify_pos(word)
        tone = classify_emotional_tone(word)

        analyzed_word = {
            'word_id': word_data['word_id'],
            'word': word,
            'frequency': word_data['frequency'],
            'pos': pos,
            'emotional_tone': tone,
            'first_seen': word_data['first_seen'],
            'rank': len(analyzed_words) + 1
        }

        analyzed_words.append(analyzed_word)
        pos_stats[pos] += 1
        tone_stats[tone] += 1

    return analyzed_words, dict(pos_stats), dict(tone_stats)

def create_semantic_clusters(analyzed_words):
    """Create basic semantic clusters based on word patterns"""
    print('🔗 Creating semantic clusters...')

    clusters = defaultdict(list)

    for word_data in analyzed_words:
        word = word_data['word']
        pos = word_data['pos']
        tone = word_data['emotional_tone']

        # Cluster by POS and tone
        cluster_key = f"{pos}_{tone}"
        clusters[cluster_key].append(word_data)

        # Special clusters for high-frequency technical terms
        if word_data['frequency'] > 1000 and pos == 'unknown':
            if any(tech_word in word for tech_word in ['data', 'system', 'process', 'method']):
                clusters['technical_high_freq'].append(word_data)

    return dict(clusters)

def main():
    parser = argparse.ArgumentParser(description='Analyze word frequency and create tag graph genesis')
    parser.add_argument('-w', '--words', required=True, help='Words CSV file from sentence analysis')
    parser.add_argument('-s', '--sentences', help='Sentence occurrences CSV file (optional)')
    parser.add_argument('-o', '--output', required=True, help='Output directory for analysis results')
    parser.add_argument('-n', '--top-n', type=int, default=500, help='Number of top words to analyze')

    args = parser.parse_args()

    print('🔍 Starting word frequency analysis for tag graph genesis...')

    os.makedirs(args.output, exist_ok=True)

    # Load word data
    print('📖 Loading word frequency data...')
    words = load_word_data(args.words)

    if not words:
        print('❌ No word data found!')
        return

    print(f'✅ Loaded {len(words)} words')

    # Analyze top words
    analyzed_words, pos_stats, tone_stats = analyze_top_words(words, args.top_n)

    # Create semantic clusters
    clusters = create_semantic_clusters(analyzed_words)

    # Generate co-occurrence graph (basic version)
    co_occurrences = {}
    if args.sentences:
        co_occurrences = generate_co_occurrence_graph(words, args.sentences, args.top_n)

    # Write analyzed words
    analyzed_output = os.path.join(args.output, 'words_analyzed.csv')
    fieldnames = ['word_id', 'word', 'frequency', 'rank', 'pos', 'emotional_tone', 'first_seen']

    with open(analyzed_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analyzed_words)

    # Write semantic clusters
    clusters_output = os.path.join(args.output, 'semantic_clusters.json')
    with open(clusters_output, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    # Generate comprehensive report
    report_data = {
        'processing_date': datetime.now().isoformat(),
        'total_words_analyzed': len(words),
        'top_words_analyzed': len(analyzed_words),
        'pos_distribution': pos_stats,
        'emotional_tone_distribution': tone_stats,
        'semantic_clusters_count': len(clusters),
        'top_10_words': analyzed_words[:10],
        'cluster_summary': {cluster: len(words) for cluster, words in clusters.items()},
        'analysis_parameters': {
            'top_n': args.top_n,
            'pos_patterns_used': list(POS_PATTERNS.keys()),
            'emotional_tones_used': list(EMOTIONAL_TONE.keys())
        }
    }

    report_output = os.path.join(args.output, 'word_analysis_report.json')
    with open(report_output, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Write co-occurrence data if available
    if co_occurrences:
        cooccur_output = os.path.join(args.output, 'word_cooccurrences.json')
        with open(cooccur_output, 'w', encoding='utf-8') as f:
            json.dump(co_occurrences, f, indent=2, ensure_ascii=False)

    print(f'\n✅ Word frequency analysis complete!')
    print(f'📄 Analyzed words: {os.path.abspath(analyzed_output)}')
    print(f'🔗 Semantic clusters: {os.path.abspath(clusters_output)}')
    print(f'📊 Analysis report: {os.path.abspath(report_output)}')
    print(f'\n📈 POS Distribution: {pos_stats}')
    print(f'😊 Tone Distribution: {tone_stats}')
    print(f'🏷️  Semantic Clusters: {len(clusters)} clusters created')

if __name__ == '__main__':
    main()
