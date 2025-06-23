#!/usr/bin/env python3
import json
import csv
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import spacy

# Try to load spaCy model, with fallback instructions
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except OSError:
    print("❌ spaCy model not found. Please install with:")
    print("pip install spacy")
    print("python -m spacy download en_core_web_sm")
    exit(1)

# Enhanced emotional tone categories based on Plutchik's wheel
ENHANCED_EMOTIONAL_TONE = {
    # Primary emotions
    'joy': ['happy', 'joyful', 'cheerful', 'delighted', 'pleased', 'content', 'satisfied', 'glad', 'elated', 'euphoric'],
    'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'sorrowful', 'dejected', 'downcast', 'gloomy', 'miserable'],
    'anger': ['angry', 'furious', 'enraged', 'irritated', 'annoyed', 'frustrated', 'outraged', 'livid', 'irate'],
    'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'apprehensive', 'panicked', 'frightened'],
    'trust': ['trust', 'confident', 'secure', 'reliable', 'dependable', 'faithful', 'loyal', 'certain', 'assured'],
    'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'appalled', 'horrified'],
    'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'stunned', 'bewildered', 'perplexed'],
    'anticipation': ['excited', 'eager', 'hopeful', 'expectant', 'optimistic', 'enthusiastic', 'anticipating'],

    # Secondary emotions (combinations)
    'love': ['love', 'adore', 'cherish', 'treasure', 'devoted', 'passionate', 'romantic', 'affectionate'],
    'submission': ['submissive', 'humble', 'modest', 'respectful', 'deferential', 'obedient'],
    'awe': ['awe', 'wonder', 'amazement', 'reverence', 'admiration', 'respect', 'veneration'],
    'disapproval': ['disapprove', 'condemn', 'criticize', 'reject', 'scorn', 'disdain'],
    'remorse': ['regret', 'sorry', 'remorseful', 'guilty', 'ashamed', 'repentant'],
    'contempt': ['contempt', 'scorn', 'disdain', 'disgust', 'loathing', 'hatred'],
    'aggressiveness': ['aggressive', 'hostile', 'combative', 'confrontational', 'belligerent'],
    'optimism': ['optimistic', 'hopeful', 'positive', 'upbeat', 'confident', 'encouraging'],

    # Tertiary emotions (more nuanced)
    'pride': ['proud', 'accomplished', 'successful', 'triumphant', 'victorious', 'satisfied'],
    'shame': ['ashamed', 'embarrassed', 'humiliated', 'mortified', 'disgraced'],
    'curiosity': ['curious', 'interested', 'intrigued', 'wondering', 'questioning', 'inquisitive'],
    'confusion': ['confused', 'puzzled', 'perplexed', 'bewildered', 'baffled', 'unclear'],
    'determination': ['determined', 'resolute', 'persistent', 'tenacious', 'steadfast', 'committed'],
    'skepticism': ['skeptical', 'doubtful', 'uncertain', 'questioning', 'suspicious', 'dubious'],
    'excitement': ['excited', 'thrilled', 'exhilarated', 'energetic', 'animated', 'enthusiastic'],
    'frustration': ['frustrated', 'annoyed', 'irritated', 'exasperated', 'impatient', 'vexed'],
    'relief': ['relieved', 'reassured', 'comforted', 'calmed', 'soothed', 'peaceful'],
    'boredom': ['bored', 'uninterested', 'apathetic', 'indifferent', 'disengaged', 'listless'],

    # Technical/analytical states
    'analytical': ['analyze', 'examine', 'evaluate', 'assess', 'investigate', 'study', 'research'],
    'creative': ['creative', 'innovative', 'imaginative', 'artistic', 'inventive', 'original'],
    'focused': ['focused', 'concentrated', 'attentive', 'dedicated', 'absorbed', 'engaged'],
    'methodical': ['systematic', 'organized', 'structured', 'methodical', 'logical', 'rational'],

    # Neutral/baseline
    'neutral': ['okay', 'fine', 'normal', 'standard', 'regular', 'typical', 'average', 'usual', 'common', 'basic']
}

# Technical domain classifications
TECHNICAL_DOMAINS = {
    'programming': ['code', 'function', 'class', 'method', 'variable', 'algorithm', 'syntax', 'debug', 'compile'],
    'ai_ml': ['model', 'training', 'neural', 'learning', 'prediction', 'classification', 'embedding', 'transformer'],
    'data_science': ['data', 'analysis', 'statistics', 'visualization', 'dataset', 'metric', 'correlation'],
    'web_dev': ['html', 'css', 'javascript', 'frontend', 'backend', 'api', 'server', 'database'],
    'infrastructure': ['cloud', 'deployment', 'docker', 'kubernetes', 'server', 'network', 'security'],
    'business': ['strategy', 'market', 'customer', 'revenue', 'growth', 'profit', 'investment', 'startup'],
    'design': ['design', 'visual', 'aesthetic', 'layout', 'typography', 'color', 'brand', 'interface'],
    'research': ['research', 'study', 'experiment', 'hypothesis', 'methodology', 'analysis', 'findings']
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

def enhanced_pos_classification(word):
    """Enhanced POS classification using spaCy"""
    doc = nlp(word)
    if len(doc) > 0:
        token = doc[0]
        return {
            'pos': token.pos_,
            'tag': token.tag_,
            'lemma': token.lemma_,
            'is_alpha': token.is_alpha,
            'is_stop': token.is_stop,
            'is_punct': token.is_punct,
            'shape': token.shape_
        }
    return {
        'pos': 'UNKNOWN',
        'tag': 'UNKNOWN',
        'lemma': word,
        'is_alpha': False,
        'is_stop': False,
        'is_punct': False,
        'shape': 'UNKNOWN'
    }

def enhanced_emotional_classification(word):
    """Enhanced emotional classification with 25+ categories"""
    word_lower = word.lower()

    for emotion, keywords in ENHANCED_EMOTIONAL_TONE.items():
        if word_lower in keywords:
            return emotion

    return 'neutral'

def technical_domain_classification(word):
    """Classify word into technical domains"""
    word_lower = word.lower()

    for domain, keywords in TECHNICAL_DOMAINS.items():
        if word_lower in keywords:
            return domain

    return 'general'

def analyze_words_enhanced(words, top_n=1000):
    """Enhanced analysis of top N words"""
    print(f'🔍 Enhanced analysis of top {top_n} words...')

    top_words = sorted(words, key=lambda x: x['frequency'], reverse=True)[:top_n]

    analyzed_words = []
    pos_stats = Counter()
    emotion_stats = Counter()
    domain_stats = Counter()

    # Process in batches for better performance
    batch_size = 100
    for i in range(0, len(top_words), batch_size):
        batch = top_words[i:i+batch_size]
        print(f'Processing batch {i//batch_size + 1}/{(len(top_words)-1)//batch_size + 1}...')

        for word_data in batch:
            word = word_data['word']

            # Enhanced classifications
            pos_info = enhanced_pos_classification(word)
            emotion = enhanced_emotional_classification(word)
            domain = technical_domain_classification(word)

            analyzed_word = {
                'word_id': word_data['word_id'],
                'word': word,
                'frequency': word_data['frequency'],
                'rank': len(analyzed_words) + 1,
                'pos': pos_info['pos'],
                'pos_tag': pos_info['tag'],
                'lemma': pos_info['lemma'],
                'is_alpha': pos_info['is_alpha'],
                'is_stop': pos_info['is_stop'],
                'emotional_tone': emotion,
                'technical_domain': domain,
                'first_seen': word_data['first_seen']
            }

            analyzed_words.append(analyzed_word)
            pos_stats[pos_info['pos']] += 1
            emotion_stats[emotion] += 1
            domain_stats[domain] += 1

    return analyzed_words, dict(pos_stats), dict(emotion_stats), dict(domain_stats)

def create_enhanced_clusters(analyzed_words):
    """Create enhanced semantic clusters"""
    print('🔗 Creating enhanced semantic clusters...')

    clusters = defaultdict(list)

    for word_data in analyzed_words:
        pos = word_data['pos']
        emotion = word_data['emotional_tone']
        domain = word_data['technical_domain']

        # Multi-dimensional clustering
        clusters[f"pos_{pos}"].append(word_data)
        clusters[f"emotion_{emotion}"].append(word_data)
        clusters[f"domain_{domain}"].append(word_data)
        clusters[f"pos_{pos}_emotion_{emotion}"].append(word_data)
        clusters[f"domain_{domain}_emotion_{emotion}"].append(word_data)

        # High-frequency clusters
        if word_data['frequency'] > 10000:
            clusters['high_frequency'].append(word_data)
        elif word_data['frequency'] > 1000:
            clusters['medium_frequency'].append(word_data)

        # Special clusters for content words (non-stop words)
        if not word_data['is_stop'] and word_data['is_alpha']:
            clusters['content_words'].append(word_data)

    return dict(clusters)

def main():
    parser = argparse.ArgumentParser(description='Enhanced word analysis with proper POS tagging and emotion classification')
    parser.add_argument('-w', '--words', required=True, help='Words CSV file from sentence analysis')
    parser.add_argument('-o', '--output', required=True, help='Output directory for enhanced analysis')
    parser.add_argument('-n', '--top-n', type=int, default=1000, help='Number of top words to analyze')

    args = parser.parse_args()

    print('🚀 Starting enhanced word analysis with spaCy and rich emotion taxonomy...')

    os.makedirs(args.output, exist_ok=True)

    # Load word data
    print('📖 Loading word frequency data...')
    words = load_word_data(args.words)

    if not words:
        print('❌ No word data found!')
        return

    print(f'✅ Loaded {len(words)} words')

    # Enhanced analysis
    analyzed_words, pos_stats, emotion_stats, domain_stats = analyze_words_enhanced(words, args.top_n)

    # Create enhanced clusters
    clusters = create_enhanced_clusters(analyzed_words)

    # Write enhanced analyzed words
    enhanced_output = os.path.join(args.output, 'words_enhanced_analysis.csv')
    fieldnames = ['word_id', 'word', 'frequency', 'rank', 'pos', 'pos_tag', 'lemma',
                  'is_alpha', 'is_stop', 'emotional_tone', 'technical_domain', 'first_seen']

    with open(enhanced_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analyzed_words)

    # Write enhanced clusters
    enhanced_clusters_output = os.path.join(args.output, 'enhanced_semantic_clusters.json')
    with open(enhanced_clusters_output, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    # Generate comprehensive enhanced report
    enhanced_report_data = {
        'processing_date': datetime.now().isoformat(),
        'total_words_analyzed': len(words),
        'enhanced_words_analyzed': len(analyzed_words),
        'pos_distribution': pos_stats,
        'emotional_tone_distribution': emotion_stats,
        'technical_domain_distribution': domain_stats,
        'enhanced_clusters_count': len(clusters),
        'top_10_words': analyzed_words[:10],
        'cluster_summary': {cluster: len(words) for cluster, words in clusters.items()},
        'emotion_categories_used': list(ENHANCED_EMOTIONAL_TONE.keys()),
        'technical_domains_used': list(TECHNICAL_DOMAINS.keys()),
        'analysis_improvements': {
            'pos_tagging': 'spaCy en_core_web_sm model',
            'emotion_classification': 'Plutchik-inspired 25+ categories',
            'technical_domains': '8 specialized domains',
            'clustering': 'Multi-dimensional semantic clustering'
        }
    }

    enhanced_report_output = os.path.join(args.output, 'enhanced_word_analysis_report.json')
    with open(enhanced_report_output, 'w', encoding='utf-8') as f:
        json.dump(enhanced_report_data, f, indent=2, ensure_ascii=False)

    print(f'\n✅ Enhanced word analysis complete!')
    print(f'📄 Enhanced analysis: {os.path.abspath(enhanced_output)}')
    print(f'🔗 Enhanced clusters: {os.path.abspath(enhanced_clusters_output)}')
    print(f'📊 Enhanced report: {os.path.abspath(enhanced_report_output)}')
    print(f'\n📈 POS Distribution: {dict(Counter(pos_stats).most_common(10))}')
    print(f'😊 Emotion Distribution: {dict(Counter(emotion_stats).most_common(10))}')
    print(f'🔧 Domain Distribution: {dict(Counter(domain_stats).most_common(10))}')
    print(f'🏷️  Enhanced Clusters: {len(clusters)} multi-dimensional clusters')

if __name__ == '__main__':
    main()
