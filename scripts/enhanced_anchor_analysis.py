#!/usr/bin/env python3
"""
Enhanced Semantic Anchor Analysis
Implements noise filtering, high-value tagging, and top cluster analysis
"""

import pandas as pd
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import argparse
import os
from datetime import datetime

# High-frequency stopwords to filter (LOW_SIGNAL)
STOPWORDS = {
    'n', 'and', 'the', 'to', 'a', 'for', 'or', 'of', 'in', 'with', 'you', 'is', 'on',
    'it', 'that', 'this', 'but', 'they', 'have', 'had', 'what', 'said', 'each', 'which',
    'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many',
    'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him',
    'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been',
    'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come',
    'made', 'may', 'part', 'over', 'new', 'sound', 'take', 'only', 'little', 'work',
    'know', 'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very', 'after',
    'thing', 'our', 'just', 'name', 'good', 'sentence', 'man', 'think', 'say', 'great',
    'where', 'help', 'through', 'much', 'before', 'line', 'right', 'too', 'mean', 'old',
    'any', 'same', 'tell', 'boy', 'follow', 'came', 'want', 'show', 'also', 'around',
    'form', 'three', 'small', 'set', 'put', 'end', 'does', 'another', 'well', 'large',
    'must', 'big', 'even', 'such', 'because', 'turn', 'here', 'why', 'ask', 'went',
    'men', 'read', 'need', 'land', 'different', 'home', 'us', 'move', 'try', 'kind',
    'hand', 'picture', 'again', 'change', 'off', 'play', 'spell', 'air', 'away', 'animal',
    'house', 'point', 'page', 'letter', 'mother', 'answer', 'found', 'study', 'still',
    'learn', 'should', 'America', 'world', 'high', 'every', 'near', 'add', 'food',
    'between', 'own', 'below', 'country', 'plant', 'last', 'school', 'father', 'keep',
    'tree', 'never', 'start', 'city', 'earth', 'eye', 'light', 'thought', 'head', 'under',
    'story', 'saw', 'left', 'don\'t', 'few', 'while', 'along', 'might', 'close', 'something',
    'seem', 'next', 'hard', 'open', 'example', 'begin', 'life', 'always', 'those', 'both',
    'paper', 'together', 'got', 'group', 'often', 'run', 'important', 'until', 'children',
    'side', 'feet', 'car', 'mile', 'night', 'walk', 'white', 'sea', 'began', 'grow',
    'took', 'river', 'four', 'carry', 'state', 'once', 'book', 'hear', 'stop', 'without',
    'second', 'late', 'miss', 'idea', 'enough', 'eat', 'face', 'watch', 'far', 'Indian',
    'real', 'almost', 'let', 'above', 'girl', 'sometimes', 'mountain', 'cut', 'young',
    'talk', 'soon', 'list', 'song', 'being', 'leave', 'family', 'it\'s', 'body', 'music',
    'color', 'stand', 'sun', 'questions', 'fish', 'area', 'mark', 'dog', 'horse', 'birds',
    'problem', 'complete', 'room', 'knew', 'since', 'ever', 'piece', 'told', 'usually',
    'didn\'t', 'friends', 'easy', 'heard', 'order', 'red', 'door', 'sure', 'become',
    'top', 'ship', 'across', 'today', 'during', 'short', 'better', 'best', 'however',
    'low', 'hours', 'black', 'products', 'happened', 'whole', 'measure', 'remember',
    'early', 'waves', 'reached', 'listen', 'wind', 'rock', 'space', 'covered', 'fast',
    'several', 'hold', 'himself', 'toward', 'five', 'step', 'morning', 'passed', 'vowel',
    'true', 'hundred', 'against', 'pattern', 'numeral', 'table', 'north', 'slowly',
    'money', 'map', 'farm', 'pulled', 'draw', 'voice', 'seen', 'cold', 'cried', 'plan',
    'notice', 'south', 'sing', 'war', 'ground', 'fall', 'king', 'town', 'I\'ll', 'unit',
    'figure', 'certain', 'field', 'travel', 'wood', 'fire', 'upon'
}

# High-value semantic terms for RAG prioritization
HIGH_VALUE_TERMS = {
    'llm', 'api', 'gpt', 'ai', 'model', 'system', 'integration', 'optimization',
    'deployment', 'framework', 'architecture', 'algorithm', 'database', 'server',
    'cloud', 'nlp', 'machine learning', 'deep learning', 'neural network', 'vector',
    'embedding', 'rag', 'retrieval', 'orchestration', 'pipeline', 'workflow', 'automation',
    'microservice', 'container', 'kubernetes', 'docker', 'aws', 'azure', 'gcp', 'api',
    'rest', 'graphql', 'websocket', 'authentication', 'authorization', 'security',
    'encryption', 'hashing', 'tokenization', 'semantic', 'ontology', 'knowledge graph',
    'entity', 'concept', 'relationship', 'inference', 'reasoning', 'logic', 'constraint',
    'validation', 'testing', 'monitoring', 'logging', 'metrics', 'analytics', 'dashboard',
    'visualization', 'reporting', 'business intelligence', 'data science', 'statistics',
    'probability', 'bayesian', 'regression', 'classification', 'clustering', 'anomaly',
    'outlier', 'pattern', 'trend', 'forecast', 'prediction', 'recommendation', 'ranking',
    'search', 'index', 'query', 'filter', 'sort', 'aggregate', 'transform', 'extract',
    'load', 'etl', 'data warehouse', 'data lake', 'streaming', 'batch', 'real-time',
    'asynchronous', 'synchronous', 'concurrent', 'parallel', 'distributed', 'scalable',
    'fault-tolerant', 'resilient', 'redundant', 'backup', 'recovery', 'disaster',
    'high-availability', 'load-balancing', 'caching', 'memory', 'storage', 'network',
    'protocol', 'interface', 'abstraction', 'encapsulation', 'inheritance', 'polymorphism',
    'design pattern', 'anti-pattern', 'refactoring', 'clean code', 'maintainable',
    'readable', 'documentation', 'version control', 'git', 'repository', 'branch',
    'merge', 'conflict', 'review', 'approval', 'deployment', 'rollback', 'blue-green',
    'canary', 'feature flag', 'a/b testing', 'experiment', 'hypothesis', 'validation',
    'feedback', 'iteration', 'agile', 'scrum', 'kanban', 'sprint', 'backlog', 'story',
    'epic', 'milestone', 'roadmap', 'strategy', 'vision', 'mission', 'goal', 'objective',
    'kpi', 'metric', 'measurement', 'performance', 'efficiency', 'effectiveness',
    'quality', 'reliability', 'availability', 'usability', 'accessibility', 'security',
    'privacy', 'compliance', 'regulation', 'governance', 'risk', 'audit', 'certification'
}

def load_anchors_data(csv_path: str) -> pd.DataFrame:
    """Load semantic anchors data from CSV."""
    print(f"Loading anchors data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} anchors")
    return df

def filter_noise(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out noise (stopwords) and separate into clean and low-signal data.
    Returns: (clean_anchors, low_signal_anchors)
    """
    print("Filtering noise (stopwords)...")

    # Create mask for stopwords
    stopword_mask = df['text'].str.lower().isin(STOPWORDS)

    # Split into clean and low-signal
    clean_anchors = df[~stopword_mask].copy()
    low_signal_anchors = df[stopword_mask].copy()

    # Add signal quality tags
    clean_anchors['signal_quality'] = 'HIGH_SIGNAL'
    low_signal_anchors['signal_quality'] = 'LOW_SIGNAL'

    print(f"Clean anchors: {len(clean_anchors):,} ({len(clean_anchors)/len(df)*100:.1f}%)")
    print(f"Low signal anchors: {len(low_signal_anchors):,} ({len(low_signal_anchors)/len(df)*100:.1f}%)")

    return clean_anchors, low_signal_anchors

def tag_high_value_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Tag anchors as high-value for RAG prioritization."""
    print("Tagging high-value anchors...")

    # Create high-value mask
    high_value_mask = df['text'].str.lower().isin(HIGH_VALUE_TERMS)

    # Add RAG priority tags
    df['rag_priority'] = 'STANDARD'
    df.loc[high_value_mask, 'rag_priority'] = 'RAG_PRIORITIZED'

    # Tag context routing readiness
    df['context_ready'] = 'NEEDS_REVIEW'
    high_confidence_mask = df['confidence'] >= 0.7
    df.loc[high_confidence_mask & high_value_mask, 'context_ready'] = 'CONTEXT_ROUTE_READY'

    print(f"RAG prioritized anchors: {high_value_mask.sum():,}")
    print(f"Context route ready anchors: {(high_confidence_mask & high_value_mask).sum():,}")

    return df

def analyze_top_clusters(df: pd.DataFrame, top_n: int = 50) -> Dict:
    """Analyze top clusters by anchor count."""
    print(f"Analyzing top {top_n} clusters...")

    # Group by cluster and count
    cluster_counts = df['cluster_id'].value_counts()
    top_clusters = cluster_counts.head(top_n)

    cluster_analysis = {}

    for cluster_id, count in top_clusters.items():
        cluster_anchors = df[df['cluster_id'] == cluster_id]

        # Get sample anchors
        sample_anchors = cluster_anchors.head(10)[['anchor_id', 'text', 'type', 'confidence', 'file_path']].to_dict('records')

        # Calculate cluster statistics
        cluster_stats = {
            'size': count,
            'types': cluster_anchors['type'].value_counts().to_dict(),
            'avg_confidence': cluster_anchors['confidence'].mean(),
            'files': cluster_anchors['file_path'].nunique(),
            'sample_anchors': sample_anchors,
            'semantic_richness': calculate_semantic_richness(cluster_anchors)
        }

        cluster_analysis[cluster_id] = cluster_stats

    return cluster_analysis

def calculate_semantic_richness(cluster_anchors: pd.DataFrame) -> str:
    """Calculate semantic richness score for a cluster."""
    # Count unique types
    unique_types = cluster_anchors['type'].nunique()

    # Count high-value terms
    high_value_count = cluster_anchors['text'].str.lower().isin(HIGH_VALUE_TERMS).sum()

    # Average confidence
    avg_confidence = cluster_anchors['confidence'].mean()

    # Calculate richness score
    richness_score = (unique_types * 0.3 +
                     (high_value_count / len(cluster_anchors)) * 0.4 +
                     avg_confidence * 0.3)

    if richness_score >= 0.7:
        return 'HIGH'
    elif richness_score >= 0.4:
        return 'MEDIUM'
    else:
        return 'LOW'

def generate_enhanced_report(clean_anchors: pd.DataFrame,
                           low_signal_anchors: pd.DataFrame,
                           cluster_analysis: Dict,
                           output_dir: str):
    """Generate enhanced analysis report."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'enhanced_anchor_analysis_{timestamp}.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Enhanced Semantic Anchor Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Anchors:** {len(clean_anchors) + len(low_signal_anchors):,}\n")
        f.write(f"- **Clean Anchors:** {len(clean_anchors):,} ({len(clean_anchors)/(len(clean_anchors) + len(low_signal_anchors))*100:.1f}%)\n")
        f.write(f"- **Low Signal Anchors:** {len(low_signal_anchors):,} ({len(low_signal_anchors)/(len(clean_anchors) + len(low_signal_anchors))*100:.1f}%)\n")
        f.write(f"- **RAG Prioritized:** {len(clean_anchors[clean_anchors['rag_priority'] == 'RAG_PRIORITIZED']):,}\n")
        f.write(f"- **Context Route Ready:** {len(clean_anchors[clean_anchors['context_ready'] == 'CONTEXT_ROUTE_READY']):,}\n\n")

        # Signal Quality Analysis
        f.write("## Signal Quality Analysis\n\n")
        f.write("| Quality | Count | Percentage |\n")
        f.write("|---------|-------|------------|\n")
        f.write(f"| HIGH_SIGNAL | {len(clean_anchors):,} | {len(clean_anchors)/(len(clean_anchors) + len(low_signal_anchors))*100:.1f}% |\n")
        f.write(f"| LOW_SIGNAL | {len(low_signal_anchors):,} | {len(low_signal_anchors)/(len(clean_anchors) + len(low_signal_anchors))*100:.1f}% |\n\n")

        # RAG Priority Analysis
        f.write("## RAG Priority Analysis\n\n")
        rag_counts = clean_anchors['rag_priority'].value_counts()
        f.write("| Priority | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for priority, count in rag_counts.items():
            percentage = count / len(clean_anchors) * 100
            f.write(f"| {priority} | {count:,} | {percentage:.1f}% |\n")
        f.write("\n")

        # Context Readiness Analysis
        f.write("## Context Readiness Analysis\n\n")
        context_counts = clean_anchors['context_ready'].value_counts()
        f.write("| Readiness | Count | Percentage |\n")
        f.write("|-----------|-------|------------|\n")
        for readiness, count in context_counts.items():
            percentage = count / len(clean_anchors) * 100
            f.write(f"| {readiness} | {count:,} | {percentage:.1f}% |\n")
        f.write("\n")

        # Top Clusters Analysis
        f.write("## Top Clusters Analysis\n\n")
        f.write("| Rank | Cluster ID | Size | Richness | Avg Confidence | Files | Top Types |\n")
        f.write("|------|------------|------|----------|----------------|-------|-----------|\n")

        for i, (cluster_id, stats) in enumerate(cluster_analysis.items(), 1):
            top_types = list(stats['types'].keys())[:3]
            top_types_str = ', '.join(top_types)
            f.write(f"| {i} | {cluster_id} | {stats['size']:,} | {stats['semantic_richness']} | {stats['avg_confidence']:.3f} | {stats['files']} | {top_types_str} |\n")

        f.write("\n")

        # Sample Anchors from Top Clusters
        f.write("## Sample Anchors from Top Clusters\n\n")

        for i, (cluster_id, stats) in enumerate(cluster_analysis.items(), 1):
            if i <= 25:  # Show top 25 clusters
                f.write(f"### Cluster {cluster_id} (Rank {i}, Size: {stats['size']:,}, Richness: {stats['semantic_richness']})\n\n")
                f.write("| Anchor ID | Text | Type | Confidence | File |\n")
                f.write("|-----------|------|------|------------|------|\n")

                for anchor in stats['sample_anchors'][:10]:
                    f.write(f"| {anchor['anchor_id']} | {anchor['text']} | {anchor['type']} | {anchor['confidence']} | {anchor['file_path']} |\n")
                f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("### 1. Noise Filtering Results\n")
        f.write(f"- Successfully filtered out {len(low_signal_anchors):,} low-signal anchors\n")
        f.write(f"- Improved signal-to-noise ratio by {len(clean_anchors)/(len(clean_anchors) + len(low_signal_anchors))*100:.1f}%\n\n")

        f.write("### 2. RAG Prioritization\n")
        rag_prioritized = len(clean_anchors[clean_anchors['rag_priority'] == 'RAG_PRIORITIZED'])
        f.write(f"- {rag_prioritized:,} anchors tagged for RAG prioritization\n")
        f.write("- These anchors should be given higher weight in retrieval systems\n\n")

        f.write("### 3. Context Routing\n")
        context_ready = len(clean_anchors[clean_anchors['context_ready'] == 'CONTEXT_ROUTE_READY'])
        f.write(f"- {context_ready:,} anchors ready for automatic context routing\n")
        f.write("- These can be used for intelligent prompt injection\n\n")

        f.write("### 4. Cluster Optimization\n")
        high_richness_clusters = sum(1 for stats in cluster_analysis.values() if stats['semantic_richness'] == 'HIGH')
        f.write(f"- {high_richness_clusters} clusters identified as semantically rich\n")
        f.write("- Consider merging low-richness clusters to reduce noise\n\n")

    print(f"Enhanced report saved to: {report_path}")
    return report_path

def save_filtered_data(clean_anchors: pd.DataFrame,
                      low_signal_anchors: pd.DataFrame,
                      output_dir: str):
    """Save filtered data to CSV files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save clean anchors
    clean_path = os.path.join(output_dir, f'clean_anchors_{timestamp}.csv')
    clean_anchors.to_csv(clean_path, index=False)
    print(f"Clean anchors saved to: {clean_path}")

    # Save low signal anchors
    low_signal_path = os.path.join(output_dir, f'low_signal_anchors_{timestamp}.csv')
    low_signal_anchors.to_csv(low_signal_path, index=False)
    print(f"Low signal anchors saved to: {low_signal_path}")

    return clean_path, low_signal_path

def main():
    parser = argparse.ArgumentParser(description='Enhanced Semantic Anchor Analysis')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file with semantic anchors')
    parser.add_argument('-o', '--output', required=True, help='Output directory for results')
    parser.add_argument('-t', '--top-clusters', type=int, default=50, help='Number of top clusters to analyze')

    args = parser.parse_args()

    print("🔍 Starting Enhanced Semantic Anchor Analysis...")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    df = load_anchors_data(args.input)

    # Step 2: Filter noise
    clean_anchors, low_signal_anchors = filter_noise(df)

    # Step 3: Tag high-value anchors
    clean_anchors = tag_high_value_anchors(clean_anchors)

    # Step 4: Analyze top clusters
    cluster_analysis = analyze_top_clusters(clean_anchors, args.top_clusters)

    # Generate enhanced report
    report_path = generate_enhanced_report(clean_anchors, low_signal_anchors, cluster_analysis, args.output)

    # Save filtered data
    clean_path, low_signal_path = save_filtered_data(clean_anchors, low_signal_anchors, args.output)

    print("\n✅ Enhanced analysis complete!")
    print(f"📄 Report: {report_path}")
    print(f"📄 Clean anchors: {clean_path}")
    print(f"📄 Low signal anchors: {low_signal_path}")

if __name__ == "__main__":
    main()
