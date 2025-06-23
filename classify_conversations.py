#!/usr/bin/env python3
import json
import csv
import os
import re
import argparse
from datetime import datetime
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter
import hashlib

# Topic classification rules
TOPIC_KEYWORDS = {
    'ai_ml': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'neural network', 'deep learning', 'gpt', 'llm', 'model', 'training', 'inference', 'transformer', 'embedding', 'vector'],
    'programming': ['code', 'programming', 'python', 'javascript', 'react', 'api', 'function', 'class', 'variable', 'debug', 'error', 'syntax', 'algorithm', 'database', 'sql'],
    'technical': ['server', 'cloud', 'aws', 'docker', 'kubernetes', 'deployment', 'infrastructure', 'network', 'security', 'authentication', 'encryption', 'protocol'],
    'business': ['business', 'strategy', 'marketing', 'sales', 'revenue', 'profit', 'customer', 'market', 'competition', 'growth', 'startup', 'investment'],
    'creative': ['design', 'art', 'creative', 'writing', 'story', 'narrative', 'visual', 'aesthetic', 'brand', 'logo', 'color', 'typography'],
    'personal': ['personal', 'life', 'health', 'fitness', 'relationship', 'family', 'friend', 'hobby', 'travel', 'food', 'cooking'],
    'education': ['learn', 'education', 'study', 'course', 'tutorial', 'teach', 'explain', 'understand', 'knowledge', 'skill'],
    'data_analysis': ['data', 'analysis', 'statistics', 'chart', 'graph', 'visualization', 'excel', 'csv', 'dataset', 'metric'],
    'writing': ['write', 'writing', 'essay', 'article', 'blog', 'content', 'copy', 'edit', 'grammar', 'style'],
    'research': ['research', 'study', 'paper', 'academic', 'journal', 'citation', 'methodology', 'hypothesis', 'experiment']
}

# Emotion classification rules
EMOTION_KEYWORDS = {
    'positive': ['happy', 'excited', 'great', 'awesome', 'amazing', 'wonderful', 'excellent', 'perfect', 'love', 'enjoy', 'fantastic', 'brilliant'],
    'negative': ['frustrated', 'angry', 'disappointed', 'sad', 'terrible', 'awful', 'hate', 'annoying', 'problem', 'issue', 'error', 'fail'],
    'neutral': ['okay', 'fine', 'normal', 'standard', 'regular', 'typical', 'average'],
    'curious': ['wonder', 'curious', 'interesting', 'how', 'why', 'what', 'question', 'explore', 'discover'],
    'confused': ['confused', 'unclear', 'don\'t understand', 'puzzled', 'lost', 'help', 'stuck'],
    'urgent': ['urgent', 'asap', 'quickly', 'immediately', 'rush', 'deadline', 'time sensitive'],
    'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'consider', 'examine', 'review']
}

# Project classification rules (based on common project types)
PROJECT_KEYWORDS = {
    'web_development': ['website', 'web app', 'frontend', 'backend', 'html', 'css', 'javascript', 'react', 'vue', 'angular'],
    'data_project': ['data analysis', 'dashboard', 'visualization', 'report', 'analytics', 'metrics', 'kpi'],
    'automation': ['automate', 'script', 'workflow', 'process', 'batch', 'scheduled', 'pipeline'],
    'mobile_app': ['mobile', 'app', 'ios', 'android', 'flutter', 'react native'],
    'ai_project': ['chatbot', 'recommendation', 'classification', 'prediction', 'nlp', 'computer vision'],
    'infrastructure': ['deployment', 'server', 'cloud', 'devops', 'ci/cd', 'monitoring'],
    'research': ['research', 'experiment', 'analysis', 'study', 'investigation'],
    'content_creation': ['blog', 'article', 'documentation', 'content', 'writing', 'copywriting'],
    'learning': ['tutorial', 'course', 'learning', 'study', 'practice', 'exercise'],
    'planning': ['plan', 'strategy', 'roadmap', 'timeline', 'schedule', 'project management']
}

def generate_id(text, prefix):
    """Generate a consistent ID based on text content"""
    hash_value = hashlib.md5(text.lower().encode()).hexdigest()[:8]
    return f'{prefix}_{hash_value}'

def extract_text_from_content(content):
    """Extract text from ChatGPT export content structure"""
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        if 'parts' in content and isinstance(content['parts'], list):
            if len(content['parts']) > 0:
                part = content['parts'][0]
                if isinstance(part, str):
                    return part
                elif isinstance(part, dict) and 'text' in part:
                    return part['text']
        elif 'text' in content:
            return content['text']
    return ''

def classify_by_keywords(text, keyword_dict):
    """Classify text based on keyword presence"""
    if not text:
        return 'unknown'

    text_lower = text.lower()
    scores = {}

    for category, keywords in keyword_dict.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += text_lower.count(keyword)
        scores[category] = score

    if not scores or max(scores.values()) == 0:
        return 'unknown'

    return max(scores, key=scores.get)

def extract_conversation_text(file_path):
    """Extract all text content from a conversation"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_text = []
        if 'mapping' in data:
            for msg_id, msg_data in data['mapping'].items():
                if 'message' in msg_data and msg_data['message']:
                    message = msg_data['message']
                    content = message.get('content', {})
                    text = extract_text_from_content(content)
                    if text and text.strip():
                        all_text.append(text)

        return ' '.join(all_text)

    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return ''

def classify_conversation(file_path, filename):
    """Classify a single conversation for topic, emotion, and project"""
    conversation_text = extract_conversation_text(file_path)

    if not conversation_text:
        return {
            'filename': filename,
            'topic_id': generate_id('unknown', 'TOPIC'),
            'topic_label': 'unknown',
            'emotion_id': generate_id('neutral', 'EMOTION'),
            'emotion_label': 'neutral',
            'project_id': generate_id('unknown', 'PROJECT'),
            'project_label': 'unknown',
            'confidence_score': 0.0,
            'text_length': 0
        }

    # Classify topic, emotion, and project
    topic_label = classify_by_keywords(conversation_text, TOPIC_KEYWORDS)
    emotion_label = classify_by_keywords(conversation_text, EMOTION_KEYWORDS)
    project_label = classify_by_keywords(conversation_text, PROJECT_KEYWORDS)

    # Generate consistent IDs
    topic_id = generate_id(topic_label, 'TOPIC')
    emotion_id = generate_id(emotion_label, 'EMOTION')
    project_id = generate_id(project_label, 'PROJECT')

    # Simple confidence score based on text length and keyword matches
    confidence_score = min(1.0, len(conversation_text) / 1000)

    return {
        'filename': filename,
        'topic_id': topic_id,
        'topic_label': topic_label,
        'emotion_id': emotion_id,
        'emotion_label': emotion_label,
        'project_id': project_id,
        'project_label': project_label,
        'confidence_score': round(confidence_score, 3),
        'text_length': len(conversation_text)
    }

def load_conversation_ids(csv_path):
    """Load existing conversation IDs"""
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
    parser = argparse.ArgumentParser(description='Classify conversations by topic, emotion, and project')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output directory for classification results')

    args = parser.parse_args()

    print('🔍 Classifying conversations by topic, emotion, and project...')

    os.makedirs(args.output, exist_ok=True)

    # Load existing conversation IDs
    conversation_ids = load_conversation_ids(args.conversation_ids)

    classifications = []
    topic_stats = Counter()
    emotion_stats = Counter()
    project_stats = Counter()

    processed_count = 0
    total_files = len([f for f in os.listdir(args.input) if f.endswith('.json')])

    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input, filename)

            classification = classify_conversation(file_path, filename)
            classification['conversation_id'] = conversation_ids.get(filename, f'UNKNOWN_{filename}')

            classifications.append(classification)

            # Update statistics
            topic_stats[classification['topic_label']] += 1
            emotion_stats[classification['emotion_label']] += 1
            project_stats[classification['project_label']] += 1

            processed_count += 1
            if processed_count % 50 == 0:
                print(f'Processed {processed_count}/{total_files} conversations...')

    print(f'✅ Processed all {processed_count} conversations!')

    # Write classification results
    classifications_output = os.path.join(args.output, 'conversation_classifications.csv')
    fieldnames = ['filename', 'conversation_id', 'topic_id', 'topic_label', 'emotion_id', 'emotion_label',
                  'project_id', 'project_label', 'confidence_score', 'text_length']

    with open(classifications_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(classifications)

    # Generate summary report
    report_data = {
        'processing_date': datetime.now().isoformat(),
        'total_conversations': len(classifications),
        'topic_distribution': dict(topic_stats.most_common()),
        'emotion_distribution': dict(emotion_stats.most_common()),
        'project_distribution': dict(project_stats.most_common()),
        'average_confidence': round(sum(c['confidence_score'] for c in classifications) / len(classifications), 3),
        'classification_rules': {
            'topics': list(TOPIC_KEYWORDS.keys()),
            'emotions': list(EMOTION_KEYWORDS.keys()),
            'projects': list(PROJECT_KEYWORDS.keys())
        }
    }

    report_output = os.path.join(args.output, 'classification_report.json')
    with open(report_output, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Generate ID mappings
    topic_mapping = {generate_id(topic, 'TOPIC'): topic for topic in TOPIC_KEYWORDS.keys()}
    topic_mapping[generate_id('unknown', 'TOPIC')] = 'unknown'

    emotion_mapping = {generate_id(emotion, 'EMOTION'): emotion for emotion in EMOTION_KEYWORDS.keys()}
    emotion_mapping[generate_id('neutral', 'EMOTION')] = 'neutral'

    project_mapping = {generate_id(project, 'PROJECT'): project for project in PROJECT_KEYWORDS.keys()}
    project_mapping[generate_id('unknown', 'PROJECT')] = 'unknown'

    # Write ID mappings
    mappings_output = os.path.join(args.output, 'classification_id_mappings.json')
    mappings_data = {
        'topic_mappings': topic_mapping,
        'emotion_mappings': emotion_mapping,
        'project_mappings': project_mapping
    }

    with open(mappings_output, 'w', encoding='utf-8') as f:
        json.dump(mappings_data, f, indent=2, ensure_ascii=False)

    print(f'\n✅ Classification complete!')
    print(f'📄 Classifications: {os.path.abspath(classifications_output)}')
    print(f'📊 Report: {os.path.abspath(report_output)}')
    print(f'🗺️  ID Mappings: {os.path.abspath(mappings_output)}')
    print(f'\n📈 Top Topics: {dict(topic_stats.most_common(5))}')
    print(f'😊 Top Emotions: {dict(emotion_stats.most_common(5))}')
    print(f'🚀 Top Projects: {dict(project_stats.most_common(5))}')

if __name__ == '__main__':
    main()
