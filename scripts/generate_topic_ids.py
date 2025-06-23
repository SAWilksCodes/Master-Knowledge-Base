#!/usr/bin/env python3
"""
Generate Topic IDs
Creates unique IDs for topic categories based on conversation content analysis.
Builds on existing topic organization to create standardized topic IDs.
"""

import json
import csv
import os
from pathlib import Path
import argparse
import re
import hashlib
from collections import defaultdict

# Topic categories and their keywords
TOPIC_PATTERNS = {
    'ai_ml': {
        'keywords': ['ai', 'machine learning', 'neural', 'gpt', 'llm', 'model', 'training', 'inference'],
        'description': 'Artificial Intelligence and Machine Learning'
    },
    'programming': {
        'keywords': ['python', 'javascript', 'code', 'programming', 'function', 'class', 'api', 'debug'],
        'description': 'Programming and Software Development'
    },
    'data_analysis': {
        'keywords': ['data', 'analysis', 'pandas', 'csv', 'excel', 'database', 'sql', 'visualization'],
        'description': 'Data Analysis and Processing'
    },
    'web_development': {
        'keywords': ['web', 'website', 'html', 'css', 'react', 'frontend', 'backend', 'server'],
        'description': 'Web Development and Design'
    },
    'business': {
        'keywords': ['business', 'strategy', 'marketing', 'sales', 'revenue', 'profit', 'startup'],
        'description': 'Business and Entrepreneurship'
    },
    'education': {
        'keywords': ['learn', 'education', 'course', 'tutorial', 'study', 'knowledge', 'skill'],
        'description': 'Education and Learning'
    },
    'creative': {
        'keywords': ['creative', 'design', 'art', 'writing', 'content', 'story', 'idea'],
        'description': 'Creative and Design Work'
    },
    'technical': {
        'keywords': ['technical', 'hardware', 'system', 'architecture', 'infrastructure', 'deployment'],
        'description': 'Technical Infrastructure and Systems'
    },
    'research': {
        'keywords': ['research', 'study', 'experiment', 'analysis', 'investigation', 'discovery'],
        'description': 'Research and Investigation'
    },
    'personal': {
        'keywords': ['personal', 'life', 'family', 'health', 'wellness', 'lifestyle', 'hobby'],
        'description': 'Personal and Lifestyle Topics'
    },
    'writing': {
        'keywords': ['write', 'writing', 'essay', 'article', 'content', 'document', 'text'],
        'description': 'Writing and Content Creation'
    }
}

def extract_text_content(conversation_data):
    """Extract all text content from conversation for topic analysis."""
    text_content = []

    if 'mapping' in conversation_data:
        for node_id, node in conversation_data['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']
                if 'content' in msg:
                    content = msg['content']
                    if isinstance(content, dict) and 'parts' in content:
                        for part in content['parts']:
                            if isinstance(part, str):
                                text_content.append(part.lower())

    return ' '.join(text_content)

def detect_topics(text_content, title):
    """Detect topics based on content analysis."""
    full_text = f"{title.lower()} {text_content}"
    detected_topics = []

    for topic_id, topic_info in TOPIC_PATTERNS.items():
        score = 0
        keyword_matches = 0

        for keyword in topic_info['keywords']:
            # Count keyword occurrences
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, full_text))
            if matches > 0:
                keyword_matches += 1
                score += matches

        # Topic is detected if we have at least 2 keyword matches or significant score
        if keyword_matches >= 2 or score >= 3:
            detected_topics.append({
                'topic_id': topic_id,
                'score': score,
                'keyword_matches': keyword_matches,
                'description': topic_info['description']
            })

    # Sort by score
    detected_topics.sort(key=lambda x: x['score'], reverse=True)
    return detected_topics

def generate_topic_id(topic_name, topic_hash):
    """Generate unique topic ID."""
    # Format: TOPIC_NAME_HASH
    return f"TOPIC_{topic_name.upper()}_{topic_hash}"

def process_conversations(input_dir, conversation_ids_file, output_file):
    """Process conversations and generate topic IDs."""
    input_path = Path(input_dir)

    # Load conversation IDs
    conversation_ids = {}
    with open(conversation_ids_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conversation_ids[row['filename']] = row['conversation_id']

    json_files = list(input_path.glob('*.json'))
    print(f"🔍 Processing {len(json_files)} conversations for topic ID generation...")

    # Track all topics and their conversations
    topic_conversations = defaultdict(list)
    conversation_topics = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)

            conversation_id = conversation_ids.get(json_file.name, f"CONV_{json_file.stem}")
            title = conversation_data.get('title', 'Untitled')

            # Extract text content
            text_content = extract_text_content(conversation_data)

            # Detect topics
            detected_topics = detect_topics(text_content, title)

            # Record conversation topics
            for topic in detected_topics:
                topic_conversations[topic['topic_id']].append({
                    'conversation_id': conversation_id,
                    'filename': json_file.name,
                    'title': title,
                    'score': topic['score'],
                    'keyword_matches': topic['keyword_matches']
                })

                conversation_topics.append({
                    'conversation_id': conversation_id,
                    'filename': json_file.name,
                    'title': title,
                    'topic_id': topic['topic_id'],
                    'topic_description': topic['description'],
                    'score': topic['score'],
                    'keyword_matches': topic['keyword_matches']
                })

            # If no topics detected, mark as uncategorized
            if not detected_topics:
                topic_conversations['uncategorized'].append({
                    'conversation_id': conversation_id,
                    'filename': json_file.name,
                    'title': title,
                    'score': 0,
                    'keyword_matches': 0
                })

                conversation_topics.append({
                    'conversation_id': conversation_id,
                    'filename': json_file.name,
                    'title': title,
                    'topic_id': 'uncategorized',
                    'topic_description': 'Uncategorized',
                    'score': 0,
                    'keyword_matches': 0
                })

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Generate topic IDs
    topic_ids = []
    for topic_name, conversations in topic_conversations.items():
        # Create hash from topic name and conversation count
        topic_hash = hashlib.md5(f"{topic_name}_{len(conversations)}".encode()).hexdigest()[:6]
        topic_id = generate_topic_id(topic_name, topic_hash)

        topic_ids.append({
            'topic_id': topic_id,
            'topic_name': topic_name,
            'description': TOPIC_PATTERNS.get(topic_name, {}).get('description', 'Uncategorized'),
            'conversation_count': len(conversations),
            'keywords': TOPIC_PATTERNS.get(topic_name, {}).get('keywords', [])
        })

    # Write topic IDs CSV
    output_path = Path(output_file)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['topic_id', 'topic_name', 'description', 'conversation_count', 'keywords']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topic_ids)

    # Write conversation-topic mapping
    mapping_path = output_path.with_name(output_path.stem + '_mapping.csv')
    with open(mapping_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['conversation_id', 'filename', 'title', 'topic_id', 'topic_description', 'score', 'keyword_matches']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(conversation_topics)

    # Create summary
    summary_path = output_path.with_suffix('.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Topic ID Generation Summary\n\n")
        f.write(f"Total topics identified: {len(topic_ids)}\n")
        f.write(f"Total conversations processed: {len(json_files)}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `TOPIC_NAME_HASH`\n")
        f.write("- NAME: Topic category name\n")
        f.write("- HASH: 6-character hash of topic name and conversation count\n\n")

        f.write("## Topic Categories\n\n")
        for topic in sorted(topic_ids, key=lambda x: x['conversation_count'], reverse=True):
            f.write(f"- **{topic['topic_id']}**: {topic['description']} ({topic['conversation_count']} conversations)\n")
            if topic['keywords']:
                f.write(f"  - Keywords: {', '.join(topic['keywords'][:5])}\n")
            f.write("\n")

    print(f"\n✅ Topic ID generation complete!")
    print(f"📄 Topic IDs: {output_path.absolute()}")
    print(f"📄 Topic mapping: {mapping_path.absolute()}")
    print(f"📄 Summary: {summary_path.absolute()}")
    print(f"📊 Generated {len(topic_ids)} topic IDs")

    return topic_ids, conversation_topics

def main():
    parser = argparse.ArgumentParser(description="Generate unique topic IDs for conversation categorization")
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-c", "--conversation-ids", required=True, help="CSV file with conversation IDs")
    parser.add_argument("-o", "--output", default="topic_ids.csv", help="Output CSV file")

    args = parser.parse_args()
    process_conversations(args.input_dir, args.conversation_ids, args.output)

if __name__ == "__main__":
    main()
