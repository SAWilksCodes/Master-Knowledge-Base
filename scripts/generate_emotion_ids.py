#!/usr/bin/env python3
"""
Generate Emotion IDs for ChatGPT Conversations

This script identifies emotional content and sentiment in conversations and assigns unique IDs.
Emotions are identified through keyword analysis, sentiment patterns, and contextual clues.
"""

import json
import csv
import os
import re
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Emotion categories and keywords
EMOTION_CATEGORIES = {
    'joy': [
        'happy', 'joy', 'excited', 'thrilled', 'delighted', 'pleased', 'great',
        'wonderful', 'amazing', 'fantastic', 'excellent', 'brilliant', 'perfect',
        'love', 'adore', 'enjoy', 'fun', 'entertaining', 'hilarious', 'laugh'
    ],
    'sadness': [
        'sad', 'unhappy', 'depressed', 'melancholy', 'gloomy', 'sorrow', 'grief',
        'disappointed', 'heartbroken', 'devastated', 'miserable', 'hopeless',
        'lonely', 'isolated', 'abandoned', 'rejected', 'worthless'
    ],
    'anger': [
        'angry', 'mad', 'furious', 'enraged', 'irritated', 'annoyed', 'frustrated',
        'outraged', 'livid', 'hostile', 'aggressive', 'violent', 'hate', 'despise',
        'disgusted', 'repulsed', 'offended', 'insulted'
    ],
    'fear': [
        'afraid', 'scared', 'frightened', 'terrified', 'panicked', 'anxious',
        'worried', 'nervous', 'concerned', 'apprehensive', 'dread', 'horror',
        'threatened', 'vulnerable', 'unsafe', 'dangerous', 'risky'
    ],
    'surprise': [
        'surprised', 'shocked', 'astonished', 'amazed', 'stunned', 'bewildered',
        'confused', 'puzzled', 'perplexed', 'baffled', 'unexpected', 'unbelievable',
        'incredible', 'remarkable', 'extraordinary', 'unusual'
    ],
    'disgust': [
        'disgusted', 'repulsed', 'revolted', 'sickened', 'nauseated', 'gross',
        'vile', 'filthy', 'dirty', 'contaminated', 'polluted', 'corrupt'
    ],
    'trust': [
        'trust', 'confident', 'secure', 'safe', 'reliable', 'dependable',
        'faithful', 'loyal', 'honest', 'sincere', 'genuine', 'authentic',
        'believable', 'credible', 'trustworthy'
    ],
    'anticipation': [
        'excited', 'eager', 'enthusiastic', 'optimistic', 'hopeful', 'looking forward',
        'anticipate', 'expect', 'await', 'prepare', 'plan', 'ready', 'keen'
    ],
    'curiosity': [
        'curious', 'interested', 'intrigued', 'fascinated', 'wondering', 'questioning',
        'explore', 'discover', 'investigate', 'learn', 'understand', 'figure out'
    ],
    'gratitude': [
        'thankful', 'grateful', 'appreciative', 'blessed', 'fortunate', 'lucky',
        'thank you', 'thanks', 'appreciate', 'value', 'cherish', 'treasure'
    ],
    'pride': [
        'proud', 'accomplished', 'achieved', 'successful', 'victorious', 'triumphant',
        'confident', 'self-assured', 'satisfied', 'fulfilled', 'content'
    ],
    'shame': [
        'ashamed', 'embarrassed', 'humiliated', 'guilty', 'remorseful', 'regretful',
        'sorry', 'apologetic', 'inferior', 'inadequate', 'unworthy', 'defective'
    ],
    'contempt': [
        'contempt', 'disdain', 'scorn', 'disrespect', 'look down on', 'superior',
        'arrogant', 'condescending', 'patronizing', 'dismissive', 'disregard'
    ],
    'neutral': [
        'neutral', 'calm', 'peaceful', 'serene', 'tranquil', 'balanced',
        'stable', 'steady', 'composed', 'collected', 'unemotional', 'objective'
    ]
}

# Emotional intensity indicators
INTENSITY_INDICATORS = {
    'very': 2,
    'extremely': 3,
    'incredibly': 3,
    'absolutely': 3,
    'completely': 2,
    'totally': 2,
    'really': 1,
    'quite': 1,
    'somewhat': 0.5,
    'slightly': 0.5,
    'barely': 0.25,
    'hardly': 0.25
}

# Emotional expression patterns
EMOTION_PATTERNS = {
    'exclamation': r'!+',
    'question': r'\?+',
    'capitalization': r'\b[A-Z]{3,}\b',
    'repetition': r'(\w+)(?:\s+\1)+',
    'emoticons': r'[:;=]-?[)(/\\|pPoO]',
    'emoji_indicators': r'[😀-🙏🌀-🗿]'
}

def extract_emotions(text: str) -> List[Dict[str, Any]]:
    """Extract emotions from text using keyword analysis and pattern matching."""
    emotions = []
    text_lower = text.lower()

    # Check each emotion category
    for category, keywords in EMOTION_CATEGORIES.items():
        matches = []
        intensity = 1.0

        for keyword in keywords:
            if keyword in text_lower:
                matches.append(keyword)

                # Check for intensity modifiers
                for modifier, multiplier in INTENSITY_INDICATORS.items():
                    if f"{modifier} {keyword}" in text_lower:
                        intensity = max(intensity, multiplier)
                        break

        if matches:
            # Calculate emotion strength
            strength = len(matches) * intensity

            # Check for emotional expression patterns
            pattern_matches = []
            for pattern_name, pattern in EMOTION_PATTERNS.items():
                if re.search(pattern, text):
                    pattern_matches.append(pattern_name)
                    strength += 0.5

            # Extract context around emotion mentions
            context_start = max(0, text_lower.find(matches[0]) - 50)
            context_end = min(len(text), text_lower.find(matches[-1]) + len(matches[-1]) + 50)
            context = text[context_start:context_end].strip()

            emotions.append({
                'category': category,
                'keywords': matches,
                'strength': round(strength, 2),
                'intensity': intensity,
                'context': context,
                'patterns': pattern_matches
            })

    return emotions

def generate_emotion_id(conversation_id: str, emotion: Dict[str, Any], index: int) -> str:
    """Generate unique emotion ID."""
    # Create hash from conversation ID, category, and keywords
    hash_input = f"{conversation_id}_{emotion['category']}_{'_'.join(emotion['keywords'][:3])}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"EMOTION_{conversation_id}_{emotion['category'].upper()}_{index:03d}_{hash_value}"

def process_conversation(file_path: str, conversation_id: str) -> List[Dict[str, Any]]:
    """Process a single conversation file for emotions."""
    emotions = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process each message
        for message in data.get('messages', []):
            content = message.get('content', '')
            if not content or not isinstance(content, str):
                continue

            # Extract emotions from message content
            message_emotions = extract_emotions(content)

            for i, emotion in enumerate(message_emotions):
                emotion_id = generate_emotion_id(conversation_id, emotion, len(emotions))

                emotions.append({
                    'emotion_id': emotion_id,
                    'conversation_id': conversation_id,
                    'category': emotion['category'],
                    'keywords': '; '.join(emotion['keywords']),
                    'strength': emotion['strength'],
                    'intensity': emotion['intensity'],
                    'context': emotion['context'],
                    'patterns': '; '.join(emotion['patterns']),
                    'message_index': data['messages'].index(message),
                    'role': message.get('role', 'unknown'),
                    'timestamp': message.get('timestamp', ''),
                    'file_path': file_path
                })

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return emotions

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
    parser = argparse.ArgumentParser(description='Generate Emotion IDs for ChatGPT conversations')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file for emotion IDs')

    args = parser.parse_args()

    print("🔍 Processing conversations for emotion ID generation...")

    # Load conversation IDs
    conversation_ids = load_conversation_ids(args.conversation_ids)

    all_emotions = []
    processed_count = 0

    # Process each JSON file
    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input, filename)
            conversation_id = conversation_ids.get(filename, f"UNKNOWN_{filename}")

            emotions = process_conversation(file_path, conversation_id)
            all_emotions.extend(emotions)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} conversations...")

    # Write results to CSV
    if all_emotions:
        fieldnames = [
            'emotion_id', 'conversation_id', 'category', 'keywords', 'strength',
            'intensity', 'context', 'patterns', 'message_index', 'role',
            'timestamp', 'file_path'
        ]

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_emotions)

        # Generate summary
        summary_path = args.output.replace('.csv', '.md')
        generate_summary(all_emotions, summary_path, processed_count)

        print(f"\n✅ Emotion ID generation complete!")
        print(f"📄 CSV file: {os.path.abspath(args.output)}")
        print(f"📄 Summary: {os.path.abspath(summary_path)}")
        print(f"📊 Generated {len(all_emotions)} emotion IDs")
    else:
        print("❌ No emotions found in conversations")

def generate_summary(emotions: List[Dict[str, Any]], output_path: str, total_conversations: int):
    """Generate summary markdown file."""
    if not emotions:
        return

    # Count by category
    category_counts = defaultdict(int)
    strength_distribution = defaultdict(int)
    intensity_distribution = defaultdict(int)
    pattern_counts = defaultdict(int)

    for emotion in emotions:
        category_counts[emotion['category']] += 1
        strength_distribution[round(emotion['strength'])] += 1
        intensity_distribution[round(emotion['intensity'], 1)] += 1

        if emotion['patterns']:
            for pattern in emotion['patterns'].split('; '):
                pattern_counts[pattern] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Emotion ID Generation Summary\n\n")
        f.write(f"Total emotions generated: {len(emotions)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `EMOTION_CONVID_CATEGORY_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- CATEGORY: Emotion category\n")
        f.write("- INDEX: Emotion index within conversation\n")
        f.write("- HASH: 6-character hash of keywords\n\n")

        f.write("## Category Distribution\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(emotions)) * 100
            f.write(f"- **{category.title()}**: {count} emotions ({percentage:.1f}%)\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(emotions)) * 100
            f.write(f"- **Strength {strength}**: {count} emotions ({percentage:.1f}%)\n")

        f.write("\n## Intensity Distribution\n\n")
        for intensity in sorted(intensity_distribution.keys()):
            count = intensity_distribution[intensity]
            percentage = (count / len(emotions)) * 100
            f.write(f"- **Intensity {intensity}**: {count} emotions ({percentage:.1f}%)\n")

        f.write("\n## Pattern Distribution\n\n")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(emotions)) * 100
            f.write(f"- **{pattern.title()}**: {count} emotions ({percentage:.1f}%)\n")

        f.write("\n## Recent Emotions\n\n")
        recent_emotions = sorted(emotions, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        for emotion in recent_emotions:
            f.write(f"- **{emotion['emotion_id']}**: {emotion['category']} (strength: {emotion['strength']}) - {emotion['keywords'][:100]}...\n")

if __name__ == "__main__":
    main()
