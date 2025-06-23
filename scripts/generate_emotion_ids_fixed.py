#!/usr/bin/env python3
"""
Generate Emotion IDs for ChatGPT Conversations (Fixed Version)

This script identifies emotional language patterns in conversations and assigns unique IDs.
Emotions are identified through pattern matching and linguistic analysis.
Fixed to handle the actual JSON structure with messages in mapping object.
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

# Emotion patterns and indicators (Plutchik-inspired wheel with 25+ categories)
EMOTION_PATTERNS = {
    'joy': [
        r'\b(happy|joy|excited|thrilled|delighted|pleased|great|wonderful|fantastic|amazing)\b',
        r'\b(awesome|excellent|brilliant|perfect|outstanding|superb|magnificent|splendid)\b',
        r'\b(celebration|success|achievement|victory|triumph|accomplishment|breakthrough)\b'
    ],
    'trust': [
        r'\b(trust|confident|reliable|dependable|secure|safe|assured|certain|sure|positive)\b',
        r'\b(trustworthy|faithful|loyal|steadfast|unwavering|solid|stable|firm)\b',
        r'\b(confidence|belief|conviction|certainty|assurance|reliability|dependability)\b'
    ],
    'fear': [
        r'\b(fear|afraid|scared|terrified|worried|anxious|nervous|concerned|apprehensive)\b',
        r'\b(panic|dread|horror|terror|fright|alarm|distress|unease|disquiet)\b',
        r'\b(threat|danger|risk|hazard|peril|menace|vulnerability|insecurity)\b'
    ],
    'surprise': [
        r'\b(surprise|shocked|amazed|astonished|stunned|startled|bewildered|perplexed)\b',
        r'\b(unexpected|unforeseen|unanticipated|sudden|abrupt|unpredictable|unusual)\b',
        r'\b(wow|incredible|unbelievable|remarkable|extraordinary|exceptional|striking)\b'
    ],
    'sadness': [
        r'\b(sad|sorrow|grief|melancholy|depression|despair|hopeless|disappointed)\b',
        r'\b(miserable|wretched|dejected|downcast|gloomy|somber|melancholic)\b',
        r'\b(loss|failure|defeat|setback|disappointment|frustration|regret|remorse)\b'
    ],
    'disgust': [
        r'\b(disgust|revolted|repulsed|sickened|nauseated|appalled|horrified|offended)\b',
        r'\b(repulsive|revolting|disgusting|sickening|nauseating|appalling|horrible)\b',
        r'\b(contempt|scorn|derision|mockery|ridicule|disdain|aversion|abhorrence)\b'
    ],
    'anger': [
        r'\b(anger|angry|furious|enraged|irritated|annoyed|frustrated|mad|livid)\b',
        r'\b(rage|wrath|fury|outrage|indignation|resentment|hostility|aggression)\b',
        r'\b(irritation|annoyance|exasperation|aggravation|provocation|incitement)\b'
    ],
    'anticipation': [
        r'\b(anticipation|excited|eager|enthusiastic|looking forward|expecting)\b',
        r'\b(hopeful|optimistic|positive|confident|assured|certain|sure)\b',
        r'\b(expectation|hope|optimism|confidence|assurance|certainty|conviction)\b'
    ],
    'curiosity': [
        r'\b(curious|inquisitive|interested|intrigued|fascinated|wondering|questioning)\b',
        r'\b(explore|investigate|examine|study|analyze|research|discover)\b',
        r'\b(interest|intrigue|fascination|wonder|inquiry|exploration|investigation)\b'
    ],
    'determination': [
        r'\b(determined|resolved|committed|dedicated|persistent|tenacious|steadfast)\b',
        r'\b(focused|concentrated|single-minded|purposeful|driven|motivated)\b',
        r'\b(resolution|commitment|dedication|persistence|tenacity|focus|purpose)\b'
    ],
    'gratitude': [
        r'\b(grateful|thankful|appreciative|indebted|obliged|thank you|thanks)\b',
        r'\b(appreciation|gratitude|thanksgiving|recognition|acknowledgment)\b',
        r'\b(blessed|fortunate|lucky|privileged|honored|valued|cherished)\b'
    ],
    'pride': [
        r'\b(pride|proud|accomplished|achieved|successful|victorious|triumphant)\b',
        r'\b(achievement|success|accomplishment|victory|triumph|mastery|excellence)\b',
        r'\b(satisfaction|fulfillment|contentment|gratification|self-respect)\b'
    ],
    'empathy': [
        r'\b(empathy|compassion|sympathy|understanding|caring|concerned|supportive)\b',
        r'\b(compassionate|sympathetic|understanding|caring|supportive|helpful)\b',
        r'\b(compassion|sympathy|understanding|care|support|help|assistance)\b'
    ],
    'inspiration': [
        r'\b(inspired|motivated|encouraged|stimulated|energized|enthusiastic)\b',
        r'\b(inspiration|motivation|encouragement|stimulation|energy|enthusiasm)\b',
        r'\b(inspiring|motivating|encouraging|stimulating|energizing|uplifting)\b'
    ],
    'relief': [
        r'\b(relief|relieved|eased|comforted|soothed|calmed|reassured)\b',
        r'\b(comfort|ease|soothing|calming|reassuring|consoling|solace)\b',
        r'\b(comfortable|at ease|peaceful|tranquil|serene|calm|relaxed)\b'
    ],
    'confusion': [
        r'\b(confused|puzzled|perplexed|bewildered|baffled|mystified|uncertain)\b',
        r'\b(confusion|puzzle|perplexity|bewilderment|bafflement|uncertainty)\b',
        r'\b(unclear|vague|ambiguous|unclear|obscure|enigmatic|cryptic)\b'
    ],
    'satisfaction': [
        r'\b(satisfied|content|pleased|fulfilled|gratified|happy|satisfactory)\b',
        r'\b(satisfaction|contentment|fulfillment|gratification|happiness|pleasure)\b',
        r'\b(adequate|sufficient|enough|complete|finished|done|accomplished)\b'
    ],
    'frustration': [
        r'\b(frustrated|annoyed|irritated|exasperated|aggravated|bothered|troubled)\b',
        r'\b(frustration|annoyance|irritation|exasperation|aggravation|trouble)\b',
        r'\b(blocked|hindered|obstructed|prevented|stopped|impeded|thwarted)\b'
    ],
    'wonder': [
        r'\b(wonder|amazement|awe|astonishment|marvel|miracle|extraordinary)\b',
        r'\b(wonderful|amazing|astonishing|marvelous|miraculous|extraordinary)\b',
        r'\b(magical|mystical|enchanting|captivating|mesmerizing|spellbinding)\b'
    ],
    'caution': [
        r'\b(cautious|careful|wary|vigilant|alert|attentive|mindful)\b',
        r'\b(caution|care|wariness|vigilance|alertness|attention|mindfulness)\b',
        r'\b(prudent|discreet|judicious|sensible|reasonable|thoughtful)\b'
    ],
    'hope': [
        r'\b(hope|hopeful|optimistic|positive|encouraging|promising|bright)\b',
        r'\b(optimism|positivity|encouragement|promise|brightness|potential)\b',
        r'\b(potential|possibility|opportunity|chance|prospect|outlook)\b'
    ],
    'doubt': [
        r'\b(doubt|doubtful|uncertain|unsure|questioning|skeptical|suspicious)\b',
        r'\b(uncertainty|question|skepticism|suspicion|hesitation|reluctance)\b',
        r'\b(maybe|perhaps|possibly|potentially|theoretically|hypothetically)\b'
    ],
    'regret': [
        r'\b(regret|regretful|remorseful|sorry|apologetic|contrite|repentant)\b',
        r'\b(remorse|sorrow|apology|contrition|repentance|penitence)\b',
        r'\b(wish|if only|should have|could have|would have|missed opportunity)\b'
    ],
    'overwhelmed': [
        r'\b(overwhelmed|overloaded|swamped|flooded|drowned|buried|crushed)\b',
        r'\b(too much|excessive|extreme|intense|powerful|forceful|strong)\b',
        r'\b(exhausted|tired|fatigued|weary|drained|depleted|spent)\b'
    ],
    'collaborative': [
        r'\b(collaborate|cooperate|work together|team|partnership|alliance)\b',
        r'\b(collaboration|cooperation|teamwork|partnership|alliance|unity)\b',
        r'\b(together|joint|shared|mutual|collective|united|combined)\b'
    ],
    'competitive': [
        r'\b(compete|competition|rival|opponent|challenge|contest|match)\b',
        r'\b(competitive|rivalrous|oppositional|challenging|contesting)\b',
        r'\b(win|lose|victory|defeat|triumph|success|achievement)\b'
    ]
}

EMOTION_INDICATORS = [
    r'\b(feel|feeling|emotion|emotional|mood|sentiment|attitude)\b',
    r'\b(seem|appear|look|sound|taste|smell|appears to be)\b',
    r'\b(very|really|extremely|quite|rather|somewhat|slightly)\b'
]

def extract_emotions(text: str) -> List[Dict[str, Any]]:
    """Extract emotions from text using pattern matching."""
    emotions = []
    text_lower = text.lower()

    # Check for emotion indicators
    has_indicators = any(re.search(pattern, text_lower) for pattern in EMOTION_INDICATORS)

    # Check each emotion category
    for category, patterns in EMOTION_PATTERNS.items():
        matches = []
        for pattern in patterns:
            found = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in found:
                matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': pattern
                })

        if matches:
            # Calculate emotion strength based on matches and context
            strength = len(matches)
            if has_indicators:
                strength += 2

            # Extract surrounding context
            context_start = max(0, min(matches[0]['start'] - 50, len(text)))
            context_end = min(len(text), max(matches[-1]['end'] + 50, len(text)))
            context = text[context_start:context_end].strip()

            emotions.append({
                'category': category,
                'matches': matches,
                'strength': strength,
                'context': context,
                'indicators_present': has_indicators
            })

    return emotions

def generate_emotion_id(conversation_id: str, emotion: Dict[str, Any], index: int) -> str:
    """Generate unique emotion ID."""
    # Create hash from conversation ID, category, and context
    hash_input = f"{conversation_id}_{emotion['category']}_{emotion['context'][:50]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"EMOTION_{conversation_id}_{emotion['category'].upper()}_{index:03d}_{hash_value}"

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
        # Handle content.parts array structure
        parts = content['parts']
        if isinstance(parts, list):
            return ' '.join(str(part) for part in parts if part)
        else:
            return str(parts)
    else:
        return str(content)

def process_conversation(file_path: str, conversation_id: str) -> List[Dict[str, Any]]:
    """Process a single conversation file for emotions."""
    emotions = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle the actual JSON structure with mapping object
        mapping = data.get('mapping', {})
        if not mapping:
            return emotions

        # Process each message in the mapping
        for message_id, message_data in mapping.items():
            if not message_data or 'message' not in message_data:
                continue

            message = message_data['message']
            if not message:
                continue

            # Extract content from the message
            content = extract_message_content(message_data)
            if not content or not isinstance(content, str):
                continue

            # Extract emotions from message content
            message_emotions = extract_emotions(content)

            for i, emotion in enumerate(message_emotions):
                emotion_id = generate_emotion_id(conversation_id, emotion, len(emotions))

                # Get message metadata
                author = message.get('author', {})
                role = author.get('role', 'unknown') if author else 'unknown'
                create_time = message.get('create_time', '')
                timestamp = datetime.fromtimestamp(create_time).isoformat() if create_time else ''

                emotions.append({
                    'emotion_id': emotion_id,
                    'conversation_id': conversation_id,
                    'category': emotion['category'],
                    'strength': emotion['strength'],
                    'context': emotion['context'],
                    'indicators_present': emotion['indicators_present'],
                    'message_id': message_id,
                    'role': role,
                    'timestamp': timestamp,
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
            'emotion_id', 'conversation_id', 'category', 'strength',
            'context', 'indicators_present', 'message_id', 'role',
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

    for emotion in emotions:
        category_counts[emotion['category']] += 1
        strength_distribution[emotion['strength']] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Emotion ID Generation Summary\n\n")
        f.write(f"Total emotions generated: {len(emotions)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `EMOTION_CONVID_CATEGORY_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- CATEGORY: Emotion category\n")
        f.write("- INDEX: Emotion index within conversation\n")
        f.write("- HASH: 6-character hash of context\n\n")

        f.write("## Category Distribution\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(emotions)) * 100
            f.write(f"- **{category.title()}**: {count} emotions ({percentage:.1f}%)\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(emotions)) * 100
            f.write(f"- **Strength {strength}**: {count} emotions ({percentage:.1f}%)\n")

        f.write("\n## Recent Emotions\n\n")
        recent_emotions = sorted(emotions, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        for emotion in recent_emotions:
            f.write(f"- **{emotion['emotion_id']}**: {emotion['category']} - {emotion['context'][:100]}...\n")

if __name__ == "__main__":
    main()
