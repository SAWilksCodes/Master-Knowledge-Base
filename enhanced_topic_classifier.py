#!/usr/bin/env python3
import json
import csv
import os
import argparse
from datetime import datetime
from collections import defaultdict, Counter

# Enhanced granular topic categories (35+)
ENHANCED_TOPIC_CATEGORIES = {
    # AI/ML subcategories
    'ai_fundamentals': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ai basics', 'ml concepts'],
    'llm_development': ['large language model', 'llm', 'gpt', 'transformer', 'bert', 'chatgpt', 'language model', 'nlp'],
    'computer_vision': ['computer vision', 'image recognition', 'opencv', 'cnn', 'image processing', 'visual', 'detection'],
    'data_science': ['data science', 'data analysis', 'pandas', 'numpy', 'matplotlib', 'statistics', 'visualization'],
    'ml_engineering': ['model deployment', 'mlops', 'model training', 'feature engineering', 'model optimization'],

    # Programming subcategories
    'python_development': ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy', 'pip', 'conda'],
    'web_development': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 'frontend', 'backend'],
    'mobile_development': ['android', 'ios', 'flutter', 'react native', 'swift', 'kotlin', 'mobile app'],
    'game_development': ['unity', 'unreal', 'game engine', 'game development', 'gaming', 'graphics'],
    'api_development': ['api', 'rest', 'graphql', 'endpoint', 'microservices', 'webhook'],

    # Infrastructure & DevOps
    'cloud_computing': ['aws', 'azure', 'gcp', 'cloud', 'serverless', 'lambda', 'docker', 'kubernetes'],
    'database_management': ['sql', 'mysql', 'postgresql', 'mongodb', 'database', 'query', 'schema'],
    'devops_automation': ['ci/cd', 'jenkins', 'github actions', 'deployment', 'automation', 'pipeline'],
    'cybersecurity': ['security', 'encryption', 'authentication', 'vulnerability', 'penetration testing'],

    # Business & Strategy
    'startup_strategy': ['startup', 'business model', 'mvp', 'product market fit', 'venture capital', 'funding'],
    'product_management': ['product manager', 'roadmap', 'user story', 'agile', 'scrum', 'backlog'],
    'marketing_growth': ['marketing', 'seo', 'social media', 'content marketing', 'growth hacking'],
    'financial_planning': ['budget', 'finance', 'investment', 'roi', 'revenue', 'profit', 'cost analysis'],

    # Creative & Design
    'ui_ux_design': ['ui', 'ux', 'user interface', 'user experience', 'design system', 'wireframe', 'prototype'],
    'graphic_design': ['logo', 'branding', 'typography', 'color theory', 'visual design', 'illustration'],
    'content_creation': ['writing', 'copywriting', 'content strategy', 'blog', 'article', 'documentation'],
    'video_production': ['video editing', 'animation', 'motion graphics', 'streaming', 'youtube'],

    # Personal Development
    'career_development': ['career', 'job search', 'resume', 'interview', 'professional development', 'networking'],
    'skill_learning': ['learning', 'education', 'tutorial', 'course', 'certification', 'training'],
    'productivity_tools': ['productivity', 'time management', 'organization', 'workflow', 'automation'],
    'health_wellness': ['health', 'fitness', 'mental health', 'wellness', 'meditation', 'exercise'],

    # Technical Specializations
    'hardware_engineering': ['hardware', 'electronics', 'circuit', 'embedded systems', 'iot', 'raspberry pi'],
    '3d_printing': ['3d printing', 'additive manufacturing', 'cad', 'modeling', 'prototyping'],
    'blockchain_crypto': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'smart contract', 'defi'],
    'automation_robotics': ['automation', 'robotics', 'plc', 'industrial automation', 'control systems'],

    # Research & Academia
    'scientific_research': ['research', 'experiment', 'hypothesis', 'methodology', 'peer review', 'publication'],
    'academic_writing': ['thesis', 'dissertation', 'academic paper', 'citation', 'literature review'],

    # Lifestyle & Personal
    'home_improvement': ['diy', 'home improvement', 'renovation', 'construction', 'interior design'],
    'travel_planning': ['travel', 'vacation', 'trip planning', 'destination', 'itinerary', 'booking'],
    'cooking_food': ['cooking', 'recipe', 'food', 'nutrition', 'meal planning', 'restaurant'],
    'entertainment': ['movie', 'tv show', 'book', 'music', 'entertainment', 'review', 'recommendation'],

    # General categories
    'troubleshooting': ['error', 'bug', 'fix', 'troubleshoot', 'debug', 'problem solving', 'issue'],
    'general_inquiry': ['question', 'help', 'how to', 'what is', 'explain', 'clarify']
}

# Enhanced nuanced emotion states (22+)
ENHANCED_EMOTION_STATES = {
    # Positive emotions
    'excitement': ['excited', 'thrilled', 'enthusiastic', 'eager', 'pumped', 'energetic', 'amazing', 'awesome'],
    'curiosity': ['curious', 'interested', 'wondering', 'intrigued', 'fascinated', 'want to know', 'how does'],
    'confidence': ['confident', 'sure', 'certain', 'believe', 'know', 'convinced', 'positive'],
    'satisfaction': ['satisfied', 'pleased', 'happy', 'content', 'glad', 'good', 'great', 'excellent'],
    'gratitude': ['thank', 'grateful', 'appreciate', 'thanks', 'thankful', 'blessed'],
    'pride': ['proud', 'accomplished', 'achieved', 'successful', 'victory', 'won'],
    'hope': ['hope', 'hopeful', 'optimistic', 'looking forward', 'expect', 'anticipate'],

    # Negative emotions
    'frustration': ['frustrated', 'annoyed', 'irritated', 'bothered', 'upset', 'mad', 'angry'],
    'confusion': ['confused', 'puzzled', 'unclear', 'dont understand', 'lost', 'baffled', 'perplexed'],
    'disappointment': ['disappointed', 'let down', 'sad', 'unhappy', 'bummed', 'discouraged'],
    'anxiety': ['anxious', 'worried', 'nervous', 'stressed', 'concerned', 'afraid', 'scared'],
    'doubt': ['doubt', 'uncertain', 'unsure', 'skeptical', 'questionable', 'not sure'],
    'regret': ['regret', 'wish', 'should have', 'mistake', 'wrong', 'sorry'],

    # Analytical/neutral emotions
    'analytical': ['analyze', 'consider', 'evaluate', 'assess', 'examine', 'think', 'study'],
    'focused': ['focus', 'concentrate', 'attention', 'dedicated', 'committed', 'serious'],
    'methodical': ['systematic', 'organized', 'structured', 'planned', 'methodical', 'step by step'],
    'cautious': ['careful', 'cautious', 'conservative', 'safe', 'prudent', 'measured'],

    # Social emotions
    'collaborative': ['collaborate', 'work together', 'team', 'partnership', 'cooperation', 'help'],
    'competitive': ['compete', 'beat', 'win', 'better than', 'outperform', 'challenge'],

    # Learning emotions
    'determined': ['determined', 'persistent', 'committed', 'dedicated', 'wont give up'],
    'overwhelmed': ['overwhelmed', 'too much', 'cant handle', 'stressed out', 'burned out'],

    # Default
    'neutral': ['okay', 'fine', 'normal', 'standard', 'regular', 'typical']
}

# Enhanced specific project types (25+)
ENHANCED_PROJECT_TYPES = {
    # Software Development
    'web_application': ['web app', 'website', 'web development', 'frontend', 'backend', 'full stack'],
    'mobile_application': ['mobile app', 'android app', 'ios app', 'flutter', 'react native'],
    'desktop_application': ['desktop app', 'gui', 'tkinter', 'electron', 'pyqt', 'windows app'],
    'api_service': ['api', 'rest api', 'microservice', 'backend service', 'web service'],
    'game_project': ['game', 'unity', 'unreal', 'game development', 'indie game'],

    # AI/ML Projects
    'ai_project': ['ai model', 'machine learning', 'neural network', 'deep learning', 'ai system'],
    'chatbot_assistant': ['chatbot', 'virtual assistant', 'conversational ai', 'chat system'],
    'computer_vision_project': ['image recognition', 'computer vision', 'opencv', 'image processing'],
    'nlp_project': ['natural language processing', 'text analysis', 'sentiment analysis', 'nlp'],

    # Data & Analytics
    'data_analysis_project': ['data analysis', 'data visualization', 'dashboard', 'analytics', 'reporting'],
    'database_project': ['database', 'sql', 'data modeling', 'data migration', 'database design'],
    'etl_pipeline': ['etl', 'data pipeline', 'data processing', 'data transformation'],

    # Infrastructure & DevOps
    'automation_script': ['automation', 'script', 'batch processing', 'workflow automation'],
    'deployment_project': ['deployment', 'ci/cd', 'devops', 'infrastructure', 'cloud deployment'],
    'monitoring_system': ['monitoring', 'logging', 'alerting', 'observability', 'metrics'],

    # Business & Productivity
    'business_tool': ['business application', 'productivity tool', 'workflow tool', 'business process'],
    'ecommerce_platform': ['ecommerce', 'online store', 'shopping cart', 'payment system'],
    'content_management': ['cms', 'content management', 'blog platform', 'publishing system'],

    # Creative & Media
    'creative_project': ['creative', 'art', 'design', 'visual', 'graphics', 'multimedia'],
    'video_project': ['video', 'streaming', 'media player', 'video processing', 'animation'],
    'audio_project': ['audio', 'music', 'sound', 'podcast', 'audio processing'],

    # Hardware & IoT
    'iot_project': ['iot', 'internet of things', 'sensor', 'embedded', 'raspberry pi', 'arduino'],
    'hardware_project': ['hardware', 'electronics', 'circuit', 'pcb', 'embedded system'],

    # Personal & Learning
    'learning_project': ['learning', 'tutorial', 'practice', 'exercise', 'study project'],
    'portfolio_project': ['portfolio', 'showcase', 'demo', 'personal project', 'side project'],
    'research_project': ['research', 'experiment', 'prototype', 'proof of concept', 'investigation']
}

def load_conversations(directory):
    """Load conversation files from directory"""
    conversations = []
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    for filename in json_files:
        try:
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations.append({'filename': filename, 'data': data})
        except Exception as e:
            print(f'Error loading {filename}: {e}')

    return conversations

def extract_text_content(conversation_data):
    """Extract all text content from conversation"""
    text_content = []

    if 'mapping' in conversation_data:
        for node_id, node in conversation_data['mapping'].items():
            if 'message' in node and node['message']:
                message = node['message']
                if 'content' in message and 'parts' in message['content']:
                    for part in message['content']['parts']:
                        if isinstance(part, str) and part.strip():
                            text_content.append(part.strip())

    return ' '.join(text_content)

def calculate_confidence_score(text, keywords, total_words):
    """Calculate confidence score for classification"""
    if not keywords or total_words == 0:
        return 0.0

    text_lower = text.lower()
    matches = sum(1 for keyword in keywords if keyword in text_lower)

    # Base score from keyword matches
    base_score = matches / len(keywords)

    # Boost score based on text length (longer texts get slight boost for having more context)
    length_factor = min(1.2, 1 + (total_words / 10000))

    # Apply diminishing returns
    confidence = min(1.0, base_score * length_factor)

    return round(confidence, 3)

def enhanced_classify_topic(text):
    """Enhanced topic classification with confidence scoring"""
    text_lower = text.lower()
    total_words = len(text.split())

    topic_scores = {}

    for topic, keywords in ENHANCED_TOPIC_CATEGORIES.items():
        confidence = calculate_confidence_score(text, keywords, total_words)
        if confidence > 0:
            topic_scores[topic] = confidence

    if not topic_scores:
        return 'general_inquiry', 0.1, []

    # Sort by confidence
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

    primary_topic = sorted_topics[0][0]
    primary_confidence = sorted_topics[0][1]
    secondary_topics = [topic for topic, score in sorted_topics[1:3]]  # Top 2 alternatives

    return primary_topic, primary_confidence, secondary_topics

def enhanced_classify_emotion(text):
    """Enhanced emotion classification with confidence scoring"""
    text_lower = text.lower()
    total_words = len(text.split())

    emotion_scores = {}

    for emotion, keywords in ENHANCED_EMOTION_STATES.items():
        confidence = calculate_confidence_score(text, keywords, total_words)
        if confidence > 0:
            emotion_scores[emotion] = confidence

    if not emotion_scores:
        return 'neutral', 0.1, []

    # Sort by confidence
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

    primary_emotion = sorted_emotions[0][0]
    primary_confidence = sorted_emotions[0][1]
    secondary_emotions = [emotion for emotion, score in sorted_emotions[1:3]]  # Top 2 alternatives

    return primary_emotion, primary_confidence, secondary_emotions

def enhanced_classify_project(text):
    """Enhanced project classification with confidence scoring"""
    text_lower = text.lower()
    total_words = len(text.split())

    project_scores = {}

    for project_type, keywords in ENHANCED_PROJECT_TYPES.items():
        confidence = calculate_confidence_score(text, keywords, total_words)
        if confidence > 0:
            project_scores[project_type] = confidence

    if not project_scores:
        return 'general_inquiry', 0.1, []

    # Sort by confidence
    sorted_projects = sorted(project_scores.items(), key=lambda x: x[1], reverse=True)

    primary_project = sorted_projects[0][0]
    primary_confidence = sorted_projects[0][1]
    secondary_projects = [project for project, score in sorted_projects[1:3]]  # Top 2 alternatives

    return primary_project, primary_confidence, secondary_projects

def main():
    parser = argparse.ArgumentParser(description='Enhanced conversation classification with granular topics, nuanced emotions, and specific project types')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing conversation JSON files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for enhanced classification results')
    parser.add_argument('--min-confidence', type=float, default=0.1, help='Minimum confidence threshold for classification')

    args = parser.parse_args()

    print('🚀 Starting enhanced conversation classification...')
    print(f'📂 Input directory: {args.input}')
    print(f'📁 Output directory: {args.output}')
    print(f'🎯 Minimum confidence: {args.min_confidence}')

    os.makedirs(args.output, exist_ok=True)

    # Load conversations
    print('📖 Loading conversations...')
    conversations = load_conversations(args.input)

    if not conversations:
        print('❌ No conversations found!')
        return

    print(f'✅ Loaded {len(conversations)} conversations')

    # Enhanced classification
    classified_conversations = []
    topic_stats = Counter()
    emotion_stats = Counter()
    project_stats = Counter()
    confidence_stats = {'topic': [], 'emotion': [], 'project': []}

    print('🔍 Performing enhanced classification...')

    for i, conv in enumerate(conversations):
        if i % 50 == 0:
            print(f'Processing conversation {i+1}/{len(conversations)}...')

        text_content = extract_text_content(conv['data'])

        if not text_content.strip():
            continue

        # Enhanced classifications with confidence
        topic, topic_conf, topic_alt = enhanced_classify_topic(text_content)
        emotion, emotion_conf, emotion_alt = enhanced_classify_emotion(text_content)
        project, project_conf, project_alt = enhanced_classify_project(text_content)

        # Only include if meets minimum confidence
        if (topic_conf >= args.min_confidence and
            emotion_conf >= args.min_confidence and
            project_conf >= args.min_confidence):

            classified_conv = {
                'filename': conv['filename'],
                'conversation_id': conv['filename'].replace('.json', ''),
                'word_count': len(text_content.split()),
                'char_count': len(text_content),

                # Primary classifications
                'primary_topic': topic,
                'topic_confidence': topic_conf,
                'secondary_topics': topic_alt,

                'primary_emotion': emotion,
                'emotion_confidence': emotion_conf,
                'secondary_emotions': emotion_alt,

                'primary_project': project,
                'project_confidence': project_conf,
                'secondary_projects': project_alt,

                # Metadata
                'classification_date': datetime.now().isoformat(),
                'text_preview': text_content[:200] + '...' if len(text_content) > 200 else text_content
            }

            classified_conversations.append(classified_conv)

            # Update statistics
            topic_stats[topic] += 1
            emotion_stats[emotion] += 1
            project_stats[project] += 1

            confidence_stats['topic'].append(topic_conf)
            confidence_stats['emotion'].append(emotion_conf)
            confidence_stats['project'].append(project_conf)

    print(f'✅ Enhanced classification complete! Classified {len(classified_conversations)} conversations')

    # Write enhanced classification results
    enhanced_output = os.path.join(args.output, 'enhanced_conversation_classifications.csv')
    fieldnames = ['filename', 'conversation_id', 'word_count', 'char_count',
                  'primary_topic', 'topic_confidence', 'secondary_topics',
                  'primary_emotion', 'emotion_confidence', 'secondary_emotions',
                  'primary_project', 'project_confidence', 'secondary_projects',
                  'classification_date', 'text_preview']

    with open(enhanced_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for conv in classified_conversations:
            # Convert lists to strings for CSV
            conv_copy = conv.copy()
            conv_copy['secondary_topics'] = ', '.join(conv['secondary_topics'])
            conv_copy['secondary_emotions'] = ', '.join(conv['secondary_emotions'])
            conv_copy['secondary_projects'] = ', '.join(conv['secondary_projects'])
            writer.writerow(conv_copy)

    # Generate comprehensive enhanced report
    avg_confidences = {
        'topic': sum(confidence_stats['topic']) / len(confidence_stats['topic']) if confidence_stats['topic'] else 0,
        'emotion': sum(confidence_stats['emotion']) / len(confidence_stats['emotion']) if confidence_stats['emotion'] else 0,
        'project': sum(confidence_stats['project']) / len(confidence_stats['project']) if confidence_stats['project'] else 0
    }

    enhanced_report_data = {
        'processing_date': datetime.now().isoformat(),
        'total_conversations_loaded': len(conversations),
        'conversations_classified': len(classified_conversations),
        'classification_rate': len(classified_conversations) / len(conversations) if conversations else 0,
        'minimum_confidence_threshold': args.min_confidence,

        'average_confidences': avg_confidences,

        'topic_distribution': dict(topic_stats.most_common(20)),
        'emotion_distribution': dict(emotion_stats.most_common(20)),
        'project_distribution': dict(project_stats.most_common(20)),

        'classification_categories': {
            'topics_available': len(ENHANCED_TOPIC_CATEGORIES),
            'emotions_available': len(ENHANCED_EMOTION_STATES),
            'projects_available': len(ENHANCED_PROJECT_TYPES)
        },

        'top_combinations': [
            {
                'topic': conv['primary_topic'],
                'emotion': conv['primary_emotion'],
                'project': conv['primary_project'],
                'filename': conv['filename']
            }
            for conv in sorted(classified_conversations,
                             key=lambda x: x['topic_confidence'] + x['emotion_confidence'] + x['project_confidence'],
                             reverse=True)[:10]
        ],

        'enhancement_features': {
            'granular_topics': '35+ specific topic categories',
            'nuanced_emotions': '22+ emotional states',
            'specific_projects': '25+ project types',
            'confidence_scoring': 'Weighted keyword matching with confidence scores',
            'secondary_classification': 'Top 2 alternative classifications tracked',
            'text_analysis': 'Word count and character count analysis'
        }
    }

    enhanced_report_output = os.path.join(args.output, 'enhanced_classification_report.json')
    with open(enhanced_report_output, 'w', encoding='utf-8') as f:
        json.dump(enhanced_report_data, f, indent=2, ensure_ascii=False)

    print(f'\n✅ Enhanced classification complete!')
    print(f'📄 Classifications: {os.path.abspath(enhanced_output)}')
    print(f'📊 Enhanced report: {os.path.abspath(enhanced_report_output)}')
    print(f'\n📈 Top Topics: {dict(list(topic_stats.most_common(5)))}')
    print(f'😊 Top Emotions: {dict(list(emotion_stats.most_common(5)))}')
    print(f'🚀 Top Projects: {dict(list(project_stats.most_common(5)))}')
    print(f'🎯 Avg Confidences: Topic={avg_confidences["topic"]:.3f}, Emotion={avg_confidences["emotion"]:.3f}, Project={avg_confidences["project"]:.3f}')

if __name__ == '__main__':
    main()
