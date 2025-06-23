import json
import os
from pathlib import Path
import argparse
import shutil
import re

# Topic keywords - expanded and more comprehensive
TOPIC_KEYWORDS = {
    'programming': [
        'python', 'javascript', 'code', 'function', 'variable', 'debug', 'error',
        'api', 'database', 'sql', 'git', 'github', 'programming', 'script',
        'class', 'method', 'array', 'loop', 'algorithm', 'react', 'node',
        'java', 'c++', 'rust', 'golang', 'typescript', 'html', 'css', 'web',
        'app', 'software', 'developer', 'coding', 'bug', 'fix', 'compile'
    ],
    'data_analysis': [
        'data', 'analysis', 'pandas', 'numpy', 'dataframe', 'csv', 'excel',
        'visualization', 'plot', 'graph', 'statistics', 'machine learning',
        'dataset', 'correlation', 'regression', 'chart', 'analytics', 'metrics'
    ],
    'writing': [
        'write', 'writing', 'essay', 'article', 'blog', 'story', 'narrative',
        'paragraph', 'grammar', 'edit', 'proofread', 'content', 'copywriting',
        'document', 'text', 'draft', 'revision', 'author', 'publish'
    ],
    'ai_ml': [
        'ai', 'artificial intelligence', 'machine learning', 'neural network',
        'deep learning', 'gpt', 'llm', 'transformer', 'training', 'model',
        'chatgpt', 'claude', 'prompt', 'openai', 'hugging face', 'tensorflow',
        'pytorch', 'bert', 'nlp', 'computer vision', 'reinforcement learning',
        'fine-tuning', 'inference', 'dataset', 'epoch', 'batch', 'gpu', 'vram',
        'omnisync', 'agent', 'rag', 'vector', 'embedding', 'langchain'
    ],
    'business': [
        'business', 'marketing', 'sales', 'strategy', 'management', 'startup',
        'entrepreneur', 'revenue', 'customer', 'product', 'market', 'company',
        'profit', 'investment', 'roi', 'budget', 'finance', 'accounting',
        'insurance', 'policy', 'quote', 'client', 'proposal', 'contract'
    ],
    'research': [
        'research', 'study', 'paper', 'academic', 'literature', 'citation',
        'hypothesis', 'methodology', 'analysis', 'findings', 'conclusion',
        'experiment', 'theory', 'peer review', 'journal', 'publication'
    ],
    'creative': [
        'creative', 'design', 'art', 'music', 'poem', 'story', 'fiction',
        'character', 'plot', 'theme', 'drawing', 'artistic', 'illustration',
        'animation', 'video', 'audio', 'media', 'content creation'
    ],
    'technical': [
        'technical', 'server', 'network', 'system', 'infrastructure', 'devops',
        'docker', 'kubernetes', 'cloud', 'aws', 'deployment', 'configuration',
        'linux', 'windows', 'macos', 'terminal', 'command', 'shell', 'bash',
        'hardware', 'cpu', 'gpu', 'ram', 'storage', 'computer', 'pc', 'laptop'
    ],
    'personal': [
        'personal', 'life', 'help', 'advice', 'question', 'problem', 'solution',
        'recommendation', 'suggest', 'opinion', 'thought', 'idea', 'plan',
        'goal', 'task', 'project', 'schedule', 'organize', 'manage'
    ],
    'education': [
        'learn', 'learning', 'education', 'course', 'tutorial', 'guide',
        'teach', 'teaching', 'student', 'lesson', 'curriculum', 'study',
        'knowledge', 'skill', 'training', 'workshop', 'certification'
    ]
}

def extract_text_from_conversation(convo):
    """Extract all text content from a conversation."""
    texts = []

    # Get title
    if 'title' in convo:
        texts.append(convo['title'].lower())

    # Extract from mapping structure (ChatGPT format)
    if 'mapping' in convo:
        for node_id, node in convo['mapping'].items():
            if 'message' in node and node['message']:
                msg = node['message']
                if 'content' in msg:
                    content = msg['content']
                    if isinstance(content, dict) and 'parts' in content:
                        for part in content['parts']:
                            if isinstance(part, str):
                                texts.append(part.lower())
                    elif isinstance(content, str):
                        texts.append(content.lower())

    # Extract from messages array (alternative format)
    elif 'messages' in convo:
        for msg in convo['messages']:
            if 'content' in msg:
                if isinstance(msg['content'], str):
                    texts.append(msg['content'].lower())
                elif isinstance(msg['content'], list):
                    for item in msg['content']:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'].lower())
                        elif isinstance(item, str):
                            texts.append(item.lower())

    # Extract from linear_conversation if present
    elif 'linear_conversation' in convo:
        for msg in convo['linear_conversation']:
            if 'text' in msg and isinstance(msg['text'], str):
                texts.append(msg['text'].lower())

    return ' '.join(texts)

def categorize_conversation(convo):
    """Determine the primary topic(s) of a conversation."""
    text = extract_text_from_conversation(convo)

    if not text:
        return ['uncategorized']

    # Count keyword matches for each topic
    topic_scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Use word boundaries for more accurate matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches

        if score > 0:
            topic_scores[topic] = score

    if not topic_scores:
        return ['uncategorized']

    # More lenient scoring - return topics with at least 2 matches
    # or topics that have at least 20% of the max score
    max_score = max(topic_scores.values())
    threshold = max(2, max_score * 0.2)  # At least 2 matches or 20% of max

    top_topics = [topic for topic, score in topic_scores.items()
                  if score >= threshold]

    # If we have too many topics (>3), take only the top 3
    if len(top_topics) > 3:
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        top_topics = [topic for topic, _ in sorted_topics[:3]]

    return top_topics if top_topics else ['uncategorized']

def split_by_topic(input_dir, output_dir, copy_files=True):
    """Split conversations by detected topics."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Clear output directory if it exists
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob('*.json'))
    print(f"🏷️  Processing {len(json_files)} conversations by topic...")

    stats = {}
    multi_topic_count = 0
    processed_count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                convo = json.load(f)

            topics = categorize_conversation(convo)

            # Handle multi-topic conversations
            if len(topics) > 1:
                multi_topic_count += 1

            for topic in topics:
                # Create topic folder
                topic_folder = output_path / topic
                topic_folder.mkdir(parents=True, exist_ok=True)

                # Update stats
                stats[topic] = stats.get(topic, 0) + 1

                # Copy or move file
                dest_file = topic_folder / json_file.name
                if copy_files or len(topics) > 1:
                    shutil.copy2(json_file, dest_file)
                else:
                    shutil.move(str(json_file), str(dest_file))
                    break  # Only move once if not copying

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"   Processed {processed_count}/{len(json_files)} files...")

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Print summary
    print(f"\n✅ Topic-based split complete!")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"\n📊 Conversations by topic:")

    for topic in sorted(stats.keys()):
        print(f"   {topic}: {stats[topic]} conversations")

    if multi_topic_count > 0:
        print(f"\n📌 {multi_topic_count} conversations matched multiple topics")

    total_categorized = sum(stats.values())
    print(f"\n📈 Total categorizations: {total_categorized}")
    print(f"📄 Total files processed: {processed_count}")

def main():
    parser = argparse.ArgumentParser(
        description="Split conversations by detected topics"
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for organized files")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")

    args = parser.parse_args()
    split_by_topic(args.input_dir, args.output_dir, copy_files=not args.move)

if __name__ == "__main__":
    main()
