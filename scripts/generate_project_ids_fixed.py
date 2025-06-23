#!/usr/bin/env python3
"""
Generate Project IDs for ChatGPT Conversations (Fixed Version)

This script identifies project-related language patterns in conversations and assigns unique IDs.
Projects are identified through pattern matching and linguistic analysis.
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

# Project patterns and indicators (25+ project categories)
PROJECT_PATTERNS = {
    'ai_development': [
        r'\b(ai|artificial intelligence|machine learning|ml|neural network|model|algorithm)\b',
        r'\b(training|fine-tuning|inference|prediction|classification|regression)\b',
        r'\b(openai|gpt|llm|large language model|transformer|bert|tensorflow|pytorch)\b'
    ],
    'automation': [
        r'\b(automation|automate|script|workflow|pipeline|process|automated)\b',
        r'\b(zapier|make\.com|ifttt|webhook|api|integration|connector)\b',
        r'\b(schedule|trigger|event|action|task|job|batch|cron)\b'
    ],
    'web_development': [
        r'\b(website|web app|frontend|backend|api|rest|graphql|http|https)\b',
        r'\b(html|css|javascript|react|vue|angular|node|express|django|flask)\b',
        r'\b(domain|hosting|deployment|server|client|database|sql|nosql)\b'
    ],
    'data_analysis': [
        r'\b(data|dataset|analysis|analytics|visualization|chart|graph|plot)\b',
        r'\b(pandas|numpy|matplotlib|seaborn|plotly|tableau|powerbi|excel)\b',
        r'\b(statistics|correlation|regression|classification|clustering|ml)\b'
    ],
    'mobile_development': [
        r'\b(mobile|app|ios|android|react native|flutter|swift|kotlin)\b',
        r'\b(phone|tablet|responsive|native|hybrid|progressive web app|pwa)\b',
        r'\b(app store|google play|deployment|testing|beta|release)\b'
    ],
    'hardware_project': [
        r'\b(hardware|electronics|circuit|sensor|arduino|raspberry pi|microcontroller)\b',
        r'\b(3d printing|cnc|laser cutter|prototype|manufacturing|assembly)\b',
        r'\b(component|part|tool|equipment|device|gadget|invention)\b'
    ],
    'content_creation': [
        r'\b(content|article|blog|video|podcast|social media|marketing)\b',
        r'\b(writing|editing|publishing|seo|keyword|audience|engagement)\b',
        r'\b(campaign|strategy|brand|promotion|advertising|outreach)\b'
    ],
    'research_project': [
        r'\b(research|study|experiment|investigation|analysis|survey|interview)\b',
        r'\b(literature review|methodology|hypothesis|conclusion|findings)\b',
        r'\b(academic|scientific|thesis|dissertation|paper|publication)\b'
    ],
    'business_project': [
        r'\b(business|startup|company|enterprise|venture|entrepreneur)\b',
        r'\b(strategy|plan|market|customer|revenue|profit|investment)\b',
        r'\b(management|leadership|team|organization|operations|finance)\b'
    ],
    'learning_project': [
        r'\b(learning|education|course|tutorial|training|skill|knowledge)\b',
        r'\b(study|practice|exercise|assignment|homework|certification)\b',
        r'\b(online|e-learning|mooc|workshop|seminar|conference)\b'
    ],
    'design_project': [
        r'\b(design|ui|ux|user interface|user experience|wireframe|prototype)\b',
        r'\b(figma|sketch|adobe|photoshop|illustrator|invision|figma)\b',
        r'\b(layout|typography|color|branding|logo|visual|graphic)\b'
    ],
    'database_project': [
        r'\b(database|db|sql|nosql|mongodb|postgresql|mysql|sqlite)\b',
        r'\b(schema|table|query|index|migration|backup|replication)\b',
        r'\b(data modeling|erd|normalization|optimization|performance)\b'
    ],
    'devops_project': [
        r'\b(devops|deployment|ci/cd|docker|kubernetes|aws|azure|gcp)\b',
        r'\b(infrastructure|server|cloud|monitoring|logging|security)\b',
        r'\b(git|github|gitlab|jenkins|travis|pipeline|automation)\b'
    ],
    'game_development': [
        r'\b(game|gaming|unity|unreal|game engine|sprite|animation)\b',
        r'\b(level|character|physics|collision|ai|multiplayer|single player)\b',
        r'\b(steam|console|mobile game|indie|publisher|release)\b'
    ],
    'blockchain_project': [
        r'\b(blockchain|bitcoin|ethereum|cryptocurrency|smart contract|defi)\b',
        r'\b(nft|token|wallet|mining|staking|yield|liquidity)\b',
        r'\b(web3|metamask|solidity|rust|consensus|distributed)\b'
    ],
    'iot_project': [
        r'\b(iot|internet of things|sensor|device|connected|smart home)\b',
        r'\b(mqtt|coap|zigbee|bluetooth|wifi|cellular|lora)\b',
        r'\b(edge computing|gateway|cloud|analytics|monitoring)\b'
    ],
    'cybersecurity_project': [
        r'\b(security|cybersecurity|hacking|penetration testing|vulnerability)\b',
        r'\b(encryption|authentication|authorization|firewall|vpn|ssl)\b',
        r'\b(threat|risk|compliance|audit|incident response|forensics)\b'
    ],
    'financial_project': [
        r'\b(finance|financial|investment|trading|portfolio|budget|accounting)\b',
        r'\b(stock|bond|mutual fund|etf|roi|profit|loss|revenue)\b',
        r'\b(banking|payment|transaction|tax|audit|compliance)\b'
    ],
    'healthcare_project': [
        r'\b(healthcare|medical|health|patient|diagnosis|treatment|therapy)\b',
        r'\b(hospital|clinic|pharmacy|insurance|telemedicine|wearable)\b',
        r'\b(compliance|hipaa|fda|clinical trial|research|data)\b'
    ],
    'education_project': [
        r'\b(education|teaching|learning|student|curriculum|syllabus)\b',
        r'\b(classroom|online|blended|assessment|grading|feedback)\b',
        r'\b(lms|moodle|canvas|blackboard|google classroom|zoom)\b'
    ],
    'social_impact': [
        r'\b(social impact|nonprofit|charity|volunteer|community|sustainability)\b',
        r'\b(environmental|climate|renewable|green|eco-friendly|conservation)\b',
        r'\b(advocacy|activism|awareness|campaign|fundraising|donation)\b'
    ],
    'entertainment_project': [
        r'\b(entertainment|media|streaming|video|audio|music|podcast)\b',
        r'\b(netflix|youtube|spotify|apple|amazon|disney|hulu)\b',
        r'\b(production|editing|post-production|distribution|marketing)\b'
    ],
    'ecommerce_project': [
        r'\b(ecommerce|online store|shopify|woocommerce|amazon|ebay)\b',
        r'\b(shopping cart|payment|checkout|inventory|order|shipping)\b',
        r'\b(customer|product|catalog|pricing|discount|loyalty)\b'
    ],
    'consulting_project': [
        r'\b(consulting|consultant|advisory|expert|specialist|professional)\b',
        r'\b(client|project|deliverable|proposal|scope|timeline|budget)\b',
        r'\b(assessment|recommendation|strategy|implementation|support)\b'
    ],
    'product_development': [
        r'\b(product|development|feature|roadmap|sprint|agile|scrum)\b',
        r'\b(requirements|specification|design|prototype|testing|launch)\b',
        r'\b(user story|backlog|iteration|release|version|update)\b'
    ],
    'monitoring_system': [
        r'\b(monitoring|alerting|dashboard|metrics|analytics|tracking)\b',
        r'\b(prometheus|grafana|datadog|newrelic|splunk|elk stack)\b',
        r'\b(performance|availability|uptime|latency|throughput|error)\b'
    ]
}

PROJECT_INDICATORS = [
    r'\b(project|task|work|assignment|job|undertaking|initiative)\b',
    r'\b(develop|build|create|implement|deploy|launch|release)\b',
    r'\b(plan|strategy|approach|methodology|framework|process)\b'
]

def extract_projects(text: str) -> List[Dict[str, Any]]:
    """Extract projects from text using pattern matching."""
    projects = []
    text_lower = text.lower()

    # Check for project indicators
    has_indicators = any(re.search(pattern, text_lower) for pattern in PROJECT_INDICATORS)

    # Check each project category
    for category, patterns in PROJECT_PATTERNS.items():
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
            # Calculate project strength based on matches and context
            strength = len(matches)
            if has_indicators:
                strength += 2

            # Extract surrounding context
            context_start = max(0, min(matches[0]['start'] - 50, len(text)))
            context_end = min(len(text), max(matches[-1]['end'] + 50, len(text)))
            context = text[context_start:context_end].strip()

            projects.append({
                'category': category,
                'matches': matches,
                'strength': strength,
                'context': context,
                'indicators_present': has_indicators
            })

    return projects

def generate_project_id(conversation_id: str, project: Dict[str, Any], index: int) -> str:
    """Generate unique project ID."""
    # Create hash from conversation ID, category, and context
    hash_input = f"{conversation_id}_{project['category']}_{project['context'][:50]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"PROJECT_{conversation_id}_{project['category'].upper()}_{index:03d}_{hash_value}"

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
    """Process a single conversation file for projects."""
    projects = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle the actual JSON structure with mapping object
        mapping = data.get('mapping', {})
        if not mapping:
            return projects

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

            # Extract projects from message content
            message_projects = extract_projects(content)

            for i, project in enumerate(message_projects):
                project_id = generate_project_id(conversation_id, project, len(projects))

                # Get message metadata
                author = message.get('author', {})
                role = author.get('role', 'unknown') if author else 'unknown'
                create_time = message.get('create_time', '')
                timestamp = datetime.fromtimestamp(create_time).isoformat() if create_time else ''

                projects.append({
                    'project_id': project_id,
                    'conversation_id': conversation_id,
                    'category': project['category'],
                    'strength': project['strength'],
                    'context': project['context'],
                    'indicators_present': project['indicators_present'],
                    'message_id': message_id,
                    'role': role,
                    'timestamp': timestamp,
                    'file_path': file_path
                })

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return projects

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
    parser = argparse.ArgumentParser(description='Generate Project IDs for ChatGPT conversations')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files')
    parser.add_argument('-c', '--conversation-ids', required=True, help='CSV file with conversation ID mappings')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file for project IDs')

    args = parser.parse_args()

    print("🔍 Processing conversations for project ID generation...")

    # Load conversation IDs
    conversation_ids = load_conversation_ids(args.conversation_ids)

    all_projects = []
    processed_count = 0

    # Process each JSON file
    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input, filename)
            conversation_id = conversation_ids.get(filename, f"UNKNOWN_{filename}")

            projects = process_conversation(file_path, conversation_id)
            all_projects.extend(projects)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} conversations...")

    # Write results to CSV
    if all_projects:
        fieldnames = [
            'project_id', 'conversation_id', 'category', 'strength',
            'context', 'indicators_present', 'message_id', 'role',
            'timestamp', 'file_path'
        ]

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_projects)

        # Generate summary
        summary_path = args.output.replace('.csv', '.md')
        generate_summary(all_projects, summary_path, processed_count)

        print(f"\n✅ Project ID generation complete!")
        print(f"📄 CSV file: {os.path.abspath(args.output)}")
        print(f"📄 Summary: {os.path.abspath(summary_path)}")
        print(f"📊 Generated {len(all_projects)} project IDs")
    else:
        print("❌ No projects found in conversations")

def generate_summary(projects: List[Dict[str, Any]], output_path: str, total_conversations: int):
    """Generate summary markdown file."""
    if not projects:
        return

    # Count by category
    category_counts = defaultdict(int)
    strength_distribution = defaultdict(int)

    for project in projects:
        category_counts[project['category']] += 1
        strength_distribution[project['strength']] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Project ID Generation Summary\n\n")
        f.write(f"Total projects generated: {len(projects)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `PROJECT_CONVID_CATEGORY_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- CATEGORY: Project category\n")
        f.write("- INDEX: Project index within conversation\n")
        f.write("- HASH: 6-character hash of context\n\n")

        f.write("## Category Distribution\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(projects)) * 100
            f.write(f"- **{category.replace('_', ' ').title()}**: {count} projects ({percentage:.1f}%)\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(projects)) * 100
            f.write(f"- **Strength {strength}**: {count} projects ({percentage:.1f}%)\n")

        f.write("\n## Recent Projects\n\n")
        recent_projects = sorted(projects, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        for project in recent_projects:
            f.write(f"- **{project['project_id']}**: {project['category']} - {project['context'][:100]}...\n")

if __name__ == "__main__":
    main()
