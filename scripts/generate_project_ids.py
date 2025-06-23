#!/usr/bin/env python3
"""
Generate Project IDs for ChatGPT Conversations

This script identifies project-related content and goals in conversations and assigns unique IDs.
Projects are identified through goal-oriented language, planning patterns, and task management indicators.
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

# Project-related keywords and patterns
PROJECT_KEYWORDS = {
    'planning': [
        'plan', 'strategy', 'roadmap', 'timeline', 'schedule', 'milestone',
        'deadline', 'goal', 'objective', 'target', 'aim', 'purpose'
    ],
    'development': [
        'develop', 'build', 'create', 'design', 'implement', 'construct',
        'program', 'code', 'develop', 'engineer', 'architect', 'prototype'
    ],
    'management': [
        'manage', 'organize', 'coordinate', 'oversee', 'supervise', 'lead',
        'direct', 'guide', 'facilitate', 'administer', 'control'
    ],
    'research': [
        'research', 'investigate', 'study', 'analyze', 'examine', 'explore',
        'discover', 'find', 'identify', 'understand', 'learn'
    ],
    'testing': [
        'test', 'validate', 'verify', 'check', 'debug', 'troubleshoot',
        'evaluate', 'assess', 'review', 'inspect', 'quality assurance'
    ],
    'deployment': [
        'deploy', 'launch', 'release', 'publish', 'distribute', 'install',
        'setup', 'configure', 'implement', 'rollout', 'go live'
    ],
    'maintenance': [
        'maintain', 'update', 'upgrade', 'fix', 'repair', 'improve',
        'optimize', 'enhance', 'refactor', 'modernize', 'support'
    ]
}

# Project status indicators
PROJECT_STATUS = {
    'planning': [
        'plan', 'design', 'concept', 'idea', 'proposal', 'draft',
        'outline', 'sketch', 'blueprint', 'framework'
    ],
    'in_progress': [
        'working on', 'developing', 'building', 'creating', 'implementing',
        'currently', 'ongoing', 'active', 'underway', 'in development'
    ],
    'completed': [
        'done', 'finished', 'completed', 'accomplished', 'achieved',
        'finalized', 'delivered', 'launched', 'published', 'released'
    ],
    'on_hold': [
        'paused', 'suspended', 'delayed', 'postponed', 'waiting',
        'blocked', 'stuck', 'halted', 'stopped'
    ],
    'cancelled': [
        'cancelled', 'abandoned', 'terminated', 'discontinued', 'dropped',
        'scrapped', 'killed', 'ended'
    ]
}

# Project complexity indicators
COMPLEXITY_INDICATORS = {
    'simple': [
        'simple', 'basic', 'straightforward', 'easy', 'quick', 'minor',
        'small', 'minimal', 'elementary', 'fundamental'
    ],
    'medium': [
        'moderate', 'standard', 'typical', 'normal', 'average', 'reasonable',
        'manageable', 'feasible', 'practical', 'realistic'
    ],
    'complex': [
        'complex', 'complicated', 'advanced', 'sophisticated', 'intricate',
        'challenging', 'difficult', 'ambitious', 'comprehensive', 'extensive'
    ]
}

# Project domain keywords
PROJECT_DOMAINS = {
    'software': [
        'software', 'application', 'app', 'program', 'system', 'platform',
        'website', 'web app', 'mobile app', 'desktop app', 'api', 'database'
    ],
    'hardware': [
        'hardware', 'device', 'equipment', 'machine', 'component', 'circuit',
        'electronics', 'robotics', 'automation', 'sensor', 'controller'
    ],
    'business': [
        'business', 'company', 'startup', 'enterprise', 'organization',
        'venture', 'initiative', 'campaign', 'strategy', 'operation'
    ],
    'research': [
        'research', 'study', 'experiment', 'investigation', 'analysis',
        'survey', 'paper', 'thesis', 'dissertation', 'publication'
    ],
    'creative': [
        'creative', 'art', 'design', 'content', 'media', 'production',
        'story', 'video', 'audio', 'graphic', 'visual'
    ],
    'education': [
        'education', 'course', 'training', 'tutorial', 'lesson', 'workshop',
        'curriculum', 'program', 'learning', 'teaching', 'instruction'
    ]
}

def extract_projects(text: str) -> List[Dict[str, Any]]:
    """Extract project information from text using pattern matching."""
    projects = []
    text_lower = text.lower()

    # Check for project keywords
    project_matches = {}
    for category, keywords in PROJECT_KEYWORDS.items():
        matches = []
        for keyword in keywords:
            if keyword in text_lower:
                matches.append(keyword)
        if matches:
            project_matches[category] = matches

    if project_matches:
        # Determine project status
        status = 'unknown'
        for status_name, indicators in PROJECT_STATUS.items():
            if any(indicator in text_lower for indicator in indicators):
                status = status_name
                break

        # Determine complexity
        complexity = 'medium'
        for comp_name, indicators in COMPLEXITY_INDICATORS.items():
            if any(indicator in text_lower for indicator in indicators):
                complexity = comp_name
                break

        # Determine domain
        domain = 'general'
        for domain_name, keywords in PROJECT_DOMAINS.items():
            if any(keyword in text_lower for keyword in keywords):
                domain = domain_name
                break

        # Extract project name/title
        project_name = extract_project_name(text)

        # Calculate project strength
        strength = sum(len(matches) for matches in project_matches.values())

        # Extract context
        context_start = max(0, text_lower.find(list(project_matches.values())[0][0]) - 100)
        context_end = min(len(text), text_lower.find(list(project_matches.values())[-1][-1]) + 100)
        context = text[context_start:context_end].strip()

        projects.append({
            'name': project_name,
            'domain': domain,
            'status': status,
            'complexity': complexity,
            'strength': strength,
            'context': context,
            'categories': list(project_matches.keys())
        })

    return projects

def extract_project_name(text: str) -> str:
    """Extract project name from text."""
    # Look for common project naming patterns
    patterns = [
        r'project\s+["\']([^"\']+)["\']',
        r'["\']([^"\']+)\s+project["\']',
        r'building\s+["\']([^"\']+)["\']',
        r'creating\s+["\']([^"\']+)["\']',
        r'developing\s+["\']([^"\']+)["\']',
        r'project:\s*([^\n]+)',
        r'goal:\s*([^\n]+)',
        r'objective:\s*([^\n]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: extract first meaningful phrase
    sentences = text.split('.')
    for sentence in sentences[:3]:  # Check first 3 sentences
        words = sentence.strip().split()
        if len(words) >= 3 and len(words) <= 8:
            return ' '.join(words)

    return "Unnamed Project"

def generate_project_id(conversation_id: str, project: Dict[str, Any], index: int) -> str:
    """Generate unique project ID."""
    # Create hash from conversation ID, domain, and name
    hash_input = f"{conversation_id}_{project['domain']}_{project['name'][:30]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"PROJECT_{conversation_id}_{project['domain'].upper()}_{index:03d}_{hash_value}"

def process_conversation(file_path: str, conversation_id: str) -> List[Dict[str, Any]]:
    """Process a single conversation file for projects."""
    projects = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process each message
        for message in data.get('messages', []):
            content = message.get('content', '')
            if not content or not isinstance(content, str):
                continue

            # Extract projects from message content
            message_projects = extract_projects(content)

            for i, project in enumerate(message_projects):
                project_id = generate_project_id(conversation_id, project, len(projects))

                projects.append({
                    'project_id': project_id,
                    'conversation_id': conversation_id,
                    'name': project['name'],
                    'domain': project['domain'],
                    'status': project['status'],
                    'complexity': project['complexity'],
                    'strength': project['strength'],
                    'context': project['context'],
                    'categories': '; '.join(project['categories']),
                    'message_index': data['messages'].index(message),
                    'role': message.get('role', 'unknown'),
                    'timestamp': message.get('timestamp', ''),
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
            'project_id', 'conversation_id', 'name', 'domain', 'status',
            'complexity', 'strength', 'context', 'categories', 'message_index',
            'role', 'timestamp', 'file_path'
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

    # Count by domain
    domain_counts = defaultdict(int)
    status_counts = defaultdict(int)
    complexity_counts = defaultdict(int)
    strength_distribution = defaultdict(int)

    for project in projects:
        domain_counts[project['domain']] += 1
        status_counts[project['status']] += 1
        complexity_counts[project['complexity']] += 1
        strength_distribution[round(project['strength'])] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Project ID Generation Summary\n\n")
        f.write(f"Total projects generated: {len(projects)}\n")
        f.write(f"Total conversations processed: {total_conversations}\n\n")

        f.write("## ID Format\n\n")
        f.write("Format: `PROJECT_CONVID_DOMAIN_INDEX_HASH`\n")
        f.write("- CONVID: Conversation ID\n")
        f.write("- DOMAIN: Project domain\n")
        f.write("- INDEX: Project index within conversation\n")
        f.write("- HASH: 6-character hash of project name\n\n")

        f.write("## Domain Distribution\n\n")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(projects)) * 100
            f.write(f"- **{domain.title()}**: {count} projects ({percentage:.1f}%)\n")

        f.write("\n## Status Distribution\n\n")
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(projects)) * 100
            f.write(f"- **{status.replace('_', ' ').title()}**: {count} projects ({percentage:.1f}%)\n")

        f.write("\n## Complexity Distribution\n\n")
        for complexity, count in sorted(complexity_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(projects)) * 100
            f.write(f"- **{complexity.title()}**: {count} projects ({percentage:.1f}%)\n")

        f.write("\n## Strength Distribution\n\n")
        for strength in sorted(strength_distribution.keys()):
            count = strength_distribution[strength]
            percentage = (count / len(projects)) * 100
            f.write(f"- **Strength {strength}**: {count} projects ({percentage:.1f}%)\n")

        f.write("\n## Recent Projects\n\n")
        recent_projects = sorted(projects, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        for project in recent_projects:
            f.write(f"- **{project['project_id']}**: {project['name']} ({project['domain']}, {project['status']})\n")

if __name__ == "__main__":
    main()
