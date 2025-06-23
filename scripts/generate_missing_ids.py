#!/usr/bin/env python3
"""
Generate Missing ID Types
Generate the missing ID files identified in validation:
- metaphor_ids.csv
- project_ids.csv
- conversation_ids.csv
"""

import os
import json
import pandas as pd
import uuid
import hashlib
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any
import re

class MissingIDGenerator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_conversation_ids(self) -> str:
        """Generate conversation IDs from existing conversation files."""
        print("🔄 Generating conversation IDs...")

        conversation_data = []

        # Look for conversation files in various locations
        conversation_locations = [
            "integrated_20250623_011924/01_conversations/new_only",
            "integrated_20250623_011924/01_conversations/new_only/split",
            "long_convos",
            "split_chats"
        ]

        for location in conversation_locations:
            loc_path = Path(location)
            if loc_path.exists():
                for json_file in loc_path.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)

                        # Extract conversation metadata
                        title = conv_data.get('title', Path(json_file).stem)
                        create_time = conv_data.get('create_time', 0)
                        conversation_id = conv_data.get('conversation_id', '')

                        # Generate unique ID
                        hash_input = f"{title}_{create_time}_{conversation_id}"
                        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

                        if create_time:
                            try:
                                dt = datetime.fromtimestamp(create_time)
                                date_str = dt.strftime('%Y%m%d')
                            except:
                                date_str = '00000000'
                        else:
                            date_str = '00000000'

                        conv_id = f"CONV_{date_str}_{hash_id}"

                        conversation_data.append({
                            'conversation_id': conv_id,
                            'title': title,
                            'date': date_str,
                            'create_time': create_time,
                            'original_id': conversation_id,
                            'source_file': str(json_file),
                            'message_count': len(conv_data.get('mapping', {}))
                        })

                    except Exception as e:
                        print(f"⚠️ Error processing {json_file}: {e}")
                        continue

        if conversation_data:
            df = pd.DataFrame(conversation_data)
            output_file = self.output_dir / 'conversation_ids.csv'
            df.to_csv(output_file, index=False)
            print(f"✅ Generated {len(conversation_data)} conversation IDs")
            return str(output_file)
        else:
            print("⚠️ No conversation files found")
            return ""

    def generate_metaphor_ids(self) -> str:
        """Generate metaphor IDs by analyzing text for figurative language."""
        print("🔄 Generating metaphor IDs...")

        metaphor_data = []
        metaphor_patterns = [
            r'\b(is|are|was|were)\s+(like|as)\s+',
            r'\b(like|as)\s+\w+',
            r'\b(metaphor|analogy|simile)\b',
            r'\b(represents|symbolizes|stands for)\b',
            r'\b(imagine|picture|think of)\s+\w+\s+as\b'
        ]

        # Load sentence data to analyze for metaphors
        sentence_file = Path("integrated_20250623_011924/id_generation_results_new/sentence_ids.csv")
        if sentence_file.exists():
            sentences_df = pd.read_csv(sentence_file)

            for _, row in sentences_df.head(1000).iterrows():  # Sample first 1000 sentences
                text = str(row.get('text', ''))

                # Check for metaphor patterns
                for pattern in metaphor_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        metaphor_id = f"metaphor_{uuid.uuid4().hex[:8]}"
                        metaphor_text = match.group(0)

                        metaphor_data.append({
                            'metaphor_id': metaphor_id,
                            'metaphor_text': metaphor_text,
                            'full_context': text[:200],
                            'pattern_type': pattern,
                            'sentence_id': row.get('sentence_id', ''),
                            'confidence': 0.7,  # Placeholder confidence
                            'metaphor_type': 'simile' if 'like' in metaphor_text.lower() or 'as' in metaphor_text.lower() else 'metaphor'
                        })

        # Add some common metaphor templates
        common_metaphors = [
            "knowledge is power",
            "time is money",
            "life is a journey",
            "love is a battlefield",
            "the world is a stage",
            "ideas are seeds",
            "problems are puzzles",
            "learning is building",
            "thinking is exploring",
            "creativity is a spark"
        ]

        for metaphor in common_metaphors:
            metaphor_id = f"metaphor_{uuid.uuid4().hex[:8]}"
            metaphor_data.append({
                'metaphor_id': metaphor_id,
                'metaphor_text': metaphor,
                'full_context': f"Common metaphor: {metaphor}",
                'pattern_type': 'template',
                'sentence_id': '',
                'confidence': 0.9,
                'metaphor_type': 'template'
            })

        if metaphor_data:
            df = pd.DataFrame(metaphor_data)
            output_file = self.output_dir / 'metaphor_ids.csv'
            df.to_csv(output_file, index=False)
            print(f"✅ Generated {len(metaphor_data)} metaphor IDs")
            return str(output_file)
        else:
            print("⚠️ No metaphors found")
            return ""

    def generate_project_ids(self) -> str:
        """Generate project IDs by identifying project-related content."""
        print("🔄 Generating project IDs...")

        project_data = []

        # Project indicators
        project_keywords = [
            'project', 'system', 'application', 'platform', 'tool', 'framework',
            'engine', 'module', 'component', 'service', 'api', 'database',
            'pipeline', 'workflow', 'automation', 'integration', 'deployment'
        ]

        # Load sentence data to identify projects
        sentence_file = Path("integrated_20250623_011924/id_generation_results_new/sentence_ids.csv")
        if sentence_file.exists():
            sentences_df = pd.read_csv(sentence_file)

            for _, row in sentences_df.head(2000).iterrows():  # Sample first 2000 sentences
                text = str(row.get('text', '')).lower()

                # Check for project indicators
                for keyword in project_keywords:
                    if keyword in text:
                        project_id = f"project_{uuid.uuid4().hex[:8]}"

                        # Extract project name (simplified)
                        words = text.split()
                        try:
                            keyword_idx = words.index(keyword)
                            if keyword_idx > 0:
                                project_name = words[keyword_idx - 1] + " " + keyword
                            else:
                                project_name = keyword
                        except:
                            project_name = keyword

                        project_data.append({
                            'project_id': project_id,
                            'project_name': project_name.title(),
                            'project_type': keyword,
                            'description': text[:100],
                            'sentence_id': row.get('sentence_id', ''),
                            'confidence': 0.8,
                            'status': 'identified'
                        })
                        break  # Only count once per sentence

        # Add some common project templates
        common_projects = [
            "Semantic Intelligence Engine",
            "Knowledge Graph Builder",
            "Conversation Analyzer",
            "ID Generation System",
            "Visualization Layer",
            "Query Engine",
            "Data Pipeline",
            "Analysis Framework",
            "Integration Platform",
            "Export System"
        ]

        for project in common_projects:
            project_id = f"project_{uuid.uuid4().hex[:8]}"
            project_data.append({
                'project_id': project_id,
                'project_name': project,
                'project_type': 'system',
                'description': f"Core system component: {project}",
                'sentence_id': '',
                'confidence': 0.95,
                'status': 'template'
            })

        if project_data:
            df = pd.DataFrame(project_data)
            output_file = self.output_dir / 'project_ids.csv'
            df.to_csv(output_file, index=False)
            print(f"✅ Generated {len(project_data)} project IDs")
            return str(output_file)
        else:
            print("⚠️ No projects found")
            return ""

    def update_id_generation_summary(self) -> str:
        """Update the ID generation summary with new counts."""
        print("📊 Updating ID generation summary...")

        summary_file = Path("integrated_20250623_011924/id_generation_results_new/id_generation_summary.json")

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                "generation_timestamp": datetime.now().isoformat(),
                "total_id_types": 32,
                "id_counts": {}
            }

        # Count new files
        for filename in ['conversation_ids.csv', 'metaphor_ids.csv', 'project_ids.csv']:
            filepath = self.output_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                id_type = filename.replace('_ids.csv', '')
                summary['id_counts'][f'{id_type}_ids'] = len(df)

        # Update timestamp
        summary['generation_timestamp'] = datetime.now().isoformat()

        # Save updated summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Updated ID generation summary")
        return str(summary_file)

def main():
    parser = argparse.ArgumentParser(description="Generate missing ID types")
    parser.add_argument("--input-dir", default="integrated_20250623_011924/01_conversations",
                       help="Input directory with conversation data")
    parser.add_argument("--output-dir", default="integrated_20250623_011924/id_generation_results_new",
                       help="Output directory for ID files")

    args = parser.parse_args()

    generator = MissingIDGenerator(args.input_dir, args.output_dir)

    # Generate missing IDs
    conversation_file = generator.generate_conversation_ids()
    metaphor_file = generator.generate_metaphor_ids()
    project_file = generator.generate_project_ids()

    # Update summary
    summary_file = generator.update_id_generation_summary()

    print(f"\n✅ Missing ID generation complete!")
    print(f"🔄 Conversation IDs: {conversation_file}")
    print(f"🔄 Metaphor IDs: {metaphor_file}")
    print(f"🔄 Project IDs: {project_file}")
    print(f"📊 Summary updated: {summary_file}")

if __name__ == "__main__":
    main()
