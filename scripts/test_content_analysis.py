#!/usr/bin/env python3
"""
Test script to analyze conversation content and debug pattern matching issues.
"""

import json
import os
import re

def test_conversation_content(input_dir: str, sample_size: int = 5):
    """Test conversation content to understand why pattern matching isn't working."""

    print("🔍 Testing conversation content...")
    print(f"📁 Input directory: {input_dir}")

    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Directory does not exist: {input_dir}")
        return

    # Get sample files
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')][:sample_size]
    print(f"📄 Found {len(files)} files to test")

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"\n📄 Testing: {filename}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            messages = data.get('messages', [])
            print(f"   Messages: {len(messages)}")

            # Test first few messages
            for i, message in enumerate(messages[:3]):
                content = message.get('content', '')
                if content and isinstance(content, str):
                    print(f"   Message {i}: {content[:200]}...")

                    # Test some basic patterns
                    text_lower = content.lower()

                    # Test emotion keywords
                    emotion_words = ['happy', 'sad', 'excited', 'frustrated', 'great', 'wonderful']
                    found_emotions = [word for word in emotion_words if word in text_lower]
                    if found_emotions:
                        print(f"     Emotions found: {found_emotions}")

                    # Test concept keywords
                    concept_words = ['technology', 'business', 'research', 'education', 'creative']
                    found_concepts = [word for word in concept_words if word in text_lower]
                    if found_concepts:
                        print(f"     Concepts found: {found_concepts}")

                    # Test project keywords
                    project_words = ['plan', 'develop', 'create', 'build', 'project']
                    found_projects = [word for word in project_words if word in text_lower]
                    if found_projects:
                        print(f"     Projects found: {found_projects}")

                    # Test metaphor keywords
                    metaphor_words = ['like', 'as if', 'journey', 'path', 'build']
                    found_metaphors = [word for word in metaphor_words if word in text_lower]
                    if found_metaphors:
                        print(f"     Metaphors found: {found_metaphors}")

                    # Test thread keywords
                    thread_words = ['continuing', 'following up', 'as mentioned', 'previously']
                    found_threads = [word for word in thread_words if word in text_lower]
                    if found_threads:
                        print(f"     Threads found: {found_threads}")
                else:
                    print(f"   Message {i}: No content or not string")

        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    test_conversation_content("processed_20250621_032815/01_split_json", 3)
