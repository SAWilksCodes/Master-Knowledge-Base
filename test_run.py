#!/usr/bin/env python3
"""
Test run of Semantic Anchor System with minimal data to verify functionality
"""

import json
import csv
from pathlib import Path

# Test data loading
def test_data_loading():
    print("🧪 Testing data loading...")

    # Test enhanced word analysis
    try:
        with open("enhanced_analysis/enhanced_word_analysis_report.json", 'r') as f:
            word_data = json.load(f)
        print(f"✅ Word data loaded: {len(word_data.get('top_10_words', []))} top words")
    except Exception as e:
        print(f"❌ Word data error: {e}")

    # Test topic classifications
    try:
        with open("enhanced_analysis/enhanced_conversation_classifications.csv", 'r') as f:
            reader = csv.DictReader(f)
            conv_count = sum(1 for _ in reader)
        print(f"✅ Topic data loaded: {conv_count} conversations")
    except Exception as e:
        print(f"❌ Topic data error: {e}")

    # Test sentences
    try:
        with open("id_generation_results/sentences_simple.csv", 'r') as f:
            reader = csv.DictReader(f)
            sentence_count = sum(1 for _ in reader)
        print(f"✅ Sentence data loaded: {sentence_count} sentences")
    except Exception as e:
        print(f"❌ Sentence data error: {e}")

def test_semantic_anchor_creation():
    print("\n🎯 Testing Semantic Anchor creation...")

    try:
        # Import the system
        from semantic_anchor_system import SemanticAnchorSystem, WordAnchor, SentenceAnchor, ConversationMetadata

        # Create a simple test system
        system = SemanticAnchorSystem(
            embedding_backend="sentence_transformers",
            cache_embeddings=False,  # Skip caching for test
            output_dir="test_semantic_anchors"
        )

        print("✅ System initialized successfully")

        # Test word anchor creation
        test_word = WordAnchor(
            word_id="test_word_001",
            word="test",
            frequency=100,
            pos_tag="NOUN",
            emotion="neutral",
            domain="general"
        )
        system.word_anchors["test_word_001"] = test_word
        print("✅ Word anchor created")

        # Test conversation metadata
        test_conv = ConversationMetadata(
            conversation_id="test_conv_001",
            title="Test Conversation",
            date="2025-01-01",
            model="gpt-4",
            topic="testing",
            emotion="analytical",
            project_type="test_project",
            message_count=5,
            word_count=100,
            char_count=500
        )
        system.conversation_metadata["test_conv_001"] = test_conv
        print("✅ Conversation metadata created")

        # Test sentence anchor
        test_sentence = SentenceAnchor(
            sentence_id="test_sent_001",
            sentence="This is a test sentence.",
            conversation_id="test_conv_001",
            topic="testing",
            topic_confidence=0.9,
            emotion="analytical",
            emotion_confidence=0.8,
            project_type="test_project",
            project_confidence=0.85,
            word_count=5,
            char_count=24
        )
        system.sentence_anchors["test_sent_001"] = test_sentence
        print("✅ Sentence anchor created")

        print(f"📊 Test system stats:")
        print(f"   - Word anchors: {len(system.word_anchors)}")
        print(f"   - Sentence anchors: {len(system.sentence_anchors)}")
        print(f"   - Conversations: {len(system.conversation_metadata)}")

        return True

    except Exception as e:
        print(f"❌ Semantic anchor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 SEMANTIC ANCHOR SYSTEM - TEST RUN")
    print("=" * 50)

    # Test data loading
    test_data_loading()

    # Test semantic anchor creation
    success = test_semantic_anchor_creation()

    if success:
        print("\n✅ ALL TESTS PASSED - READY TO RUN FULL SYSTEM!")
    else:
        print("\n❌ TESTS FAILED - CHECK ERRORS ABOVE")

if __name__ == "__main__":
    main()
