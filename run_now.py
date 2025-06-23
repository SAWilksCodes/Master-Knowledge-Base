#!/usr/bin/env python3

# Import the semantic anchor system and run it
import sys
import os
sys.path.insert(0, os.getcwd())

from semantic_anchor_system import SemanticAnchorSystem
import json
from pathlib import Path

def main():
    print("🚀 SEMANTIC ANCHOR SYSTEM - PHASE 1: ANCHOR THE DATA")
    print("=" * 60)

    # Initialize system
    system = SemanticAnchorSystem(
        embedding_backend="sentence_transformers",
        cache_embeddings=True,
        output_dir="semantic_anchors"
    )

    # Load data
    print("\n📊 Loading enhanced analysis results...")
    system.load_enhanced_word_results()
    system.load_enhanced_topic_results()
    system.load_sentences_data(max_sentences=1000)  # Start with 1000 for testing

    # Generate embeddings (limited for demo)
    print("\n🧠 Generating semantic embeddings...")
    system.generate_embeddings(batch_size=25, max_items=500)

    # Export data
    print("\n💾 Exporting semantic anchors...")
    jsonl_file = system.export_to_jsonl()
    csv_file = system.export_to_csv()

    # Generate summary report
    print("\n📋 Generating summary report...")
    report = system.generate_summary_report()

    print("\n✅ SEMANTIC ANCHOR SYSTEM PHASE 1 COMPLETE!")
    print(f"📁 Word Anchors: {report['data_summary']['word_anchors']}")
    print(f"📝 Sentence Anchors: {report['data_summary']['sentence_anchors']}")
    print(f"💬 Conversations: {report['data_summary']['conversations']}")
    print(f"🧠 Embeddings Generated: {report['embedding_stats']['total_embeddings']}")
    print(f"📤 Exported to: {jsonl_file.name}, {csv_file.name}")

if __name__ == "__main__":
    main()
