#!/usr/bin/env python3
"""
🚀 EXECUTE SEMANTIC ANCHOR SYSTEM - PHASE 1: ANCHOR THE DATA
Complete execution with full feature set
"""

print("🚀 SEMANTIC ANCHOR SYSTEM - PHASE 1: ANCHOR THE DATA")
print("=" * 60)
print("🎯 Creating semantic embedding anchors from enhanced analysis results")
print("🧠 Using sentence-transformers for local embeddings")
print("🗄️ Setting up ChromaDB vector database")
print("📊 Processing conversation data with rich metadata")
print()

# Execute the main semantic anchor system
try:
    from semantic_anchor_system import SemanticAnchorSystem
    import json
    from pathlib import Path
    from datetime import datetime

    # Initialize system with full capabilities
    print("⚙️ Initializing Semantic Anchor System...")
    system = SemanticAnchorSystem(
        embedding_backend="sentence_transformers",  # Local model for reliability
        cache_embeddings=True,                      # Cache for efficiency
        output_dir="semantic_anchors"               # Output directory
    )

    # Load all enhanced analysis data
    print("\n📊 Loading enhanced analysis results...")
    print("   - Loading enhanced word analysis...")
    system.load_enhanced_word_results()

    print("   - Loading enhanced topic classifications...")
    system.load_enhanced_topic_results()

    print("   - Loading sentence data (limited to 2000 for initial run)...")
    system.load_sentences_data(max_sentences=2000)

    # Generate embeddings
    print("\n🧠 Generating semantic embeddings...")
    print("   - Processing word anchors...")
    print("   - Processing sentence anchors...")
    system.generate_embeddings(batch_size=50, max_items=1000)

    # Export semantic anchors
    print("\n💾 Exporting semantic anchors...")
    jsonl_file = system.export_to_jsonl()
    csv_file = system.export_to_csv()

    # Setup vector database
    print("\n🗄️ Setting up vector database...")
    try:
        collection = system.setup_chromadb()
        if collection:
            system.ingest_to_chromadb(collection, batch_size=50)
            print("   ✅ ChromaDB setup and ingestion completed")
        else:
            print("   ⚠️ ChromaDB setup skipped (optional)")
    except Exception as e:
        print(f"   ⚠️ ChromaDB setup failed (optional): {e}")

    # Generate comprehensive report
    print("\n📋 Generating summary report...")
    report = system.generate_summary_report()

    # Display results
    print("\n" + "=" * 60)
    print("✅ SEMANTIC ANCHOR SYSTEM PHASE 1 COMPLETE!")
    print("=" * 60)

    print(f"📁 Word Anchors Created: {report['data_summary']['word_anchors']}")
    print(f"📝 Sentence Anchors Created: {report['data_summary']['sentence_anchors']}")
    print(f"💬 Conversations Processed: {report['data_summary']['conversations']}")
    print(f"🧠 Embeddings Generated: {report['embedding_stats']['total_embeddings']}")
    print(f"📤 Data Exported to: {jsonl_file.name}, {csv_file.name}")

    # Show top statistics
    if 'word_anchor_stats' in report:
        word_stats = report['word_anchor_stats']
        print(f"\n📊 Word Analysis:")
        print(f"   - Total words: {word_stats.get('total_words', 0)}")
        print(f"   - Embeddings: {word_stats.get('embeddings_generated', 0)}")
        if 'pos_distribution' in word_stats:
            top_pos = list(word_stats['pos_distribution'].items())[:3]
            print(f"   - Top POS: {', '.join([f'{k}({v})' for k, v in top_pos])}")

    if 'sentence_anchor_stats' in report:
        sent_stats = report['sentence_anchor_stats']
        print(f"\n📝 Sentence Analysis:")
        print(f"   - Total sentences: {sent_stats.get('total_sentences', 0)}")
        print(f"   - Embeddings: {sent_stats.get('embeddings_generated', 0)}")
        if 'topic_distribution' in sent_stats:
            top_topics = list(sent_stats['topic_distribution'].items())[:3]
            print(f"   - Top topics: {', '.join([f'{k}({v})' for k, v in top_topics])}")

    print(f"\n🎯 PHASE 1 COMPLETE - Ready for Phase 2: Semantic Intelligence Engine")
    print(f"📁 All results saved to: semantic_anchors/")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all dependencies are installed:")
    print("   pip install sentence-transformers chromadb openai tqdm numpy")

except Exception as e:
    print(f"❌ Execution error: {e}")
    import traceback
    traceback.print_exc()
    print("\n🔧 Check the error above and try again")

print("\n" + "=" * 60)
