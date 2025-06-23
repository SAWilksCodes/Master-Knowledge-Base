#!/usr/bin/env python3
"""
Semantic Anchor System - Phase 1: Anchor the Data
Creates semantic embedding anchors from enhanced analysis results with vector database integration.
"""

import json
import csv
import os
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
import re

# Optional imports with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Sentence Transformers not available. Install with: pip install sentence-transformers")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

@dataclass
class WordAnchor:
    """Represents a semantic anchor for a word with rich metadata."""
    word_id: str
    word: str
    frequency: int
    pos_tag: str
    emotion: str
    domain: str
    embedding: Optional[List[float]] = None
    conversations: List[str] = None

    def __post_init__(self):
        if self.conversations is None:
            self.conversations = []

@dataclass
class SentenceAnchor:
    """Represents a semantic anchor for a sentence with conversation metadata."""
    sentence_id: str
    sentence: str
    conversation_id: str
    topic: str
    topic_confidence: float
    emotion: str
    emotion_confidence: float
    project_type: str
    project_confidence: float
    word_count: int
    char_count: int
    embedding: Optional[List[float]] = None
    secondary_topic: Optional[str] = None
    secondary_emotion: Optional[str] = None

@dataclass
class ConversationMetadata:
    """Metadata for a conversation."""
    conversation_id: str
    title: str
    date: str
    model: str
    topic: str
    emotion: str
    project_type: str
    message_count: int
    word_count: int
    char_count: int

class SemanticAnchorSystem:
    """
    Comprehensive system for creating and managing semantic anchors from enhanced analysis results.
    """

    def __init__(self,
                 embedding_backend: str = "sentence_transformers",
                 openai_api_key: Optional[str] = None,
                 cache_embeddings: bool = True,
                 output_dir: str = "semantic_anchors"):
        """
        Initialize the Semantic Anchor System.

        Args:
            embedding_backend: "openai", "sentence_transformers", or "both"
            openai_api_key: OpenAI API key if using OpenAI backend
            cache_embeddings: Whether to cache embeddings to avoid recomputation
            output_dir: Directory to save outputs
        """
        self.embedding_backend = embedding_backend
        self.cache_embeddings = cache_embeddings
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize embedding models
        self._init_embedding_models(openai_api_key)

        # Data storage
        self.word_anchors: Dict[str, WordAnchor] = {}
        self.sentence_anchors: Dict[str, SentenceAnchor] = {}
        self.conversation_metadata: Dict[str, ConversationMetadata] = {}

        # Embedding cache
        self.embedding_cache: Dict[str, List[float]] = {}
        if cache_embeddings:
            self._load_embedding_cache()

    def _init_embedding_models(self, openai_api_key: Optional[str]):
        """Initialize embedding models based on backend selection."""
        self.openai_client = None
        self.sentence_model = None

        if self.embedding_backend in ["openai", "both"]:
            if OPENAI_AVAILABLE and openai_api_key:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                self.logger.info("OpenAI client initialized")
            else:
                self.logger.warning("OpenAI backend requested but not available or no API key")

        if self.embedding_backend in ["sentence_transformers", "both"]:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Sentence Transformers model loaded")
            else:
                self.logger.warning("Sentence Transformers backend requested but not available")

    def _load_embedding_cache(self):
        """Load cached embeddings if available."""
        cache_file = self.output_dir / "embedding_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.embedding_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding cache: {e}")

    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        if self.cache_embeddings:
            cache_file = self.output_dir / "embedding_cache.json"
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.embedding_cache, f)
                self.logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
            except Exception as e:
                self.logger.warning(f"Failed to save embedding cache: {e}")

    def _get_embedding(self, text: str, backend: str = None) -> Optional[List[float]]:
        """Get embedding for text using specified backend."""
        if backend is None:
            backend = self.embedding_backend

        # Check cache first
        cache_key = f"{backend}:{hashlib.md5(text.encode()).hexdigest()}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        embedding = None

        if backend == "openai" and self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embedding = response.data[0].embedding
            except Exception as e:
                self.logger.error(f"OpenAI embedding failed: {e}")

        elif backend == "sentence_transformers" and self.sentence_model:
            try:
                embedding = self.sentence_model.encode(text).tolist()
            except Exception as e:
                self.logger.error(f"Sentence Transformers embedding failed: {e}")

        # Cache the embedding
        if embedding and self.cache_embeddings:
            self.embedding_cache[cache_key] = embedding

        return embedding

    def load_enhanced_word_results(self, word_results_file: str = "enhanced_analysis/enhanced_word_analysis_report.json"):
        """Load enhanced word analysis results to create word anchors."""
        try:
            with open(word_results_file, 'r', encoding='utf-8') as f:
                word_data = json.load(f)

            self.logger.info(f"Loading word results from {word_results_file}")

            # Extract top words with rich metadata
            top_words = word_data.get('top_10_words', [])

            # Also load from the detailed CSV if available
            csv_file = "enhanced_analysis/words_enhanced_analysis.csv"
            if Path(csv_file).exists():
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        word_anchor = WordAnchor(
                            word_id=row.get('word_id', f"word_{hashlib.md5(row['word'].encode()).hexdigest()[:12]}"),
                            word=row['word'],
                            frequency=int(row.get('frequency', 0)),
                            pos_tag=row.get('pos', 'UNKNOWN'),
                            emotion=row.get('emotional_tone', 'neutral'),
                            domain=row.get('technical_domain', 'general')
                        )
                        self.word_anchors[word_anchor.word_id] = word_anchor
            else:
                # Fall back to top words from JSON
                for word_info in top_words:
                    word_anchor = WordAnchor(
                        word_id=word_info.get('word_id', f"word_{hashlib.md5(word_info['word'].encode()).hexdigest()[:12]}"),
                        word=word_info['word'],
                        frequency=word_info.get('frequency', 0),
                        pos_tag=word_info.get('pos', 'UNKNOWN'),
                        emotion=word_info.get('emotional_tone', 'neutral'),
                        domain=word_info.get('technical_domain', 'general')
                    )
                    self.word_anchors[word_anchor.word_id] = word_anchor

            self.logger.info(f"Created {len(self.word_anchors)} word anchors")

        except Exception as e:
            self.logger.error(f"Failed to load word results: {e}")

    def _find_word_attribute(self, word: str, distribution: Dict, default: str) -> str:
        """Find the most appropriate attribute for a word from distribution data."""
        # This is a simplified approach - in practice, you'd want more sophisticated mapping
        for attr, count in distribution.items():
            if word.lower() in attr.lower() or attr.lower() in word.lower():
                return attr
        return default

    def load_enhanced_topic_results(self, topic_results_file: str = "enhanced_analysis/enhanced_conversation_classifications.csv"):
        """Load enhanced topic classification results to create sentence anchors."""
        try:
            sentence_anchors = []
            conversation_metadata = {}

            with open(topic_results_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    conv_id = row['conversation_id']

                    # Create conversation metadata
                    conversation_metadata[conv_id] = ConversationMetadata(
                        conversation_id=conv_id,
                        title=row.get('title', ''),
                        date=row.get('date', ''),
                        model=row.get('model', 'unknown'),
                        topic=row.get('primary_topic', 'general_inquiry'),
                        emotion=row.get('primary_emotion', 'neutral'),
                        project_type=row.get('primary_project_type', 'general'),
                        message_count=int(row.get('message_count', 0)),
                        word_count=int(row.get('word_count', 0)),
                        char_count=int(row.get('char_count', 0))
                    )

            self.conversation_metadata = conversation_metadata
            self.logger.info(f"Loaded metadata for {len(conversation_metadata)} conversations")

        except Exception as e:
            self.logger.error(f"Failed to load topic results: {e}")

    def load_sentences_data(self, sentences_file: str = "id_generation_results/sentences_simple.csv", max_sentences: int = 10000):
        """Load sentences data to create sentence anchors."""
        try:
            sentence_count = 0
            with open(sentences_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if sentence_count >= max_sentences:
                        break

                    sentence_id = row['sentence_id']
                    conversation_id = row['conversation_id']
                    sentence_text = row['sentence']

                    # Get conversation metadata
                    conv_meta = self.conversation_metadata.get(conversation_id)
                    if not conv_meta:
                        continue

                    # Create sentence anchor
                    sentence_anchor = SentenceAnchor(
                        sentence_id=sentence_id,
                        sentence=sentence_text,
                        conversation_id=conversation_id,
                        topic=conv_meta.topic,
                        topic_confidence=0.8,  # Default confidence
                        emotion=conv_meta.emotion,
                        emotion_confidence=0.7,  # Default confidence
                        project_type=conv_meta.project_type,
                        project_confidence=0.75,  # Default confidence
                        word_count=len(sentence_text.split()),
                        char_count=len(sentence_text)
                    )

                    self.sentence_anchors[sentence_id] = sentence_anchor
                    sentence_count += 1

            self.logger.info(f"Created {len(self.sentence_anchors)} sentence anchors")

        except Exception as e:
            self.logger.error(f"Failed to load sentences data: {e}")

    def generate_embeddings(self, batch_size: int = 100, max_items: int = None):
        """Generate embeddings for all anchors."""
        self.logger.info("Starting embedding generation...")

        # Generate word embeddings
        word_items = list(self.word_anchors.items())
        if max_items:
            word_items = word_items[:max_items//2]

        self.logger.info(f"Generating embeddings for {len(word_items)} words...")
        for i, (word_id, word_anchor) in enumerate(word_items):
            if i % batch_size == 0:
                self.logger.info(f"Processing word batch {i//batch_size + 1}/{(len(word_items)-1)//batch_size + 1}")

            embedding = self._get_embedding(word_anchor.word)
            if embedding:
                word_anchor.embedding = embedding

        # Generate sentence embeddings
        sentence_items = list(self.sentence_anchors.items())
        if max_items:
            sentence_items = sentence_items[:max_items//2]

        self.logger.info(f"Generating embeddings for {len(sentence_items)} sentences...")
        for i, (sentence_id, sentence_anchor) in enumerate(sentence_items):
            if i % batch_size == 0:
                self.logger.info(f"Processing sentence batch {i//batch_size + 1}/{(len(sentence_items)-1)//batch_size + 1}")

            embedding = self._get_embedding(sentence_anchor.sentence)
            if embedding:
                sentence_anchor.embedding = embedding

        # Save embedding cache
        self._save_embedding_cache()

        self.logger.info("Embedding generation completed")

    def export_to_jsonl(self, filename: str = None):
        """Export anchors to JSONL format for vector database ingestion."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"semantic_anchors_{timestamp}.jsonl"

        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            # Export word anchors
            for word_anchor in self.word_anchors.values():
                if word_anchor.embedding:
                    record = {
                        'id': word_anchor.word_id,
                        'type': 'word',
                        'text': word_anchor.word,
                        'embedding': word_anchor.embedding,
                        'metadata': {
                            'frequency': word_anchor.frequency,
                            'pos_tag': word_anchor.pos_tag,
                            'emotion': word_anchor.emotion,
                            'domain': word_anchor.domain,
                            'conversations': word_anchor.conversations
                        }
                    }
                    f.write(json.dumps(record) + '\n')

            # Export sentence anchors
            for sentence_anchor in self.sentence_anchors.values():
                if sentence_anchor.embedding:
                    record = {
                        'id': sentence_anchor.sentence_id,
                        'type': 'sentence',
                        'text': sentence_anchor.sentence,
                        'embedding': sentence_anchor.embedding,
                        'metadata': {
                            'conversation_id': sentence_anchor.conversation_id,
                            'topic': sentence_anchor.topic,
                            'topic_confidence': sentence_anchor.topic_confidence,
                            'emotion': sentence_anchor.emotion,
                            'emotion_confidence': sentence_anchor.emotion_confidence,
                            'project_type': sentence_anchor.project_type,
                            'project_confidence': sentence_anchor.project_confidence,
                            'word_count': sentence_anchor.word_count,
                            'char_count': sentence_anchor.char_count,
                            'secondary_topic': sentence_anchor.secondary_topic,
                            'secondary_emotion': sentence_anchor.secondary_emotion
                        }
                    }
                    f.write(json.dumps(record) + '\n')

        self.logger.info(f"Exported semantic anchors to {output_file}")
        return output_file

    def export_to_csv(self, filename: str = None):
        """Export anchors to CSV format for analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"semantic_anchors_{timestamp}.csv"

        output_file = self.output_dir / filename

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'id', 'type', 'text', 'embedding_dim', 'frequency', 'pos_tag',
                'emotion', 'domain', 'conversation_id', 'topic', 'topic_confidence',
                'emotion_confidence', 'project_type', 'project_confidence',
                'word_count', 'char_count'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Export word anchors
            for word_anchor in self.word_anchors.values():
                if word_anchor.embedding:
                    writer.writerow({
                        'id': word_anchor.word_id,
                        'type': 'word',
                        'text': word_anchor.word,
                        'embedding_dim': len(word_anchor.embedding),
                        'frequency': word_anchor.frequency,
                        'pos_tag': word_anchor.pos_tag,
                        'emotion': word_anchor.emotion,
                        'domain': word_anchor.domain,
                        'conversation_id': '',
                        'topic': '',
                        'topic_confidence': '',
                        'emotion_confidence': '',
                        'project_type': '',
                        'project_confidence': '',
                        'word_count': '',
                        'char_count': ''
                    })

            # Export sentence anchors
            for sentence_anchor in self.sentence_anchors.values():
                if sentence_anchor.embedding:
                    writer.writerow({
                        'id': sentence_anchor.sentence_id,
                        'type': 'sentence',
                        'text': sentence_anchor.sentence[:100] + '...' if len(sentence_anchor.sentence) > 100 else sentence_anchor.sentence,
                        'embedding_dim': len(sentence_anchor.embedding),
                        'frequency': '',
                        'pos_tag': '',
                        'emotion': sentence_anchor.emotion,
                        'domain': '',
                        'conversation_id': sentence_anchor.conversation_id,
                        'topic': sentence_anchor.topic,
                        'topic_confidence': sentence_anchor.topic_confidence,
                        'emotion_confidence': sentence_anchor.emotion_confidence,
                        'project_type': sentence_anchor.project_type,
                        'project_confidence': sentence_anchor.project_confidence,
                        'word_count': sentence_anchor.word_count,
                        'char_count': sentence_anchor.char_count
                    })

        self.logger.info(f"Exported semantic anchors to {output_file}")
        return output_file

    def setup_chromadb(self, collection_name: str = "conversation_anchors"):
        """Set up ChromaDB collection for vector storage."""
        if not CHROMADB_AVAILABLE:
            self.logger.error("ChromaDB not available. Install with: pip install chromadb")
            return None

        try:
            # Initialize ChromaDB client
            chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.output_dir / "chromadb")
            ))

            # Create or get collection
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Semantic anchors from conversation analysis"}
            )

            self.logger.info(f"ChromaDB collection '{collection_name}' ready")
            return collection

        except Exception as e:
            self.logger.error(f"Failed to setup ChromaDB: {e}")
            return None

    def ingest_to_chromadb(self, collection, batch_size: int = 100):
        """Ingest anchors into ChromaDB collection."""
        if not collection:
            self.logger.error("No ChromaDB collection provided")
            return

        # Prepare data for ingestion
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        # Add word anchors
        for word_anchor in self.word_anchors.values():
            if word_anchor.embedding:
                ids.append(word_anchor.word_id)
                embeddings.append(word_anchor.embedding)
                documents.append(word_anchor.word)
                metadatas.append({
                    'type': 'word',
                    'frequency': word_anchor.frequency,
                    'pos_tag': word_anchor.pos_tag,
                    'emotion': word_anchor.emotion,
                    'domain': word_anchor.domain
                })

        # Add sentence anchors
        for sentence_anchor in self.sentence_anchors.values():
            if sentence_anchor.embedding:
                ids.append(sentence_anchor.sentence_id)
                embeddings.append(sentence_anchor.embedding)
                documents.append(sentence_anchor.sentence)
                metadatas.append({
                    'type': 'sentence',
                    'conversation_id': sentence_anchor.conversation_id,
                    'topic': sentence_anchor.topic,
                    'emotion': sentence_anchor.emotion,
                    'project_type': sentence_anchor.project_type,
                    'word_count': sentence_anchor.word_count
                })

        # Batch insert
        total_items = len(ids)
        self.logger.info(f"Ingesting {total_items} items to ChromaDB...")

        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)
            batch_ids = ids[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_documents = documents[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

            self.logger.info(f"Ingested batch {i//batch_size + 1}/{(total_items-1)//batch_size + 1}")

        self.logger.info("ChromaDB ingestion completed")

    def generate_summary_report(self):
        """Generate comprehensive summary report of the semantic anchor system."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_config': {
                'embedding_backend': self.embedding_backend,
                'cache_embeddings': self.cache_embeddings,
                'output_directory': str(self.output_dir)
            },
            'data_summary': {
                'word_anchors': len(self.word_anchors),
                'sentence_anchors': len(self.sentence_anchors),
                'conversations': len(self.conversation_metadata),
                'cached_embeddings': len(self.embedding_cache)
            },
            'word_anchor_stats': self._analyze_word_anchors(),
            'sentence_anchor_stats': self._analyze_sentence_anchors(),
            'conversation_stats': self._analyze_conversations(),
            'embedding_stats': self._analyze_embeddings()
        }

        # Save report
        report_file = self.output_dir / f"semantic_anchor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Summary report saved to {report_file}")
        return report

    def _analyze_word_anchors(self) -> Dict[str, Any]:
        """Analyze word anchor statistics."""
        if not self.word_anchors:
            return {}

        pos_dist = Counter(wa.pos_tag for wa in self.word_anchors.values())
        emotion_dist = Counter(wa.emotion for wa in self.word_anchors.values())
        domain_dist = Counter(wa.domain for wa in self.word_anchors.values())
        freq_stats = [wa.frequency for wa in self.word_anchors.values()]

        return {
            'total_words': len(self.word_anchors),
            'pos_distribution': dict(pos_dist.most_common(10)),
            'emotion_distribution': dict(emotion_dist.most_common(10)),
            'domain_distribution': dict(domain_dist.most_common(10)),
            'frequency_stats': {
                'min': min(freq_stats) if freq_stats else 0,
                'max': max(freq_stats) if freq_stats else 0,
                'avg': sum(freq_stats) / len(freq_stats) if freq_stats else 0
            },
            'embeddings_generated': sum(1 for wa in self.word_anchors.values() if wa.embedding)
        }

    def _analyze_sentence_anchors(self) -> Dict[str, Any]:
        """Analyze sentence anchor statistics."""
        if not self.sentence_anchors:
            return {}

        topic_dist = Counter(sa.topic for sa in self.sentence_anchors.values())
        emotion_dist = Counter(sa.emotion for sa in self.sentence_anchors.values())
        project_dist = Counter(sa.project_type for sa in self.sentence_anchors.values())
        word_counts = [sa.word_count for sa in self.sentence_anchors.values()]

        return {
            'total_sentences': len(self.sentence_anchors),
            'topic_distribution': dict(topic_dist.most_common(10)),
            'emotion_distribution': dict(emotion_dist.most_common(10)),
            'project_distribution': dict(project_dist.most_common(10)),
            'word_count_stats': {
                'min': min(word_counts) if word_counts else 0,
                'max': max(word_counts) if word_counts else 0,
                'avg': sum(word_counts) / len(word_counts) if word_counts else 0
            },
            'embeddings_generated': sum(1 for sa in self.sentence_anchors.values() if sa.embedding)
        }

    def _analyze_conversations(self) -> Dict[str, Any]:
        """Analyze conversation metadata statistics."""
        if not self.conversation_metadata:
            return {}

        model_dist = Counter(cm.model for cm in self.conversation_metadata.values())
        topic_dist = Counter(cm.topic for cm in self.conversation_metadata.values())
        emotion_dist = Counter(cm.emotion for cm in self.conversation_metadata.values())

        return {
            'total_conversations': len(self.conversation_metadata),
            'model_distribution': dict(model_dist.most_common(10)),
            'topic_distribution': dict(topic_dist.most_common(10)),
            'emotion_distribution': dict(emotion_dist.most_common(10))
        }

    def _analyze_embeddings(self) -> Dict[str, Any]:
        """Analyze embedding statistics."""
        embeddings_with_data = []

        for wa in self.word_anchors.values():
            if wa.embedding:
                embeddings_with_data.append(len(wa.embedding))

        for sa in self.sentence_anchors.values():
            if sa.embedding:
                embeddings_with_data.append(len(sa.embedding))

        return {
            'total_embeddings': len(embeddings_with_data),
            'embedding_dimensions': list(set(embeddings_with_data)),
            'cache_size': len(self.embedding_cache)
        }

def main():
    """Main execution function for the Semantic Anchor System."""
    print("🚀 Semantic Anchor System - Phase 1: Anchor the Data")
    print("=" * 60)

    # Initialize system
    system = SemanticAnchorSystem(
        embedding_backend="sentence_transformers",  # Using local model by default
        cache_embeddings=True,
        output_dir="semantic_anchors"
    )

    # Load data
    print("\n📊 Loading enhanced analysis results...")
    system.load_enhanced_word_results()
    system.load_enhanced_topic_results()
    system.load_sentences_data(max_sentences=5000)

    # Generate embeddings (limited for demo)
    print("\n🧠 Generating semantic embeddings...")
    system.generate_embeddings(batch_size=50, max_items=1000)

    # Export data
    print("\n💾 Exporting semantic anchors...")
    jsonl_file = system.export_to_jsonl()
    csv_file = system.export_to_csv()

    # Setup vector database (optional)
    print("\n🗄️ Setting up vector database...")
    collection = system.setup_chromadb()
    if collection:
        system.ingest_to_chromadb(collection, batch_size=50)

    # Generate summary report
    print("\n📋 Generating summary report...")
    report = system.generate_summary_report()

    print("\n✅ Semantic Anchor System Phase 1 Complete!")
    print(f"📁 Word Anchors: {report['data_summary']['word_anchors']}")
    print(f"📝 Sentence Anchors: {report['data_summary']['sentence_anchors']}")
    print(f"💬 Conversations: {report['data_summary']['conversations']}")
    print(f"🧠 Embeddings Generated: {report['embedding_stats']['total_embeddings']}")
    print(f"📤 Exported to: {jsonl_file.name}, {csv_file.name}")

if __name__ == "__main__":
    main()
