#!/usr/bin/env python3
"""
Semantic Intelligence Engine - Phase 2
Advanced search, reasoning, clustering, and cognitive analysis system
"""

import json
import csv
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict, Counter
import re
import hashlib

# Optional imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("NetworkX not available. Install with: pip install networkx")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Plotting libraries not available. Install with: pip install matplotlib seaborn")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import our Phase 1 system
from semantic_anchor_system import SemanticAnchorSystem, WordAnchor, SentenceAnchor, ConversationMetadata

@dataclass
class SearchResult:
    """Represents a semantic search result with relevance scoring."""
    item_id: str
    item_type: str  # 'word', 'sentence', 'conversation'
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    reasoning: Optional[str] = None

@dataclass
class ThoughtCluster:
    """Represents a cluster of related thoughts/conversations."""
    cluster_id: str
    cluster_name: str
    center_embedding: List[float]
    member_ids: List[str]
    topics: List[str]
    emotions: List[str]
    coherence_score: float
    size: int

@dataclass
class CognitiveMirror:
    """Represents a cognitive pattern or repeated concept."""
    pattern_id: str
    pattern_type: str  # 'repeated_concept', 'emotional_loop', 'breakthrough', 'contradiction'
    description: str
    instances: List[str]  # Item IDs where pattern appears
    strength: float
    temporal_pattern: Optional[str] = None
    evolution: Optional[str] = None

class SemanticIntelligenceEngine:
    """Advanced semantic intelligence system for conversation analysis."""

    def __init__(self,
                 anchor_system: SemanticAnchorSystem = None,
                 openai_api_key: Optional[str] = None,
                 output_dir: str = "semantic_intelligence"):
        """Initialize the Semantic Intelligence Engine."""

        self.anchor_system = anchor_system or SemanticAnchorSystem()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize OpenAI if available
        if OPENAI_AVAILABLE and openai_api_key:
            openai.api_key = openai_api_key
            self.openai_enabled = True
        else:
            self.openai_enabled = False

        # Storage for intelligence components
        self.search_index = {}
        self.thought_clusters = {}
        self.cognitive_mirrors = {}
        self.thought_graph = None

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.logger.info("Semantic Intelligence Engine initialized")

    # ===== COMPONENT 1: SEARCH & REASONING RAG LOOP =====

    def build_search_index(self):
        """Build comprehensive search index from semantic anchors."""
        self.logger.info("Building semantic search index...")

        # Index word anchors
        for word_id, word_anchor in self.anchor_system.word_anchors.items():
            if word_anchor.embedding:
                self.search_index[word_id] = {
                    'type': 'word',
                    'content': word_anchor.word,
                    'embedding': np.array(word_anchor.embedding),
                    'metadata': {
                        'frequency': word_anchor.frequency,
                        'pos_tag': word_anchor.pos_tag,
                        'emotion': word_anchor.emotion,
                        'domain': word_anchor.domain,
                        'conversations': word_anchor.conversations
                    }
                }

        # Index sentence anchors
        for sent_id, sent_anchor in self.anchor_system.sentence_anchors.items():
            if sent_anchor.embedding:
                self.search_index[sent_id] = {
                    'type': 'sentence',
                    'content': sent_anchor.sentence,
                    'embedding': np.array(sent_anchor.embedding),
                    'metadata': {
                        'conversation_id': sent_anchor.conversation_id,
                        'topic': sent_anchor.topic,
                        'topic_confidence': sent_anchor.topic_confidence,
                        'emotion': sent_anchor.emotion,
                        'emotion_confidence': sent_anchor.emotion_confidence,
                        'project_type': sent_anchor.project_type,
                        'word_count': sent_anchor.word_count
                    }
                }

        self.logger.info(f"Search index built with {len(self.search_index)} items")

    def semantic_search(self,
                       query: str,
                       search_type: str = "all",
                       top_k: int = 10,
                       filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform semantic search across conversation data."""

        if not self.search_index:
            self.build_search_index()

        # Get query embedding
        query_embedding = self.anchor_system._get_embedding(query)
        if not query_embedding:
            self.logger.error("Failed to generate query embedding")
            return []

        query_embedding = np.array(query_embedding)

        # Calculate similarities
        similarities = []
        for item_id, item_data in self.search_index.items():
            # Apply type filter
            if search_type != "all" and item_data['type'] != search_type:
                continue

            # Apply metadata filters
            if filters and not self._matches_filters(item_data['metadata'], filters):
                continue

            # Calculate cosine similarity
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity([query_embedding], [item_data['embedding']])[0][0]
            else:
                # Fallback cosine similarity calculation
                similarity = self._cosine_similarity(query_embedding, item_data['embedding'])

            similarities.append({
                'item_id': item_id,
                'similarity': similarity,
                'item_data': item_data
            })

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        results = []
        for sim_data in similarities[:top_k]:
            result = SearchResult(
                item_id=sim_data['item_id'],
                item_type=sim_data['item_data']['type'],
                content=sim_data['item_data']['content'],
                similarity_score=sim_data['similarity'],
                metadata=sim_data['item_data']['metadata']
            )
            results.append(result)

        return results

    def _cosine_similarity(self, a, b):
        """Fallback cosine similarity calculation."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True

    def rag_reasoning(self,
                     query: str,
                     search_results: List[SearchResult],
                     reasoning_prompt: str = None) -> str:
        """Use RAG loop to generate reasoning from search results."""

        if not self.openai_enabled:
            return "RAG reasoning requires OpenAI API key"

        # Prepare context from search results
        context_parts = []
        for result in search_results:
            context_parts.append(f"[{result.item_type.upper()}] {result.content}")
            if result.metadata.get('topic'):
                context_parts.append(f"Topic: {result.metadata['topic']}")
            if result.metadata.get('emotion'):
                context_parts.append(f"Emotion: {result.metadata['emotion']}")
            context_parts.append(f"Relevance: {result.similarity_score:.3f}")
            context_parts.append("---")

        context = "\n".join(context_parts)

        # Default reasoning prompt
        if not reasoning_prompt:
            reasoning_prompt = """
            Based on the following conversation data, provide insights and reasoning for the query.

            Query: {query}

            Relevant Context:
            {context}

            Please provide:
            1. Key insights from the data
            2. Patterns or themes you notice
            3. Reasoning about the query based on the evidence
            4. Any contradictions or interesting connections
            """

        prompt = reasoning_prompt.format(query=query, context=context)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"RAG reasoning failed: {e}")
            return f"RAG reasoning failed: {e}"

    def intelligent_search(self,
                          query: str,
                          search_type: str = "all",
                          top_k: int = 10,
                          include_reasoning: bool = True) -> Dict[str, Any]:
        """Perform intelligent search with optional RAG reasoning."""

        # Perform semantic search
        results = self.semantic_search(query, search_type, top_k)

        # Generate reasoning if requested
        reasoning = None
        if include_reasoning and results:
            reasoning = self.rag_reasoning(query, results)

        return {
            'query': query,
            'results': results,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }

    # ===== COMPONENT 2: CLUSTER & VISUALIZE THOUGHT GRAPH =====

    def generate_thought_clusters(self,
                                 cluster_method: str = "kmeans",
                                 n_clusters: int = 10,
                                 min_cluster_size: int = 3) -> Dict[str, ThoughtCluster]:
        """Generate dynamic topic/emotion clusters from conversation data."""

        if not SKLEARN_AVAILABLE:
            self.logger.error("Clustering requires scikit-learn")
            return self._fallback_clustering(n_clusters)

        if not self.search_index:
            self.build_search_index()

        # Prepare data for clustering
        embeddings = []
        item_ids = []
        metadata_list = []

        for item_id, item_data in self.search_index.items():
            if item_data['type'] == 'sentence':  # Focus on sentence-level clustering
                embeddings.append(item_data['embedding'])
                item_ids.append(item_id)
                metadata_list.append(item_data['metadata'])

        if len(embeddings) < n_clusters:
            self.logger.warning(f"Not enough data for {n_clusters} clusters")
            n_clusters = max(2, len(embeddings) // 2)

        embeddings = np.array(embeddings)

        # Perform clustering
        if cluster_method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif cluster_method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=min_cluster_size)
        else:
            self.logger.error(f"Unknown clustering method: {cluster_method}")
            return {}

        cluster_labels = clusterer.fit_predict(embeddings)

        # Build thought clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # DBSCAN noise
                continue

            cluster_id = f"cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'member_ids': [],
                    'embeddings': [],
                    'topics': [],
                    'emotions': [],
                    'metadata': []
                }

            clusters[cluster_id]['member_ids'].append(item_ids[i])
            clusters[cluster_id]['embeddings'].append(embeddings[i])
            clusters[cluster_id]['metadata'].append(metadata_list[i])

            # Collect topics and emotions
            if metadata_list[i].get('topic'):
                clusters[cluster_id]['topics'].append(metadata_list[i]['topic'])
            if metadata_list[i].get('emotion'):
                clusters[cluster_id]['emotions'].append(metadata_list[i]['emotion'])

        # Create ThoughtCluster objects
        thought_clusters = {}
        for cluster_id, cluster_data in clusters.items():
            if len(cluster_data['member_ids']) < min_cluster_size:
                continue

            # Calculate cluster center
            center_embedding = np.mean(cluster_data['embeddings'], axis=0).tolist()

            # Get most common topics and emotions
            topic_counts = Counter(cluster_data['topics'])
            emotion_counts = Counter(cluster_data['emotions'])

            # Calculate coherence (average intra-cluster similarity)
            cluster_embeddings = np.array(cluster_data['embeddings'])
            similarities = cosine_similarity(cluster_embeddings)
            coherence_score = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

            # Generate cluster name
            top_topic = topic_counts.most_common(1)[0][0] if topic_counts else "Mixed"
            top_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Neutral"
            cluster_name = f"{top_topic}_{top_emotion}"

            thought_cluster = ThoughtCluster(
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                center_embedding=center_embedding,
                member_ids=cluster_data['member_ids'],
                topics=list(topic_counts.keys()),
                emotions=list(emotion_counts.keys()),
                coherence_score=coherence_score,
                size=len(cluster_data['member_ids'])
            )

            thought_clusters[cluster_id] = thought_cluster

        self.thought_clusters = thought_clusters
        self.logger.info(f"Generated {len(thought_clusters)} thought clusters")

        return thought_clusters

    def _fallback_clustering(self, n_clusters: int) -> Dict[str, ThoughtCluster]:
        """Fallback clustering when scikit-learn is not available."""
        self.logger.info("Using fallback clustering based on metadata")

        # Simple clustering based on topic and emotion combinations
        clusters = defaultdict(list)

        for item_id, item_data in self.search_index.items():
            if item_data['type'] == 'sentence':
                topic = item_data['metadata'].get('topic', 'unknown')
                emotion = item_data['metadata'].get('emotion', 'neutral')
                cluster_key = f"{topic}_{emotion}"
                clusters[cluster_key].append(item_id)

        # Convert to ThoughtCluster objects
        thought_clusters = {}
        for i, (cluster_key, member_ids) in enumerate(clusters.items()):
            if len(member_ids) >= 3:  # Minimum cluster size
                cluster_id = f"cluster_{i}"
                parts = cluster_key.split('_')
                topic = parts[0] if len(parts) > 0 else "unknown"
                emotion = parts[1] if len(parts) > 1 else "neutral"

                thought_cluster = ThoughtCluster(
                    cluster_id=cluster_id,
                    cluster_name=cluster_key,
                    center_embedding=[0.0] * 384,  # Dummy embedding
                    member_ids=member_ids,
                    topics=[topic],
                    emotions=[emotion],
                    coherence_score=0.8,  # Dummy score
                    size=len(member_ids)
                )

                thought_clusters[cluster_id] = thought_cluster

        self.thought_clusters = thought_clusters
        return thought_clusters

    def build_thought_graph(self) -> Optional[nx.Graph]:
        """Build a NetworkX graph of conversation relationships."""

        if not NETWORKX_AVAILABLE:
            self.logger.error("Graph building requires NetworkX")
            return None

        if not self.search_index:
            self.build_search_index()

        # Create graph
        G = nx.Graph()

        # Add nodes (conversations and topics)
        conversation_nodes = set()
        for item_id, item_data in self.search_index.items():
            if item_data['type'] == 'sentence':
                conv_id = item_data['metadata']['conversation_id']
                topic = item_data['metadata']['topic']
                emotion = item_data['metadata']['emotion']

                # Add conversation node
                if conv_id not in conversation_nodes:
                    G.add_node(conv_id,
                              type='conversation',
                              topic=topic,
                              emotion=emotion)
                    conversation_nodes.add(conv_id)

                # Add topic node
                topic_node = f"topic_{topic}"
                if not G.has_node(topic_node):
                    G.add_node(topic_node, type='topic', name=topic)

                # Add emotion node
                emotion_node = f"emotion_{emotion}"
                if not G.has_node(emotion_node):
                    G.add_node(emotion_node, type='emotion', name=emotion)

                # Add edges
                G.add_edge(conv_id, topic_node, weight=1.0)
                G.add_edge(conv_id, emotion_node, weight=0.8)

        # Add similarity edges between conversations
        conv_embeddings = {}
        for item_id, item_data in self.search_index.items():
            if item_data['type'] == 'sentence':
                conv_id = item_data['metadata']['conversation_id']
                if conv_id not in conv_embeddings:
                    conv_embeddings[conv_id] = []
                conv_embeddings[conv_id].append(item_data['embedding'])

        # Calculate conversation similarity
        conv_ids = list(conv_embeddings.keys())
        for i, conv_id1 in enumerate(conv_ids):
            for conv_id2 in conv_ids[i+1:]:
                # Average embeddings for each conversation
                emb1 = np.mean(conv_embeddings[conv_id1], axis=0)
                emb2 = np.mean(conv_embeddings[conv_id2], axis=0)

                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity([emb1], [emb2])[0][0]
                else:
                    similarity = self._cosine_similarity(emb1, emb2)

                # Add edge if similarity is above threshold
                if similarity > 0.7:
                    G.add_edge(conv_id1, conv_id2,
                              weight=similarity,
                              type='similarity')

        self.thought_graph = G
        self.logger.info(f"Built thought graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        return G

    def visualize_thought_clusters(self, output_file: str = None):
        """Create visualization of thought clusters."""

        if not PLOTTING_AVAILABLE:
            self.logger.error("Visualization requires matplotlib and seaborn")
            return

        if not self.thought_clusters:
            self.generate_thought_clusters()

        # Prepare data for visualization
        cluster_data = []
        for cluster_id, cluster in self.thought_clusters.items():
            cluster_data.append({
                'cluster_id': cluster_id,
                'name': cluster.cluster_name,
                'size': cluster.size,
                'coherence': cluster.coherence_score,
                'top_topic': cluster.topics[0] if cluster.topics else 'Unknown',
                'top_emotion': cluster.emotions[0] if cluster.emotions else 'Unknown'
            })

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Cluster sizes
        sizes = [c['size'] for c in cluster_data]
        names = [c['name'][:20] for c in cluster_data]
        ax1.bar(range(len(sizes)), sizes)
        ax1.set_title('Cluster Sizes')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')

        # Coherence scores
        coherences = [c['coherence'] for c in cluster_data]
        ax2.bar(range(len(coherences)), coherences)
        ax2.set_title('Cluster Coherence Scores')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')

        # Topic distribution
        topics = [c['top_topic'] for c in cluster_data]
        topic_counts = Counter(topics)
        ax3.pie(topic_counts.values(), labels=topic_counts.keys(), autopct='%1.1f%%')
        ax3.set_title('Topic Distribution')

        # Emotion distribution
        emotions = [c['top_emotion'] for c in cluster_data]
        emotion_counts = Counter(emotions)
        ax4.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Emotion Distribution')

        plt.tight_layout()

        if output_file:
            plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'thought_clusters_visualization.png', dpi=300, bbox_inches='tight')

        plt.close()
        self.logger.info("Thought cluster visualization saved")

    # ===== COMPONENT 3: COGNITIVE MIRRORS =====

    def detect_repeated_concepts(self, similarity_threshold: float = 0.85) -> List[CognitiveMirror]:
        """Detect repeated concepts across conversations."""

        if not self.search_index:
            self.build_search_index()

        repeated_concepts = []

        # Find similar sentences
        sentence_items = [(k, v) for k, v in self.search_index.items() if v['type'] == 'sentence']

        for i, (id1, data1) in enumerate(sentence_items):
            similar_sentences = []

            for id2, data2 in sentence_items[i+1:]:
                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity([data1['embedding']], [data2['embedding']])[0][0]
                else:
                    similarity = self._cosine_similarity(data1['embedding'], data2['embedding'])

                if similarity > similarity_threshold:
                    similar_sentences.append({
                        'id': id2,
                        'content': data2['content'],
                        'similarity': similarity,
                        'metadata': data2['metadata']
                    })

            if len(similar_sentences) >= 2:  # At least 2 similar instances
                concept_id = f"concept_{hashlib.md5(data1['content'].encode()).hexdigest()[:8]}"

                instances = [id1] + [s['id'] for s in similar_sentences]
                avg_similarity = np.mean([s['similarity'] for s in similar_sentences])

                repeated_concept = CognitiveMirror(
                    pattern_id=concept_id,
                    pattern_type="repeated_concept",
                    description=f"Repeated concept: {data1['content'][:100]}...",
                    instances=instances,
                    strength=avg_similarity
                )

                repeated_concepts.append(repeated_concept)

        return repeated_concepts

    def detect_emotional_loops(self) -> List[CognitiveMirror]:
        """Detect emotional patterns and loops in conversations."""

        emotional_loops = []

        # Group sentences by conversation and analyze emotional progression
        conv_emotions = defaultdict(list)

        for item_id, item_data in self.search_index.items():
            if item_data['type'] == 'sentence':
                conv_id = item_data['metadata']['conversation_id']
                emotion = item_data['metadata']['emotion']
                conv_emotions[conv_id].append(emotion)

        # Detect loops (repeated emotional patterns)
        for conv_id, emotions in conv_emotions.items():
            if len(emotions) < 4:  # Need at least 4 emotions to detect patterns
                continue

            # Look for repeated subsequences
            for pattern_length in range(2, len(emotions) // 2 + 1):
                for start in range(len(emotions) - pattern_length * 2 + 1):
                    pattern = emotions[start:start + pattern_length]

                    # Check if pattern repeats
                    repeats = 1
                    next_start = start + pattern_length

                    while (next_start + pattern_length <= len(emotions) and
                           emotions[next_start:next_start + pattern_length] == pattern):
                        repeats += 1
                        next_start += pattern_length

                    if repeats >= 2:  # Pattern repeats at least once
                        loop_id = f"loop_{conv_id}_{start}_{pattern_length}"

                        emotional_loop = CognitiveMirror(
                            pattern_id=loop_id,
                            pattern_type="emotional_loop",
                            description=f"Emotional loop: {' → '.join(pattern)} (repeats {repeats} times)",
                            instances=[conv_id],
                            strength=repeats / len(emotions),
                            temporal_pattern=f"Pattern length: {pattern_length}, Repeats: {repeats}"
                        )

                        emotional_loops.append(emotional_loop)

        return emotional_loops

    def detect_breakthroughs(self) -> List[CognitiveMirror]:
        """Detect breakthrough moments or significant shifts in thinking."""

        breakthroughs = []

        # Look for sudden changes in topic or emotion within conversations
        for conv_id, conv_meta in self.anchor_system.conversation_metadata.items():
            conv_sentences = []

            for item_id, item_data in self.search_index.items():
                if (item_data['type'] == 'sentence' and
                    item_data['metadata']['conversation_id'] == conv_id):
                    conv_sentences.append({
                        'id': item_id,
                        'content': item_data['content'],
                        'topic': item_data['metadata']['topic'],
                        'emotion': item_data['metadata']['emotion'],
                        'embedding': item_data['embedding']
                    })

            if len(conv_sentences) < 3:
                continue

            # Detect sudden semantic shifts
            for i in range(1, len(conv_sentences)):
                prev_emb = conv_sentences[i-1]['embedding']
                curr_emb = conv_sentences[i]['embedding']

                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity([prev_emb], [curr_emb])[0][0]
                else:
                    similarity = self._cosine_similarity(prev_emb, curr_emb)

                # If similarity drops significantly, might be a breakthrough
                if similarity < 0.3:  # Low similarity indicates shift
                    breakthrough_id = f"breakthrough_{conv_id}_{i}"

                    breakthrough = CognitiveMirror(
                        pattern_id=breakthrough_id,
                        pattern_type="breakthrough",
                        description=f"Potential breakthrough: shift from '{conv_sentences[i-1]['content'][:50]}...' to '{conv_sentences[i]['content'][:50]}...'",
                        instances=[conv_sentences[i-1]['id'], conv_sentences[i]['id']],
                        strength=1.0 - similarity,
                        evolution=f"Topic shift: {conv_sentences[i-1]['topic']} → {conv_sentences[i]['topic']}"
                    )

                    breakthroughs.append(breakthrough)

        return breakthroughs

    def detect_contradictions(self) -> List[CognitiveMirror]:
        """Detect contradictions or opposing viewpoints in conversations."""

        contradictions = []

        # Look for sentences with opposing sentiments or contradictory content
        sentence_items = [(k, v) for k, v in self.search_index.items() if v['type'] == 'sentence']

        # Define opposing emotion pairs
        opposing_emotions = [
            ('joy', 'sadness'),
            ('trust', 'disgust'),
            ('fear', 'anger'),
            ('surprise', 'anticipation'),
            ('confidence', 'anxiety'),
            ('analytical', 'emotional')
        ]

        for i, (id1, data1) in enumerate(sentence_items):
            for id2, data2 in sentence_items[i+1:]:
                # Check for opposing emotions
                emotion1 = data1['metadata']['emotion']
                emotion2 = data2['metadata']['emotion']

                is_opposing = False
                for pos, neg in opposing_emotions:
                    if (emotion1 == pos and emotion2 == neg) or (emotion1 == neg and emotion2 == pos):
                        is_opposing = True
                        break

                if is_opposing:
                    # Check if content is somewhat similar (same topic, different stance)
                    if SKLEARN_AVAILABLE:
                        similarity = cosine_similarity([data1['embedding']], [data2['embedding']])[0][0]
                    else:
                        similarity = self._cosine_similarity(data1['embedding'], data2['embedding'])

                    if 0.4 < similarity < 0.7:  # Moderate similarity suggests same topic, different stance
                        contradiction_id = f"contradiction_{hashlib.md5((id1 + id2).encode()).hexdigest()[:8]}"

                        contradiction = CognitiveMirror(
                            pattern_id=contradiction_id,
                            pattern_type="contradiction",
                            description=f"Contradiction: {emotion1} vs {emotion2} on similar topic",
                            instances=[id1, id2],
                            strength=similarity,
                            evolution=f"Emotional opposition: {emotion1} ↔ {emotion2}"
                        )

                        contradictions.append(contradiction)

        return contradictions

    def generate_cognitive_mirrors(self) -> Dict[str, List[CognitiveMirror]]:
        """Generate comprehensive cognitive mirror analysis."""

        self.logger.info("Generating cognitive mirrors...")

        mirrors = {
            'repeated_concepts': self.detect_repeated_concepts(),
            'emotional_loops': self.detect_emotional_loops(),
            'breakthroughs': self.detect_breakthroughs(),
            'contradictions': self.detect_contradictions()
        }

        # Store results
        all_mirrors = {}
        for mirror_type, mirror_list in mirrors.items():
            for mirror in mirror_list:
                all_mirrors[mirror.pattern_id] = mirror

        self.cognitive_mirrors = all_mirrors

        self.logger.info(f"Generated {len(all_mirrors)} cognitive mirrors:")
        for mirror_type, mirror_list in mirrors.items():
            self.logger.info(f"  - {mirror_type}: {len(mirror_list)}")

        return mirrors

    # ===== EXPORT AND REPORTING =====

    def export_intelligence_report(self, filename: str = None) -> Path:
        """Export comprehensive intelligence analysis report."""

        if not filename:
            filename = f"intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'search_index_size': len(self.search_index),
                'thought_clusters': len(self.thought_clusters),
                'cognitive_mirrors': len(self.cognitive_mirrors),
                'capabilities': {
                    'semantic_search': True,
                    'rag_reasoning': self.openai_enabled,
                    'clustering': SKLEARN_AVAILABLE,
                    'graph_analysis': NETWORKX_AVAILABLE,
                    'visualization': PLOTTING_AVAILABLE
                }
            },
            'search_index': {
                'total_items': len(self.search_index),
                'word_anchors': len([1 for v in self.search_index.values() if v['type'] == 'word']),
                'sentence_anchors': len([1 for v in self.search_index.values() if v['type'] == 'sentence'])
            },
            'thought_clusters': {
                cluster_id: {
                    'name': cluster.cluster_name,
                    'size': cluster.size,
                    'coherence_score': cluster.coherence_score,
                    'topics': cluster.topics,
                    'emotions': cluster.emotions
                }
                for cluster_id, cluster in self.thought_clusters.items()
            },
            'cognitive_mirrors': {
                mirror_id: {
                    'pattern_type': mirror.pattern_type,
                    'description': mirror.description,
                    'strength': mirror.strength,
                    'instance_count': len(mirror.instances),
                    'temporal_pattern': mirror.temporal_pattern,
                    'evolution': mirror.evolution
                }
                for mirror_id, mirror in self.cognitive_mirrors.items()
            }
        }

        report_file = self.output_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Intelligence report exported to {report_file}")
        return report_file

def main():
    """Main execution function for the Semantic Intelligence Engine."""
    print("🧠 SEMANTIC INTELLIGENCE ENGINE - PHASE 2")
    print("=" * 60)
    print("🔍 Advanced Search & Reasoning RAG Loop")
    print("🕸️ Cluster & Visualize Thought Graph")
    print("🪞 Develop Cognitive Mirrors")
    print()

    # Initialize Phase 1 system
    print("⚙️ Initializing Phase 1 Semantic Anchor System...")
    anchor_system = SemanticAnchorSystem(
        embedding_backend="sentence_transformers",
        cache_embeddings=True,
        output_dir="semantic_anchors"
    )

    # Load Phase 1 data
    print("📊 Loading Phase 1 semantic anchors...")
    anchor_system.load_enhanced_word_results()
    anchor_system.load_enhanced_topic_results()
    anchor_system.load_sentences_data(max_sentences=1000)

    # Generate embeddings if needed
    if not anchor_system.word_anchors or not anchor_system.sentence_anchors:
        print("🧠 Generating embeddings for Phase 1 data...")
        anchor_system.generate_embeddings(batch_size=25, max_items=500)

    # Initialize Phase 2 Intelligence Engine
    print("\n🧠 Initializing Semantic Intelligence Engine...")
    intelligence_engine = SemanticIntelligenceEngine(
        anchor_system=anchor_system,
        output_dir="semantic_intelligence"
    )

    # Build search capabilities
    print("\n🔍 Building semantic search index...")
    intelligence_engine.build_search_index()

    # Generate thought clusters
    print("\n🕸️ Generating thought clusters...")
    clusters = intelligence_engine.generate_thought_clusters(n_clusters=8)

    # Build thought graph
    print("\n🌐 Building thought graph...")
    graph = intelligence_engine.build_thought_graph()

    # Generate cognitive mirrors
    print("\n🪞 Generating cognitive mirrors...")
    mirrors = intelligence_engine.generate_cognitive_mirrors()

    # Create visualizations
    print("\n📊 Creating visualizations...")
    intelligence_engine.visualize_thought_clusters()

    # Export comprehensive report
    print("\n📋 Generating intelligence report...")
    report_file = intelligence_engine.export_intelligence_report()

    # Demonstrate search capabilities
    print("\n🔍 Demonstrating semantic search...")
    search_examples = [
        "programming and coding projects",
        "emotional challenges and breakthroughs",
        "creative and artistic endeavors"
    ]

    for query in search_examples:
        print(f"\n🔍 Searching: '{query}'")
        results = intelligence_engine.intelligent_search(
            query=query,
            top_k=3,
            include_reasoning=False  # Skip reasoning for demo
        )

        for i, result in enumerate(results['results'], 1):
            print(f"   {i}. [{result.item_type}] {result.content[:100]}... (similarity: {result.similarity_score:.3f})")

    # Display final results
    print("\n" + "=" * 60)
    print("✅ SEMANTIC INTELLIGENCE ENGINE PHASE 2 COMPLETE!")
    print("=" * 60)

    print(f"🔍 Search Index: {len(intelligence_engine.search_index)} items")
    print(f"🕸️ Thought Clusters: {len(clusters)}")
    print(f"🌐 Graph Nodes: {graph.number_of_nodes() if graph else 0}")
    print(f"🌐 Graph Edges: {graph.number_of_edges() if graph else 0}")

    print(f"\n🪞 Cognitive Mirrors:")
    for mirror_type, mirror_list in mirrors.items():
        print(f"   - {mirror_type}: {len(mirror_list)}")

    print(f"\n📁 Results saved to: semantic_intelligence/")
    print(f"📋 Full report: {report_file.name}")

if __name__ == "__main__":
    main()
