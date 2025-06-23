#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder
Creates deeply interwoven relationships between all entities
"""

import json
import pandas as pd
import networkx as nx
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any
import uuid
import hashlib
import numpy as np
from collections import defaultdict

class EnhancedKnowledgeGraphBuilder:
    def __init__(self, id_results_dir: str, output_dir: str):
        self.id_results_dir = Path(id_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.graph = nx.MultiDiGraph()

    def load_and_enhance_data(self) -> Dict[str, pd.DataFrame]:
        """Load ID data and enhance with additional relationships."""
        print("📂 Loading and enhancing ID data...")

        id_data = {}
        expected_files = [
            'conversation_ids.csv', 'sentence_ids.csv', 'concept_ids.csv',
            'emotion_ids.csv', 'metaphor_ids.csv', 'project_ids.csv',
            'thread_ids.csv', 'pair_ids.csv'
        ]

        for filename in expected_files:
            filepath = self.id_results_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                id_type = filename.replace('_ids.csv', '')
                id_data[id_type] = df
                print(f"  ✅ Loaded {len(df)} {id_type} records")

        # Enhance data with cross-references
        self._enhance_conversation_data(id_data)
        self._enhance_sentence_data(id_data)
        self._enhance_concept_data(id_data)

        return id_data

    def _enhance_conversation_data(self, id_data: Dict[str, pd.DataFrame]):
        """Add temporal and thematic relationships between conversations."""
        if 'conversation' not in id_data:
            return

        conv_df = id_data['conversation']

        # Add temporal clustering
        conv_df['date_group'] = conv_df['date'].apply(lambda x: x[:6] if pd.notna(x) else 'unknown')

        # Add thematic clustering based on titles
        conv_df['theme_keywords'] = conv_df['title'].apply(self._extract_theme_keywords)

        # Add conversation complexity score
        conv_df['complexity_score'] = conv_df['message_count'].apply(
            lambda x: min(x / 100, 1.0) if pd.notna(x) else 0.0
        )

    def _enhance_sentence_data(self, id_data: Dict[str, pd.DataFrame]):
        """Add semantic and structural relationships between sentences."""
        if 'sentence' not in id_data:
            return

        sent_df = id_data['sentence']

        # Add sentence complexity metrics
        sent_df['word_count'] = sent_df['text'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        sent_df['char_count'] = sent_df['text'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        sent_df['complexity_score'] = sent_df['word_count'].apply(
            lambda x: min(x / 50, 1.0) if pd.notna(x) else 0.0
        )

        # Add sentence position within conversation
        if 'conversation_id' in sent_df.columns:
            sent_df['position_in_conv'] = sent_df.groupby('conversation_id').cumcount()

    def _enhance_concept_data(self, id_data: Dict[str, pd.DataFrame]):
        """Add concept relationships and clustering."""
        if 'concept' not in id_data:
            return

        concept_df = id_data['concept']

        # Add concept categories
        concept_df['category'] = concept_df['concept_text'].apply(self._categorize_concept)

        # Add concept frequency ranking
        concept_df['frequency_rank'] = concept_df['frequency'].rank(ascending=False)

        # Add concept centrality score
        concept_df['centrality_score'] = concept_df['frequency'].apply(
            lambda x: min(x / 1000, 1.0) if pd.notna(x) else 0.0
        )

    def _extract_theme_keywords(self, title: str) -> List[str]:
        """Extract theme keywords from conversation titles."""
        if pd.isna(title):
            return []

        keywords = []
        title_lower = str(title).lower()

        # Common themes
        themes = {
            'ai_ml': ['ai', 'machine learning', 'neural', 'model', 'algorithm'],
            'programming': ['code', 'script', 'function', 'api', 'database'],
            'analysis': ['analyze', 'data', 'statistics', 'report', 'insights'],
            'creative': ['design', 'creative', 'art', 'story', 'narrative'],
            'business': ['business', 'strategy', 'market', 'product', 'service']
        }

        for theme, words in themes.items():
            if any(word in title_lower for word in words):
                keywords.append(theme)

        return keywords

    def _categorize_concept(self, concept_text: str) -> str:
        """Categorize concepts into semantic categories."""
        if pd.isna(concept_text):
            return 'unknown'

        text_lower = str(concept_text).lower()

        categories = {
            'technical': ['system', 'algorithm', 'function', 'api', 'database'],
            'cognitive': ['thinking', 'reasoning', 'analysis', 'understanding'],
            'creative': ['design', 'creative', 'art', 'story', 'narrative'],
            'emotional': ['feeling', 'emotion', 'mood', 'sentiment'],
            'temporal': ['time', 'sequence', 'evolution', 'progress'],
            'spatial': ['space', 'location', 'position', 'structure']
        }

        for category, words in categories.items():
            if any(word in text_lower for word in words):
                return category

        return 'general'

    def build_enhanced_graph(self, id_data: Dict[str, pd.DataFrame]):
        """Build a deeply interwoven knowledge graph."""
        print("🕸️ Building enhanced knowledge graph with deep interweaving...")

        # Add all nodes
        self._add_all_nodes(id_data)

        # Add basic structural relationships
        self._add_structural_relationships(id_data)

        # Add semantic relationships
        self._add_semantic_relationships(id_data)

        # Add temporal relationships
        self._add_temporal_relationships(id_data)

        # Add thematic relationships
        self._add_thematic_relationships(id_data)

        # Add cross-referential relationships
        self._add_cross_references(id_data)

        # Add derived relationships
        self._add_derived_relationships(id_data)

        print(f"✅ Enhanced graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _add_all_nodes(self, id_data: Dict[str, pd.DataFrame]):
        """Add all nodes to the graph."""
        for node_type, df in id_data.items():
            for _, row in df.iterrows():
                node_id = row.get(f"{node_type}_id", str(uuid.uuid4()))
                node_attrs = {k: v for k, v in row.items() if pd.notna(v)}
                self.graph.add_node(node_id, node_type=node_type, **node_attrs)

    def _add_structural_relationships(self, id_data: Dict[str, pd.DataFrame]):
        """Add basic structural relationships."""
        print("  🔗 Adding structural relationships...")

        # Conversation -> Sentence
        if 'conversation' in id_data and 'sentence' in id_data:
            sent_df = id_data['sentence']
            if 'conversation_id' in sent_df.columns:
                for _, sent_row in sent_df.iterrows():
                    conv_id = sent_row['conversation_id']
                    sent_id = sent_row['sentence_id']
                    if conv_id in self.graph and sent_id in self.graph:
                        self.graph.add_edge(conv_id, sent_id,
                                          edge_type='contains',
                                          weight=1.0,
                                          position=sent_row.get('position_in_conv', 0))

        # Conversation -> Pair
        if 'conversation' in id_data and 'pair' in id_data:
            pair_df = id_data['pair']
            if 'conversation_id' in pair_df.columns:
                for _, pair_row in pair_df.iterrows():
                    conv_id = pair_row['conversation_id']
                    pair_id = pair_row['pair_id']
                    if conv_id in self.graph and pair_id in self.graph:
                        self.graph.add_edge(conv_id, pair_id,
                                          edge_type='contains_pair',
                                          position=pair_row.get('position', 0))

    def _add_semantic_relationships(self, id_data: Dict[str, pd.DataFrame]):
        """Add semantic relationships between concepts and content."""
        print("  🧠 Adding semantic relationships...")

        if 'sentence' in id_data and 'concept' in id_data:
            sent_df = id_data['sentence']
            concept_df = id_data['concept']

            # Create concept-sentence mappings based on text similarity
            for _, sent_row in sent_df.head(5000).iterrows():  # Sample for performance
                sent_text = str(sent_row.get('text', '')).lower()
                sent_id = sent_row['sentence_id']

                for _, concept_row in concept_df.head(1000).iterrows():  # Sample
                    concept_text = str(concept_row.get('concept_text', '')).lower()
                    concept_id = concept_row['concept_id']

                    # Simple text matching for now
                    if concept_text in sent_text and len(concept_text) > 3:
                        relevance = min(len(concept_text) / len(sent_text), 1.0)
                        self.graph.add_edge(sent_id, concept_id,
                                          edge_type='mentions',
                                          relevance=relevance,
                                          frequency=1)

        # Add emotion-sentence relationships
        if 'sentence' in id_data and 'emotion' in id_data:
            emotion_df = id_data['emotion']
            for _, emotion_row in emotion_df.iterrows():
                emotion_id = emotion_row['emotion_id']
                category = emotion_row.get('category', 'neutral')

                # Connect to sentences with similar emotional content
                for _, sent_row in id_data['sentence'].head(1000).iterrows():
                    sent_id = sent_row['sentence_id']
                    sent_text = str(sent_row.get('text', '')).lower()

                    # Simple emotion keyword matching
                    emotion_keywords = {
                        'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful'],
                        'negative': ['bad', 'terrible', 'awful', 'horrible', 'disappointing'],
                        'neutral': ['okay', 'fine', 'normal', 'standard', 'typical']
                    }

                    if category in emotion_keywords:
                        if any(word in sent_text for word in emotion_keywords[category]):
                            self.graph.add_edge(sent_id, emotion_id,
                                              edge_type='expresses',
                                              confidence=0.7,
                                              intensity=0.8)

    def _add_temporal_relationships(self, id_data: Dict[str, pd.DataFrame]):
        """Add temporal and evolutionary relationships."""
        print("  ⏰ Adding temporal relationships...")

        if 'conversation' in id_data:
            conv_df = id_data['conversation']

            # Group conversations by date
            date_groups = conv_df.groupby('date_group')

            for date_group, group_df in date_groups:
                if len(group_df) > 1:
                    # Create temporal sequence within date groups
                    sorted_convos = group_df.sort_values('create_time')

                    for i in range(len(sorted_convos) - 1):
                        conv1_id = sorted_convos.iloc[i]['conversation_id']
                        conv2_id = sorted_convos.iloc[i + 1]['conversation_id']

                        if conv1_id in self.graph and conv2_id in self.graph:
                            self.graph.add_edge(conv1_id, conv2_id,
                                              edge_type='temporally_follows',
                                              time_gap=sorted_convos.iloc[i + 1]['create_time'] - sorted_convos.iloc[i]['create_time'])

    def _add_thematic_relationships(self, id_data: Dict[str, pd.DataFrame]):
        """Add thematic and topic-based relationships."""
        print("  🎯 Adding thematic relationships...")

        if 'conversation' in id_data:
            conv_df = id_data['conversation']

            # Group by themes
            theme_groups = defaultdict(list)
            for _, row in conv_df.iterrows():
                themes = row.get('theme_keywords', [])
                for theme in themes:
                    theme_groups[theme].append(row['conversation_id'])

            # Create theme-based connections
            for theme, conv_ids in theme_groups.items():
                if len(conv_ids) > 1:
                    for i in range(len(conv_ids)):
                        for j in range(i + 1, len(conv_ids)):
                            conv1_id = conv_ids[i]
                            conv2_id = conv_ids[j]

                            if conv1_id in self.graph and conv2_id in self.graph:
                                self.graph.add_edge(conv1_id, conv2_id,
                                                  edge_type='thematically_related',
                                                  theme=theme,
                                                  similarity=0.8)

    def _add_cross_references(self, id_data: Dict[str, pd.DataFrame]):
        """Add cross-references between different entity types."""
        print("  🔄 Adding cross-references...")

        # Concept-concept relationships
        if 'concept' in id_data:
            concept_df = id_data['concept']

            # Group by category
            category_groups = concept_df.groupby('category')

            for category, group_df in category_groups:
                if len(group_df) > 1:
                    concept_ids = group_df['concept_id'].tolist()

                    for i in range(len(concept_ids)):
                        for j in range(i + 1, len(concept_ids)):
                            concept1_id = concept_ids[i]
                            concept2_id = concept_ids[j]

                            if concept1_id in self.graph and concept2_id in self.graph:
                                self.graph.add_edge(concept1_id, concept2_id,
                                                  edge_type='same_category',
                                                  category=category,
                                                  similarity=0.7)

        # Project-concept relationships
        if 'project' in id_data and 'concept' in id_data:
            project_df = id_data['project']
            concept_df = id_data['concept']

            for _, project_row in project_df.iterrows():
                project_name = str(project_row.get('project_name', '')).lower()
                project_id = project_row['project_id']

                for _, concept_row in concept_df.head(500).iterrows():
                    concept_text = str(concept_row.get('concept_text', '')).lower()
                    concept_id = concept_row['concept_id']

                    # Simple keyword matching
                    if any(word in project_name for word in concept_text.split()):
                        self.graph.add_edge(project_id, concept_id,
                                          edge_type='project_uses_concept',
                                          relevance=0.6)

    def _add_derived_relationships(self, id_data: Dict[str, pd.DataFrame]):
        """Add derived and computed relationships."""
        print("  🧮 Adding derived relationships...")

        # Add complexity-based relationships
        if 'sentence' in id_data:
            sent_df = id_data['sentence']

            # Group sentences by complexity
            sent_df['complexity_group'] = pd.cut(sent_df['complexity_score'],
                                               bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])

            complexity_groups = sent_df.groupby('complexity_group')

            for complexity, group_df in complexity_groups:
                if len(group_df) > 1:
                    sent_ids = group_df['sentence_id'].tolist()

                    for i in range(len(sent_ids)):
                        for j in range(i + 1, len(sent_ids)):
                            sent1_id = sent_ids[i]
                            sent2_id = sent_ids[j]

                            if sent1_id in self.graph and sent2_id in self.graph:
                                self.graph.add_edge(sent1_id, sent2_id,
                                                  edge_type='similar_complexity',
                                                  complexity_level=complexity,
                                                  similarity=0.8)

    def export_enhanced_formats(self):
        """Export the enhanced graph in multiple formats."""
        print("📤 Exporting enhanced graph formats...")

        # Export GraphML
        graphml_path = self.output_dir / "enhanced_knowledge_graph.graphml"
        nx.write_graphml(self.graph, str(graphml_path))
        print(f"  ✅ Enhanced GraphML: {graphml_path}")

        # Export JSON
        json_path = self.output_dir / "enhanced_knowledge_graph.json"
        graph_data = nx.node_link_data(self.graph)
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"  ✅ Enhanced JSON: {json_path}")

        # Export GML
        gml_path = self.output_dir / "enhanced_knowledge_graph.gml"
        nx.write_gml(self.graph, str(gml_path))
        print(f"  ✅ Enhanced GML: {gml_path}")

        # Export relationship summary
        self._export_relationship_summary()

    def _export_relationship_summary(self):
        """Export a summary of all relationship types."""
        print("📊 Generating relationship summary...")

        edge_counts = defaultdict(int)
        edge_types = set()

        for edge in self.graph.edges(data=True):
            edge_type = edge[2].get('edge_type', 'unknown')
            edge_counts[edge_type] += 1
            edge_types.add(edge_type)

        summary = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'edge_types': list(edge_types),
            'edge_counts': dict(edge_counts),
            'generated': datetime.now().isoformat()
        }

        summary_path = self.output_dir / "enhanced_graph_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✅ Relationship summary: {summary_path}")
        print(f"  📈 Edge types: {len(edge_types)}")
        for edge_type, count in sorted(edge_counts.items()):
            print(f"    - {edge_type}: {count:,}")

def main():
    parser = argparse.ArgumentParser(description="Build enhanced knowledge graph with deep interweaving")
    parser.add_argument("--id-results-dir", default="integrated_20250623_011924/id_generation_results_new",
                       help="Directory containing ID generation results")
    parser.add_argument("--output-dir", default="integrated_20250623_011924/enhanced_knowledge_graph",
                       help="Output directory for enhanced knowledge graph")

    args = parser.parse_args()

    builder = EnhancedKnowledgeGraphBuilder(args.id_results_dir, args.output_dir)

    # Load and enhance data
    id_data = builder.load_and_enhance_data()

    # Build enhanced graph
    builder.build_enhanced_graph(id_data)

    # Export formats
    builder.export_enhanced_formats()

    print(f"\n✅ Enhanced knowledge graph complete!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🕸️ Graph: {builder.graph.number_of_nodes():,} nodes, {builder.graph.number_of_edges():,} edges")

if __name__ == "__main__":
    main()
