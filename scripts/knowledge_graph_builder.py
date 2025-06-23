#!/usr/bin/env python3
"""
Knowledge Graph Builder
Comprehensive knowledge graph construction with ChromaDB/SQLite, NetworkX, and visualization
"""

import os
import json
import pandas as pd
import sqlite3
import networkx as nx
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
import uuid
import hashlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not available, using SQLite only")

class KnowledgeGraphBuilder:
    def __init__(self, id_results_dir: str, output_dir: str, db_type: str = "sqlite"):
        self.id_results_dir = Path(id_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.db_type = db_type

        # Initialize databases
        self.sqlite_db = None
        self.chroma_client = None
        self.graph = nx.MultiDiGraph()

        # Schema definition
        self.schema = self._define_schema()

    def _define_schema(self) -> Dict[str, Any]:
        """Define the complete knowledge graph schema."""
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "nodes": {
                "conversation": {
                    "properties": {
                        "conversation_id": {"type": "string", "primary_key": True},
                        "title": {"type": "string", "indexed": True},
                        "date": {"type": "string", "indexed": True},
                        "create_time": {"type": "float", "indexed": True},
                        "model": {"type": "string", "indexed": True},
                        "message_count": {"type": "integer"},
                        "source_file": {"type": "string"}
                    },
                    "description": "Root conversation entities"
                },
                "sentence": {
                    "properties": {
                        "sentence_id": {"type": "string", "primary_key": True},
                        "text": {"type": "text", "indexed": True},
                        "conversation_id": {"type": "string", "foreign_key": "conversation.conversation_id"},
                        "position": {"type": "integer"},
                        "complexity": {"type": "float"},
                        "length": {"type": "integer"},
                        "word_count": {"type": "integer"}
                    },
                    "description": "Individual sentences within conversations"
                },
                "concept": {
                    "properties": {
                        "concept_id": {"type": "string", "primary_key": True},
                        "concept_text": {"type": "text", "indexed": True},
                        "confidence": {"type": "float"},
                        "type": {"type": "string", "indexed": True},
                        "frequency": {"type": "integer"},
                        "rag_priority": {"type": "boolean"}
                    },
                    "description": "Semantic concepts extracted from text"
                },
                "emotion": {
                    "properties": {
                        "emotion_id": {"type": "string", "primary_key": True},
                        "category": {"type": "string", "indexed": True},
                        "intensity": {"type": "float"},
                        "valence": {"type": "float"},
                        "arousal": {"type": "float"}
                    },
                    "description": "Emotional states and expressions"
                },
                "metaphor": {
                    "properties": {
                        "metaphor_id": {"type": "string", "primary_key": True},
                        "metaphor_text": {"type": "text", "indexed": True},
                        "metaphor_type": {"type": "string", "indexed": True},
                        "pattern_type": {"type": "string"},
                        "confidence": {"type": "float"},
                        "sentence_id": {"type": "string", "foreign_key": "sentence.sentence_id"}
                    },
                    "description": "Figurative language and analogies"
                },
                "project": {
                    "properties": {
                        "project_id": {"type": "string", "primary_key": True},
                        "project_name": {"type": "string", "indexed": True},
                        "project_type": {"type": "string", "indexed": True},
                        "description": {"type": "text"},
                        "status": {"type": "string"},
                        "confidence": {"type": "float"}
                    },
                    "description": "Project and system references"
                },
                "thread": {
                    "properties": {
                        "thread_id": {"type": "string", "primary_key": True},
                        "thread_type": {"type": "string", "indexed": True},
                        "description": {"type": "text"},
                        "start_conversation": {"type": "string", "foreign_key": "conversation.conversation_id"},
                        "end_conversation": {"type": "string", "foreign_key": "conversation.conversation_id"}
                    },
                    "description": "Conversation threads and continuations"
                },
                "pair": {
                    "properties": {
                        "pair_id": {"type": "string", "primary_key": True},
                        "user_message": {"type": "text"},
                        "assistant_message": {"type": "text"},
                        "conversation_id": {"type": "string", "foreign_key": "conversation.conversation_id"},
                        "position": {"type": "integer"}
                    },
                    "description": "User-assistant message pairs"
                }
            },
            "edges": {
                "contains": {
                    "from": "conversation",
                    "to": "sentence",
                    "properties": {
                        "weight": {"type": "float", "default": 1.0}
                    },
                    "description": "Conversation contains sentences"
                },
                "expresses": {
                    "from": "sentence",
                    "to": "emotion",
                    "properties": {
                        "confidence": {"type": "float"},
                        "intensity": {"type": "float"}
                    },
                    "description": "Sentence expresses emotion"
                },
                "mentions": {
                    "from": "sentence",
                    "to": "concept",
                    "properties": {
                        "relevance": {"type": "float"},
                        "frequency": {"type": "integer"}
                    },
                    "description": "Sentence mentions concept"
                },
                "uses_metaphor": {
                    "from": "sentence",
                    "to": "metaphor",
                    "properties": {
                        "confidence": {"type": "float"}
                    },
                    "description": "Sentence uses metaphor"
                },
                "discusses": {
                    "from": "conversation",
                    "to": "project",
                    "properties": {
                        "relevance": {"type": "float"},
                        "mention_count": {"type": "integer"}
                    },
                    "description": "Conversation discusses project"
                },
                "continues": {
                    "from": "thread",
                    "to": "conversation",
                    "properties": {
                        "sequence_order": {"type": "integer"}
                    },
                    "description": "Thread continues conversation"
                },
                "contains_pair": {
                    "from": "conversation",
                    "to": "pair",
                    "properties": {
                        "position": {"type": "integer"}
                    },
                    "description": "Conversation contains message pair"
                },
                "similar_to": {
                    "from": "concept",
                    "to": "concept",
                    "properties": {
                        "similarity": {"type": "float"},
                        "relationship_type": {"type": "string"}
                    },
                    "description": "Concept similarity relationships"
                },
                "depends_on": {
                    "from": "concept",
                    "to": "concept",
                    "properties": {
                        "dependency_strength": {"type": "float"},
                        "dependency_type": {"type": "string"}
                    },
                    "description": "Concept dependency relationships"
                }
            }
        }

    def initialize_databases(self):
        """Initialize SQLite and ChromaDB databases."""
        print("🗄️ Initializing databases...")

        # Initialize SQLite
        sqlite_path = self.output_dir / "knowledge_graph.db"
        self.sqlite_db = sqlite3.connect(str(sqlite_path))
        self._create_sqlite_schema()

        # Initialize ChromaDB if available
        if CHROMADB_AVAILABLE and self.db_type == "chromadb":
            chroma_path = self.output_dir / "chromadb"
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            self._create_chroma_collections()

        print("✅ Databases initialized")

    def _create_sqlite_schema(self):
        """Create SQLite tables based on schema."""
        cursor = self.sqlite_db.cursor()

        # Create node tables
        for node_type, node_schema in self.schema["nodes"].items():
            columns = []
            for prop_name, prop_schema in node_schema["properties"].items():
                sql_type = self._get_sql_type(prop_schema["type"])
                if prop_schema.get("primary_key"):
                    columns.append(f"{prop_name} {sql_type} PRIMARY KEY")
                else:
                    columns.append(f"{prop_name} {sql_type}")

            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {node_type} (
                {', '.join(columns)}
            )
            """
            cursor.execute(create_sql)

            # Create indexes
            for prop_name, prop_schema in node_schema["properties"].items():
                if prop_schema.get("indexed"):
                    index_sql = f"CREATE INDEX IF NOT EXISTS idx_{node_type}_{prop_name} ON {node_type}({prop_name})"
                    cursor.execute(index_sql)

        # Create edge tables
        for edge_type, edge_schema in self.schema["edges"].items():
            columns = [
                "edge_id TEXT PRIMARY KEY",
                f"from_node TEXT",
                f"to_node TEXT",
                "edge_type TEXT"
            ]

            for prop_name, prop_schema in edge_schema["properties"].items():
                sql_type = self._get_sql_type(prop_schema["type"])
                columns.append(f"{prop_name} {sql_type}")

            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {edge_type} (
                {', '.join(columns)}
            )
            """
            cursor.execute(create_sql)

        self.sqlite_db.commit()

    def _get_sql_type(self, schema_type: str) -> str:
        """Convert schema type to SQLite type."""
        type_mapping = {
            "string": "TEXT",
            "text": "TEXT",
            "integer": "INTEGER",
            "float": "REAL",
            "boolean": "INTEGER"
        }
        return type_mapping.get(schema_type, "TEXT")

    def _create_chroma_collections(self):
        """Create ChromaDB collections for vector search."""
        if not self.chroma_client:
            return

        # Create collections for each node type that needs vector search
        vector_node_types = ["sentence", "concept", "metaphor"]

        for node_type in vector_node_types:
            try:
                collection_name = f"{node_type}_vectors"
                self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"node_type": node_type}
                )
            except Exception as e:
                print(f"⚠️ Error creating ChromaDB collection {node_type}: {e}")

    def load_id_data(self) -> Dict[str, pd.DataFrame]:
        """Load all ID data from CSV files."""
        print("📂 Loading ID data...")

        id_data = {}
        expected_files = [
            'conversation_ids.csv', 'sentence_ids.csv', 'concept_ids.csv',
            'emotion_ids.csv', 'metaphor_ids.csv', 'project_ids.csv',
            'thread_ids.csv', 'pair_ids.csv'
        ]

        for filename in expected_files:
            filepath = self.id_results_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    id_type = filename.replace('_ids.csv', '')
                    id_data[id_type] = df
                    print(f"  ✅ Loaded {len(df)} {id_type} records")
                except Exception as e:
                    print(f"  ⚠️ Error loading {filename}: {e}")
            else:
                print(f"  ❌ Missing {filename}")

        return id_data

    def populate_graph(self, id_data: Dict[str, pd.DataFrame]):
        """Populate the NetworkX graph with nodes and edges."""
        print("🕸️ Populating knowledge graph...")

        # Add nodes
        for node_type, df in id_data.items():
            if node_type in self.schema["nodes"]:
                for _, row in df.iterrows():
                    node_id = row.get(f"{node_type}_id", str(uuid.uuid4()))
                    node_attrs = {k: v for k, v in row.items() if pd.notna(v)}
                    self.graph.add_node(node_id, node_type=node_type, **node_attrs)

        # Add edges based on relationships
        self._add_conversation_edges(id_data)
        self._add_semantic_edges(id_data)
        self._add_concept_relationships(id_data)

        print(f"✅ Graph populated: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _add_conversation_edges(self, id_data: Dict[str, pd.DataFrame]):
        """Add conversation-related edges."""
        if 'conversation' in id_data and 'sentence' in id_data:
            conv_df = id_data['conversation']
            sent_df = id_data['sentence']

            # Add contains edges (conversation -> sentence)
            if 'conversation_id' in sent_df.columns:
                for _, sent_row in sent_df.iterrows():
                    conv_id = sent_row['conversation_id']
                    sent_id = sent_row['sentence_id']
                    if conv_id in self.graph and sent_id in self.graph:
                        self.graph.add_edge(conv_id, sent_id, edge_type='contains', weight=1.0)

    def _add_semantic_edges(self, id_data: Dict[str, pd.DataFrame]):
        """Add semantic relationship edges."""
        if 'sentence' in id_data:
            sent_df = id_data['sentence']

            # Add sentence -> concept edges
            if 'concept' in id_data:
                concept_df = id_data['concept']
                for _, sent_row in sent_df.head(1000).iterrows():  # Sample
                    sent_id = sent_row['sentence_id']
                    for _, concept_row in concept_df.head(100).iterrows():  # Sample
                        concept_id = concept_row['concept_id']
                        if sent_id in self.graph and concept_id in self.graph:
                            self.graph.add_edge(sent_id, concept_id, edge_type='mentions', relevance=0.8)

            # Add sentence -> emotion edges
            if 'emotion' in id_data:
                emotion_df = id_data['emotion']
                for _, sent_row in sent_df.head(1000).iterrows():  # Sample
                    sent_id = sent_row['sentence_id']
                    for _, emotion_row in emotion_df.iterrows():
                        emotion_id = emotion_row['emotion_id']
                        if sent_id in self.graph and emotion_id in self.graph:
                            self.graph.add_edge(sent_id, emotion_id, edge_type='expresses', confidence=0.7)

    def _add_concept_relationships(self, id_data: Dict[str, pd.DataFrame]):
        """Add concept-to-concept relationships."""
        if 'concept' in id_data:
            concept_df = id_data['concept']
            concepts = concept_df.head(50).iterrows()  # Sample for relationships

            for i, (_, concept1) in enumerate(concepts):
                for j, (_, concept2) in enumerate(concepts):
                    if i != j:
                        concept1_id = concept1['concept_id']
                        concept2_id = concept2['concept_id']

                        if concept1_id in self.graph and concept2_id in self.graph:
                            # Add similarity edge
                            self.graph.add_edge(concept1_id, concept2_id,
                                              edge_type='similar_to',
                                              similarity=0.6)

    def export_graph_formats(self):
        """Export graph in multiple formats."""
        print("📤 Exporting graph formats...")

        # Export GraphML
        graphml_path = self.output_dir / "knowledge_graph.graphml"
        nx.write_graphml(self.graph, str(graphml_path))
        print(f"  ✅ GraphML: {graphml_path}")

        # Export JSON
        json_path = self.output_dir / "knowledge_graph.json"
        graph_data = nx.node_link_data(self.graph)
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"  ✅ JSON: {json_path}")

        # Export GML
        gml_path = self.output_dir / "knowledge_graph.gml"
        nx.write_gml(self.graph, str(gml_path))
        print(f"  ✅ GML: {gml_path}")

        # Export adjacency list
        adj_path = self.output_dir / "knowledge_graph_adjacency.txt"
        with open(adj_path, 'w') as f:
            for node in self.graph.nodes():
                neighbors = list(self.graph.neighbors(node))
                f.write(f"{node}: {', '.join(neighbors)}\n")
        print(f"  ✅ Adjacency list: {adj_path}")

    def generate_erd_diagram(self):
        """Generate Entity-Relationship Diagram."""
        print("📊 Generating ERD diagram...")

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Colors for different node types
        colors = {
            'conversation': '#FF6B6B',
            'sentence': '#4ECDC4',
            'concept': '#45B7D1',
            'emotion': '#96CEB4',
            'metaphor': '#FFEAA7',
            'project': '#DDA0DD',
            'thread': '#98D8C8',
            'pair': '#F7DC6F'
        }

        # Position nodes in a circular layout
        node_positions = {}
        node_types = list(self.schema["nodes"].keys())
        radius = 3
        center_x, center_y = 5, 5

        for i, node_type in enumerate(node_types):
            angle = 2 * np.pi * i / len(node_types)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            node_positions[node_type] = (x, y)

            # Draw node box
            color = colors.get(node_type, '#CCCCCC')
            box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(box)

            # Add node label
            ax.text(x, y, node_type.replace('_', '\n').title(),
                   ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw edges
        for edge_type, edge_schema in self.schema["edges"].items():
            from_node = edge_schema["from"]
            to_node = edge_schema["to"]

            if from_node in node_positions and to_node in node_positions:
                x1, y1 = node_positions[from_node]
                x2, y2 = node_positions[to_node]

                # Draw arrow
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

                # Add edge label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, edge_type, ha='center', va='center',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Add title
        ax.text(5, 9, 'Knowledge Graph Schema - Entity Relationship Diagram',
               ha='center', va='center', fontsize=16, fontweight='bold')

        # Add legend
        legend_elements = [mpatches.Patch(color=color, label=node_type.replace('_', ' ').title())
                          for node_type, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

        # Save diagram
        erd_path = self.output_dir / "knowledge_graph_erd.png"
        plt.tight_layout()
        plt.savefig(erd_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ ERD diagram: {erd_path}")

    def generate_schema_documentation(self):
        """Generate comprehensive schema documentation in markdown."""
        print("📝 Generating schema documentation...")

        md_content = f"""# Knowledge Graph Schema Documentation

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Version:** {self.schema['version']}

## Overview

This knowledge graph represents the semantic relationships between conversations, sentences, concepts, emotions, metaphors, projects, threads, and message pairs extracted from ChatGPT conversations.

## Node Types

"""

        # Node documentation
        for node_type, node_schema in self.schema["nodes"].items():
            md_content += f"### {node_type.replace('_', ' ').title()}\n\n"
            md_content += f"{node_schema['description']}\n\n"
            md_content += "| Property | Type | Indexed | Description |\n"
            md_content += "|----------|------|---------|-------------|\n"

            for prop_name, prop_schema in node_schema["properties"].items():
                indexed = "✅" if prop_schema.get("indexed") else "❌"
                pk = " (PK)" if prop_schema.get("primary_key") else ""
                fk = " (FK)" if prop_schema.get("foreign_key") else ""
                md_content += f"| {prop_name}{pk}{fk} | {prop_schema['type']} | {indexed} | |\n"

            md_content += "\n"

        # Edge documentation
        md_content += "## Edge Types\n\n"

        for edge_type, edge_schema in self.schema["edges"].items():
            md_content += f"### {edge_type.replace('_', ' ').title()}\n\n"
            md_content += f"**From:** {edge_schema['from']}  \n"
            md_content += f"**To:** {edge_schema['to']}  \n"
            md_content += f"**Description:** {edge_schema['description']}\n\n"

            if edge_schema["properties"]:
                md_content += "| Property | Type | Default | Description |\n"
                md_content += "|----------|------|---------|-------------|\n"

                for prop_name, prop_schema in edge_schema["properties"].items():
                    default = prop_schema.get("default", "")
                    md_content += f"| {prop_name} | {prop_schema['type']} | {default} | |\n"

            md_content += "\n"

        # Statistics
        md_content += "## Graph Statistics\n\n"
        md_content += f"- **Total Nodes:** {self.graph.number_of_nodes()}\n"
        md_content += f"- **Total Edges:** {self.graph.number_of_edges()}\n"
        md_content += f"- **Node Types:** {len(self.schema['nodes'])}\n"
        md_content += f"- **Edge Types:** {len(self.schema['edges'])}\n\n"

        # Node type counts
        node_counts = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('node_type', 'unknown')
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        md_content += "### Node Distribution\n\n"
        md_content += "| Node Type | Count |\n"
        md_content += "|-----------|-------|\n"
        for node_type, count in sorted(node_counts.items()):
            md_content += f"| {node_type} | {count:,} |\n"

        md_content += "\n"

        # Export formats
        md_content += "## Export Formats\n\n"
        md_content += "The knowledge graph is exported in multiple formats:\n\n"
        md_content += "- **GraphML:** `knowledge_graph.graphml` - Standard graph format\n"
        md_content += "- **JSON:** `knowledge_graph.json` - JavaScript/Web compatible\n"
        md_content += "- **GML:** `knowledge_graph.gml` - Graph Modeling Language\n"
        md_content += "- **SQLite:** `knowledge_graph.db` - Relational database\n"
        if CHROMADB_AVAILABLE:
            md_content += "- **ChromaDB:** `chromadb/` - Vector database for similarity search\n"

        md_content += "\n## Usage Examples\n\n"
        md_content += "### NetworkX\n```python\nimport networkx as nx\nG = nx.read_graphml('knowledge_graph.graphml')\n```\n\n"
        md_content += "### D3.js\n```javascript\nfetch('knowledge_graph.json')\n  .then(response => response.json())\n  .then(data => {\n    // Use with D3.js force simulation\n  });\n```\n\n"
        md_content += "### Plotly\n```python\nimport plotly.graph_objects as go\nimport json\n\nwith open('knowledge_graph.json', 'r') as f:\n    data = json.load(f)\n# Create network visualization\n```\n\n"

        # Save documentation
        doc_path = self.output_dir / "knowledge_graph_schema.md"
        with open(doc_path, 'w') as f:
            f.write(md_content)

        print(f"  ✅ Schema documentation: {doc_path}")

    def build_complete_graph(self):
        """Build the complete knowledge graph."""
        print("🚀 Building complete knowledge graph...")

        # Initialize databases
        self.initialize_databases()

        # Load ID data
        id_data = self.load_id_data()

        # Populate graph
        self.populate_graph(id_data)

        # Export formats
        self.export_graph_formats()

        # Generate visualizations
        self.generate_erd_diagram()
        self.generate_schema_documentation()

        # Save schema
        schema_path = self.output_dir / "knowledge_graph_schema.json"
        with open(schema_path, 'w') as f:
            json.dump(self.schema, f, indent=2)

        print(f"✅ Complete knowledge graph built!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

def main():
    parser = argparse.ArgumentParser(description="Build comprehensive knowledge graph")
    parser.add_argument("--id-results-dir", default="integrated_20250623_011924/id_generation_results_new",
                       help="Directory containing ID generation results")
    parser.add_argument("--output-dir", default="integrated_20250623_011924/knowledge_graph",
                       help="Output directory for knowledge graph")
    parser.add_argument("--db-type", choices=["sqlite", "chromadb"], default="sqlite",
                       help="Database type to use")

    args = parser.parse_args()

    builder = KnowledgeGraphBuilder(args.id_results_dir, args.output_dir, args.db_type)
    builder.build_complete_graph()

if __name__ == "__main__":
    main()
