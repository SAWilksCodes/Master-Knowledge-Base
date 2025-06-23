#!/usr/bin/env python3
"""
Fix Schema Documentation
Generate the complete schema documentation in markdown format
"""

import json
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any

def generate_schema_documentation(graph_path: str, schema_path: str, output_path: str):
    """Generate comprehensive schema documentation in markdown."""
    print("📝 Generating schema documentation...")

    # Load schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Load graph
    graph = nx.read_graphml(graph_path)

    md_content = f"""# Knowledge Graph Schema Documentation

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Version:** {schema['version']}

## Overview

This knowledge graph represents the semantic relationships between conversations, sentences, concepts, emotions, metaphors, projects, threads, and message pairs extracted from ChatGPT conversations.

## Node Types

"""

    # Node documentation
    for node_type, node_schema in schema["nodes"].items():
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

    for edge_type, edge_schema in schema["edges"].items():
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
    md_content += f"- **Total Nodes:** {graph.number_of_nodes():,}\n"
    md_content += f"- **Total Edges:** {graph.number_of_edges():,}\n"
    md_content += f"- **Node Types:** {len(schema['nodes'])}\n"
    md_content += f"- **Edge Types:** {len(schema['edges'])}\n\n"

    # Node type counts
    node_counts = {}
    for node in graph.nodes():
        node_type = graph.nodes[node].get('node_type', 'unknown')
        node_counts[node_type] = node_counts.get(node_type, 0) + 1

    md_content += "### Node Distribution\n\n"
    md_content += "| Node Type | Count |\n"
    md_content += "|-----------|-------|\n"
    for node_type, count in sorted(node_counts.items()):
        md_content += f"| {node_type} | {count:,} |\n"

    md_content += "\n"

    # Edge type counts
    edge_counts = {}
    for edge in graph.edges(data=True):
        edge_type = edge[2].get('edge_type', 'unknown')
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1

    md_content += "### Edge Distribution\n\n"
    md_content += "| Edge Type | Count |\n"
    md_content += "|-----------|-------|\n"
    for edge_type, count in sorted(edge_counts.items()):
        md_content += f"| {edge_type} | {count:,} |\n"

    md_content += "\n"

    # Export formats
    md_content += "## Export Formats\n\n"
    md_content += "The knowledge graph is exported in multiple formats:\n\n"
    md_content += "- **GraphML:** `knowledge_graph.graphml` - Standard graph format for Gephi, Cytoscape, NetworkX\n"
    md_content += "- **JSON:** `knowledge_graph.json` - JavaScript/Web compatible for D3.js, Plotly\n"
    md_content += "- **GML:** `knowledge_graph.gml` - Graph Modeling Language for other graph tools\n"
    md_content += "- **SQLite:** `knowledge_graph.db` - Relational database for SQL queries\n"
    md_content += "- **Adjacency:** `knowledge_graph_adjacency.txt` - Simple adjacency list format\n"
    md_content += "- **ERD:** `knowledge_graph_erd.png` - Entity-Relationship Diagram\n"

    md_content += "\n## Usage Examples\n\n"

    # NetworkX example
    md_content += "### NetworkX\n```python\nimport networkx as nx\n\n# Load the graph\nG = nx.read_graphml('knowledge_graph.graphml')\n\n# Basic statistics\nprint(f\"Nodes: {G.number_of_nodes():,}\")\nprint(f\"Edges: {G.number_of_edges():,}\")\n\n# Find all conversations\nconversations = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'conversation']\nprint(f\"Conversations: {len(conversations)}\")\n\n# Find all concepts\nconcepts = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'concept']\nprint(f\"Concepts: {len(concepts)}\")\n```\n\n"

    # D3.js example
    md_content += "### D3.js\n```javascript\n// Load the graph data\nfetch('knowledge_graph.json')\n  .then(response => response.json())\n  .then(data => {\n    // Create force simulation\n    const simulation = d3.forceSimulation(data.nodes)\n      .force(\"link\", d3.forceLink(data.links).id(d => d.id))\n      .force(\"charge\", d3.forceManyBody().strength(-100))\n      .force(\"center\", d3.forceCenter(width / 2, height / 2));\n    \n    // Create SVG and render graph\n    // ... (full D3.js implementation)\n  });\n```\n\n"

    # Plotly example
    md_content += "### Plotly\n```python\nimport plotly.graph_objects as go\nimport json\nimport networkx as nx\n\n# Load graph data\nwith open('knowledge_graph.json', 'r') as f:\n    data = json.load(f)\n\n# Create network visualization\nfig = go.Figure(data=[go.Scatter(\n    x=[], y=[],\n    mode='markers+lines',\n    hoverinfo='text',\n    text=[],\n    line=dict(width=0.5, color='#888'),\n    marker=dict(size=5)\n)])\n\nfig.update_layout(\n    title='Knowledge Graph Visualization',\n    showlegend=False,\n    hovermode='closest',\n    margin=dict(b=20,l=5,r=5,t=40)\n)\n\nfig.show()\n```\n\n"

    # SQLite example
    md_content += "### SQLite Queries\n```sql\n-- Find all conversations with their sentence counts\nSELECT c.title, COUNT(s.sentence_id) as sentence_count\nFROM conversation c\nLEFT JOIN sentence s ON c.conversation_id = s.conversation_id\nGROUP BY c.conversation_id\nORDER BY sentence_count DESC;\n\n-- Find most frequent concepts\nSELECT concept_text, COUNT(*) as frequency\nFROM concept\nGROUP BY concept_text\nORDER BY frequency DESC\nLIMIT 10;\n\n-- Find emotional patterns\nSELECT e.category, COUNT(*) as count\nFROM emotion e\nJOIN sentence s ON s.sentence_id = e.sentence_id\nGROUP BY e.category\nORDER BY count DESC;\n```\n\n"

    # Sample data
    md_content += "## Sample Data\n\n"

    # Sample conversations
    conversations = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'conversation'][:5]
    md_content += "### Sample Conversations\n\n"
    for conv_id in conversations:
        conv_data = graph.nodes[conv_id]
        md_content += f"- **{conv_data.get('title', 'Untitled')}** (ID: {conv_id})\n"
        md_content += f"  - Date: {conv_data.get('date', 'Unknown')}\n"
        md_content += f"  - Messages: {conv_data.get('message_count', 0)}\n\n"

    # Sample concepts
    concepts = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'concept'][:10]
    md_content += "### Sample Concepts\n\n"
    for concept_id in concepts:
        concept_data = graph.nodes[concept_id]
        md_content += f"- **{concept_data.get('concept_text', 'Unknown')}** (ID: {concept_id})\n"
        md_content += f"  - Type: {concept_data.get('type', 'Unknown')}\n"
        md_content += f"  - Confidence: {concept_data.get('confidence', 0):.2f}\n\n"

    # Save documentation
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✅ Schema documentation generated: {output_path}")
    return output_path

def main():
    # Paths
    graph_path = "integrated_20250623_011924/knowledge_graph/knowledge_graph.graphml"
    schema_path = "integrated_20250623_011924/knowledge_graph/knowledge_graph_schema.json"
    output_path = "integrated_20250623_011924/knowledge_graph/knowledge_graph_schema.md"

    # Check if files exist
    if not Path(graph_path).exists():
        print(f"❌ Graph file not found: {graph_path}")
        return

    if not Path(schema_path).exists():
        print(f"❌ Schema file not found: {schema_path}")
        return

    # Generate documentation
    output_file = generate_schema_documentation(graph_path, schema_path, output_path)

    print(f"\n✅ Schema documentation complete!")
    print(f"📄 File: {output_file}")
    print(f"📊 Graph: {graph_path}")
    print(f"🗂️ Schema: {schema_path}")

if __name__ == "__main__":
    main()
