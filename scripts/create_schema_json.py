#!/usr/bin/env python3
"""
Create Schema JSON
Generate the knowledge graph schema JSON file
"""

import json
from datetime import datetime
from pathlib import Path

def create_schema():
    """Create the complete knowledge graph schema."""
    schema = {
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
    return schema

def main():
    # Create schema
    schema = create_schema()

    # Save to file
    output_path = "integrated_20250623_011924/knowledge_graph/knowledge_graph_schema.json"

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"✅ Schema JSON created: {output_path}")
    print(f"📊 Schema contains:")
    print(f"   - {len(schema['nodes'])} node types")
    print(f"   - {len(schema['edges'])} edge types")
    print(f"   - Version: {schema['version']}")

if __name__ == "__main__":
    main()
