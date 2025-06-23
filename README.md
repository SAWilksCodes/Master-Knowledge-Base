# ChatGPT Conversation Analysis Pipeline

A comprehensive framework for analyzing, processing, and visualizing ChatGPT conversation data with advanced semantic analysis, ID generation, and multimodal alignment capabilities.

## 🚀 Features

### Core Analysis
- **Deep Conversation Parsing**: Multi-layer analysis of conversations with linguistic, semantic, and cognitive insights
- **Comprehensive ID Generation**: 50+ unique ID types for entities, relationships, and knowledge graph construction
- **Semantic Anchor System**: Advanced semantic analysis with clustering and relationship mapping
- **Temporal Evolution Tracking**: Learning curve analysis and conversation progression modeling

### Data Processing
- **ChromaDB Integration**: Modern vector database with new client format for efficient storage and retrieval
- **Knowledge Graph Construction**: Comprehensive graph schema with semantic relationships
- **Multimodal Alignment**: Audio, image, and gesture data integration
- **Full Ingestion Pipeline**: End-to-end processing from raw conversations to analysis

### Visualization & Insights
- **Interactive Visualizations**: Plotly-based dashboards and 3D semantic spaces
- **Network Analysis**: Relationship graphs and entity connections
- **Temporal Analysis**: Evolution tracking and learning progression
- **Word Clouds & Distributions**: Statistical analysis and pattern recognition

## 📁 Project Structure

```
chat/
├── scripts/                          # Core processing scripts
│   ├── deep_parse_conversations_enhanced.py
│   ├── generate_all_ids.py
│   ├── semantic_anchor_system_*.py
│   ├── chromadb_migration.py
│   ├── full_ingestion_pipeline.py
│   ├── visualization_layer.py
│   ├── multimodal_alignment.py
│   └── master_process.py
├── processed_*/                      # Processed data outputs
├── enhanced_analysis/                # Enhanced semantic analysis
├── semantic_anchors/                 # Semantic anchor data
├── id_generation_results/            # Generated IDs
├── long_convos/                      # Long conversation analysis
├── by_date/                          # Date-organized conversations
├── split_chats/                      # Split conversation files
└── Audio/ & Image/                   # Multimodal data
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB+ recommended for large datasets)
- 10GB+ disk space

### Setup

1. **Clone and navigate to the project:**
```bash
cd chat
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Install additional dependencies:**
```bash
# For enhanced parsing
pip install spacy textblob
python -m spacy download en_core_web_sm

# For ChromaDB
pip install chromadb sentence-transformers

# For visualization
pip install plotly networkx seaborn wordcloud

# For multimodal processing
pip install librosa soundfile opencv-python pillow whisper

# For machine learning
pip install scikit-learn umap-learn torch transformers
```

4. **Verify installation:**
```bash
python scripts/test_id_generation.py
```

## 📖 Quick Start

### 1. Basic Conversation Processing

```bash
# Process a single conversation file
python scripts/master_process.py --input conversations.json --operations all

# Process with specific operations
python scripts/master_process.py --input conversations.json --operations split,pairs,analysis
```

### 2. Deep Semantic Analysis

```bash
# Run enhanced deep parsing
python scripts/deep_parse_conversations_enhanced.py --input split_chats/ --output parsed_data/

# Generate comprehensive IDs
python scripts/generate_all_ids.py --input-dir parsed_data/ --output-dir generated_ids/
```

### 3. ChromaDB Migration

```bash
# Migrate data to ChromaDB
python scripts/chromadb_migration.py --data-dir generated_ids/ --chroma-dir ./chroma_db
```

### 4. Full Pipeline

```bash
# Run complete ingestion pipeline
python scripts/full_ingestion_pipeline.py --input conversations.json --stages all
```

### 5. Visualization

```bash
# Create interactive visualizations
python scripts/visualization_layer.py --data-dir generated_ids/ --output-dir visualizations/
```

## 🔧 Advanced Usage

### Semantic Anchor Analysis

```bash
# Generate semantic anchors
python scripts/semantic_anchor_system_enhanced.py --input-dir split_chats/ --output-dir semantic_anchors/

# Analyze anchor relationships
python scripts/analyze_semantic_anchors.py --input semantic_anchors/
```

### Multimodal Alignment

```bash
# Align conversation with audio, image, and gesture data
python scripts/multimodal_alignment.py \
    --conversation-dir processed_conversations/ \
    --audio-dir Audio/ \
    --image-dir Image/ \
    --gesture-dir gestures/
```

### Custom ID Generation

```bash
# Generate specific ID types
python scripts/generate_conversation_ids.py --input-dir split_chats/
python scripts/generate_semantic_anchors.py --input-dir parsed_data/
python scripts/generate_emotion_ids.py --input-dir parsed_data/
```

## 📊 Understanding the Outputs

### 1. Processed Data Structure

```
processed_YYYYMMDD_HHMMSS/
├── 01_split_json/           # Individual conversation files
├── 02_split_markdown/       # Markdown versions
├── 03_organized_by_date/    # Date-organized conversations
├── 04_organized_by_topic/   # Topic-organized conversations
├── 05_organized_by_model/   # Model-organized conversations
├── 06_long_conversations/   # Long conversation analysis
├── 07_extracted_data/       # Extracted code, Q&A pairs
├── 08_message_pairs/        # User-assistant message pairs
├── 09_training_data/        # Training data for ML models
└── 10_reports/              # Analysis reports and indices
```

### 2. ID Generation Results

```
id_generation_results/
├── conversation_ids.csv     # Conversation-level IDs
├── message_ids.csv          # Message-level IDs
├── sentence_ids.csv         # Sentence-level IDs
├── semantic_anchors.csv     # Semantic anchor IDs
├── emotion_ids.csv          # Emotion analysis IDs
├── concept_ids.csv          # Concept extraction IDs
└── sentence_analysis/       # Detailed sentence analysis
```

### 3. Semantic Analysis

```
semantic_anchors/
├── semantic_anchors_YYYYMMDD_HHMMSS.csv
├── semantic_anchor_report_YYYYMMDD_HHMMSS.json
├── embedding_cache.json
└── cluster_analysis/
```

### 4. Visualization Outputs

```
visualizations/
├── interactive/
│   ├── dashboard.html           # Main dashboard
│   ├── semantic_space_*.html    # 2D semantic spaces
│   ├── network_graph_*.html     # Network visualizations
│   └── temporal_evolution.html  # Time-based analysis
├── static/
│   ├── wordcloud_*.png          # Word clouds
│   └── distribution_*.png       # Statistical plots
└── index.html                   # Navigation hub
```

## 🎯 Key Concepts

### ID Types

The system generates 50+ unique ID types across categories:

**Core Structural IDs:**
- `conversation_id`: Unique conversation identifier
- `message_id`: Individual message identifier
- `pair_id`: User-assistant message pair identifier
- `sentence_id`: Sentence-level identifier
- `word_id`: Word-level identifier

**Content & Meaning IDs:**
- `topic_id`: Topic classification identifier
- `concept_id`: Conceptual entity identifier
- `metaphor_id`: Metaphorical expression identifier
- `emotion_id`: Emotional content identifier
- `intent_id`: User intent identifier

**Semantic Analysis IDs:**
- `semantic_anchor_id`: Semantic anchor identifier
- `cluster_id`: Semantic cluster identifier
- `synonym_id`: Synonym relationship identifier
- `reference_id`: Cross-reference identifier

**Temporal & Evolution IDs:**
- `sequence_id`: Temporal sequence identifier
- `evolution_id`: Evolution tracking identifier
- `milestone_id`: Learning milestone identifier
- `learning_curve_id`: Learning progression identifier

### Semantic Anchors

Semantic anchors are high-value content pieces that serve as:
- **Knowledge retrieval points** for RAG systems
- **Relationship mapping nodes** in knowledge graphs
- **Learning progression markers** for educational analysis
- **Content clustering centers** for topic organization

### ChromaDB Integration

The system uses ChromaDB for:
- **Vector storage** of conversation embeddings
- **Similarity search** for content retrieval
- **Metadata indexing** for efficient querying
- **Persistent storage** with new client format

## 🔍 Analysis Examples

### 1. Learning Progression Analysis

```python
# Analyze how conversation complexity evolves
from scripts.deep_parse_conversations_enhanced import EnhancedConversationParser

parser = EnhancedConversationParser()
conversation_data = parser.process_conversation(conversation_file)
learning_curve = conversation_data['learning_curve']

print(f"Learning Stage: {learning_curve['stage']}")
print(f"Progress: {learning_curve['progress']:.2f}")
print(f"Complexity Trend: {learning_curve['complexity_trend']:.3f}")
```

### 2. Semantic Similarity Search

```python
# Search for similar content in ChromaDB
from scripts.chromadb_migration import ChromaDBManager

chroma = ChromaDBManager()
results = chroma.search_similar(
    query="How do I implement machine learning?",
    collection_name='messages',
    n_results=10
)

for result in results:
    print(f"Similarity: {result['distance']:.3f}")
    print(f"Content: {result['document'][:100]}...")
```

### 3. Network Analysis

```python
# Analyze semantic anchor relationships
import networkx as nx
from scripts.visualization_layer import SemanticVisualizationLayer

viz = SemanticVisualizationLayer('data_dir')
fig = viz.create_network_graph('semantic_anchors')
fig.show()
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size in configuration
   - Process files individually
   - Use streaming processors for large datasets

2. **Missing Dependencies**
   - Install spaCy model: `python -m spacy download en_core_web_sm`
   - Install Whisper: `pip install openai-whisper`
   - Install ChromaDB: `pip install chromadb`

3. **File Encoding Issues**
   - Use UTF-8 encoding for all files
   - Run `python scripts/fix_unicode.py` to fix encoding issues

4. **ChromaDB Connection Issues**
   - Check ChromaDB version compatibility
   - Ensure sufficient disk space
   - Reset collections if needed

### Performance Optimization

1. **For Large Datasets:**
   - Use batch processing
   - Enable multiprocessing
   - Use SSD storage for ChromaDB

2. **For Memory Constraints:**
   - Process files in smaller batches
   - Use streaming processors
   - Clear memory between operations

## 📈 Advanced Configuration

### Custom Configuration

Create a `config.json` file:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "chroma_persist_dir": "./chroma_db",
  "batch_size": 100,
  "max_workers": 4,
  "temporal_window": 5.0,
  "similarity_threshold": 0.7
}
```

### Environment Variables

```bash
export CHAT_ANALYSIS_BASE_DIR="/path/to/chat/data"
export CHAT_ANALYSIS_OUTPUT_DIR="/path/to/output"
export CHAT_ANALYSIS_CHROMA_DIR="/path/to/chromadb"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for ChatGPT conversation format
- ChromaDB team for vector database
- spaCy team for NLP processing
- Plotly team for visualization tools
- Hugging Face for transformer models

## 📞 Support

For questions and support:
- Check the troubleshooting section
- Review example outputs
- Examine the code comments
- Create an issue with detailed error information

---

**Happy Analyzing! 🎉**
