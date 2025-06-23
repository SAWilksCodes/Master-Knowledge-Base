import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import hashlib
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import umap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticVisualizationLayer:
    """Comprehensive visualization layer for semantic conversation data."""

    def __init__(self, data_dir: str, output_dir: str = "visualizations"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Data storage
        self.conversations = []
        self.messages = []
        self.sentences = []
        self.semantic_anchors = []
        self.entities = []
        self.topics = []
        self.emotions = []
        self.intents = []
        self.domains = []

        # Visualization settings
        self.colors = {
            'conversations': '#1f77b4',
            'messages': '#ff7f0e',
            'sentences': '#2ca02c',
            'semantic_anchors': '#d62728',
            'entities': '#9467bd',
            'topics': '#8c564b',
            'emotions': '#e377c2',
            'intents': '#7f7f7f',
            'domains': '#bcbd22'
        }

        # Load data
        self.load_data()

    def load_data(self):
        """Load all data files from the data directory."""
        logger.info("Loading data from directory...")

        data_files = {
            'conversations': 'conversations_*.json',
            'messages': 'messages_*.json',
            'sentences': 'sentences_*.json',
            'semantic_anchors': 'semantic_anchors_*.json',
            'entities': 'entities_*.json',
            'topics': 'topics_*.json',
            'emotions': 'emotions_*.json',
            'intents': 'intents_*.json',
            'domains': 'domains_*.json'
        }

        for data_type, pattern in data_files.items():
            files = list(self.data_dir.glob(pattern))
            if files:
                # Load the most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    setattr(self, data_type, data)
                    logger.info(f"Loaded {len(data)} {data_type} from {latest_file}")
                except Exception as e:
                    logger.error(f"Failed to load {data_type}: {e}")
            else:
                logger.warning(f"No files found for {data_type}")

    def create_semantic_space_visualization(self, data_type: str = 'semantic_anchors',
                                         n_samples: int = 1000) -> go.Figure:
        """Create 2D semantic space visualization using dimensionality reduction."""
        logger.info(f"Creating semantic space visualization for {data_type}")

        data = getattr(self, data_type, [])
        if not data:
            logger.error(f"No data available for {data_type}")
            return go.Figure()

        # Sample data if too large
        if len(data) > n_samples:
            data = np.random.choice(data, n_samples, replace=False).tolist()

        # Extract text and metadata
        texts = []
        metadata = []

        for item in data:
            if data_type == 'semantic_anchors':
                texts.append(item.get('text', ''))
                metadata.append({
                    'type': item.get('type', ''),
                    'confidence': item.get('confidence', 0),
                    'frequency': item.get('frequency', 1)
                })
            elif data_type == 'messages':
                texts.append(item.get('content', ''))
                metadata.append({
                    'speaker': item.get('speaker', ''),
                    'complexity': item.get('complexity', {}).get('score', 0)
                })
            elif data_type == 'sentences':
                texts.append(item.get('text', ''))
                metadata.append({
                    'polarity': item.get('polarity', ''),
                    'subjectivity': item.get('subjectivity', 0)
                })

        # Create simple embeddings (word frequency vectors)
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            embeddings = vectorizer.fit_transform(texts).toarray()
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return go.Figure()

        # Dimensionality reduction
        try:
            # Try UMAP first (better for semantic relationships)
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
        except Exception as e:
            logger.warning(f"UMAP failed, using t-SNE: {e}")
            try:
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                coords = reducer.fit_transform(embeddings)
            except Exception as e2:
                logger.warning(f"t-SNE failed, using PCA: {e2}")
                reducer = PCA(n_components=2)
                coords = reducer.fit_transform(embeddings)

        # Create scatter plot
        fig = go.Figure()

        # Color by metadata
        if data_type == 'semantic_anchors':
            colors = [item['confidence'] for item in metadata]
            hover_text = [f"Text: {text[:50]}...<br>Type: {meta['type']}<br>Confidence: {meta['confidence']:.2f}"
                         for text, meta in zip(texts, metadata)]
        elif data_type == 'messages':
            colors = [item['complexity'] for item in metadata]
            hover_text = [f"Speaker: {meta['speaker']}<br>Complexity: {meta['complexity']:.2f}<br>Text: {text[:50]}..."
                         for text, meta in zip(texts, metadata)]
        else:
            colors = [i for i in range(len(texts))]
            hover_text = [f"Text: {text[:50]}..." for text in texts]

        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f"{data_type.replace('_', ' ').title()} Score")
            ),
            text=hover_text,
            hoverinfo='text',
            name=data_type.replace('_', ' ').title()
        ))

        fig.update_layout(
            title=f"Semantic Space: {data_type.replace('_', ' ').title()}",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            template="plotly_white",
            height=600
        )

        return fig

    def create_temporal_evolution_plot(self) -> go.Figure:
        """Create temporal evolution visualization showing how conversations progress over time."""
        logger.info("Creating temporal evolution plot")

        if not self.conversations:
            logger.error("No conversation data available")
            return go.Figure()

        # Extract temporal data
        temporal_data = []
        for conv in self.conversations:
            if 'learning_curve' in conv:
                lc = conv['learning_curve']
                temporal_data.append({
                    'timestamp': conv.get('timestamp', ''),
                    'progress': lc.get('progress', 0),
                    'stage': lc.get('stage', ''),
                    'complexity': lc.get('avg_complexity', 0),
                    'message_count': conv.get('message_count', 0)
                })

        if not temporal_data:
            logger.error("No temporal data available")
            return go.Figure()

        df = pd.DataFrame(temporal_data)

        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        if df.empty:
            logger.error("No valid timestamps found")
            return go.Figure()

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Learning Progress Over Time', 'Complexity Evolution', 'Message Count Distribution'),
            vertical_spacing=0.1
        )

        # Learning progress
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['progress'],
                mode='lines+markers',
                name='Learning Progress',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        # Complexity evolution
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['complexity'],
                mode='lines+markers',
                name='Complexity',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )

        # Message count distribution
        fig.add_trace(
            go.Histogram(
                x=df['message_count'],
                name='Message Count',
                nbinsx=20,
                marker_color='green'
            ),
            row=3, col=1
        )

        fig.update_layout(
            title="Temporal Evolution of Conversations",
            height=800,
            template="plotly_white",
            showlegend=False
        )

        return fig

    def create_network_graph(self, data_type: str = 'semantic_anchors',
                           min_connections: int = 2) -> go.Figure:
        """Create network graph visualization showing relationships between entities."""
        logger.info(f"Creating network graph for {data_type}")

        data = getattr(self, data_type, [])
        if not data:
            logger.error(f"No data available for {data_type}")
            return go.Figure()

        # Create graph
        G = nx.Graph()

        # Add nodes
        for item in data:
            if data_type == 'semantic_anchors':
                node_id = item.get('anchor_id', str(hash(item.get('text', ''))))
                G.add_node(node_id,
                          text=item.get('text', ''),
                          type=item.get('type', ''),
                          confidence=item.get('confidence', 0))
            elif data_type == 'entities':
                node_id = item.get('entity_id', str(hash(item.get('text', ''))))
                G.add_node(node_id,
                          text=item.get('text', ''),
                          label=item.get('label', ''),
                          description=item.get('description', ''))

        # Add edges based on similarity (simplified)
        nodes = list(G.nodes())
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # Simple similarity based on text overlap
                text1 = G.nodes[node1].get('text', '').lower()
                text2 = G.nodes[node2].get('text', '').lower()

                # Calculate Jaccard similarity
                words1 = set(text1.split())
                words2 = set(text2.split())

                if words1 and words2:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.3:  # Threshold for connection
                        G.add_edge(node1, node2, weight=similarity)

        # Filter nodes with minimum connections
        nodes_to_remove = [node for node in G.nodes() if G.degree(node) < min_connections]
        G.remove_nodes_from(nodes_to_remove)

        if len(G.nodes()) == 0:
            logger.warning("No nodes with sufficient connections")
            return go.Figure()

        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Create network visualization
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_colors = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            if data_type == 'semantic_anchors':
                node_text.append(f"Text: {G.nodes[node]['text'][:30]}...<br>Type: {G.nodes[node]['type']}")
                node_colors.append(G.nodes[node]['confidence'])
            else:
                node_text.append(f"Text: {G.nodes[node]['text']}<br>Label: {G.nodes[node]['label']}")
                node_colors.append(G.degree(node))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=node_colors,
                colorbar=dict(
                    thickness=15,
                    title='Node Score',
                    xanchor="left",
                    len=0.5
                ),
                line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Network Graph: {data_type.replace("_", " ").title()}',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )

        return fig

    def create_dashboard(self) -> go.Figure:
        """Create a comprehensive dashboard with multiple visualizations."""
        logger.info("Creating comprehensive dashboard")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Semantic Anchors Distribution',
                'Message Complexity by Speaker',
                'Emotion Distribution',
                'Domain Analysis'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # 1. Semantic anchors pie chart
        if self.semantic_anchors:
            anchor_types = {}
            for anchor in self.semantic_anchors:
                anchor_type = anchor.get('type', 'unknown')
                anchor_types[anchor_type] = anchor_types.get(anchor_type, 0) + 1

            if anchor_types:
                fig.add_trace(
                    go.Pie(
                        labels=list(anchor_types.keys()),
                        values=list(anchor_types.values()),
                        name="Semantic Anchors"
                    ),
                    row=1, col=1
                )

        # 2. Message complexity by speaker
        if self.messages:
            speaker_complexity = {}
            for msg in self.messages:
                speaker = msg.get('speaker', 'unknown')
                complexity = msg.get('complexity', {}).get('score', 0)
                if speaker not in speaker_complexity:
                    speaker_complexity[speaker] = []
                speaker_complexity[speaker].append(complexity)

            if speaker_complexity:
                speakers = list(speaker_complexity.keys())
                avg_complexities = [np.mean(speaker_complexity[s]) for s in speakers]

                fig.add_trace(
                    go.Bar(
                        x=speakers,
                        y=avg_complexities,
                        name="Avg Complexity"
                    ),
                    row=1, col=2
                )

        # 3. Emotion distribution
        if self.emotions:
            emotion_counts = {}
            for emotion in self.emotions:
                emotion_type = emotion.get('emotion', 'unknown')
                emotion_counts[emotion_type] = emotion_counts.get(emotion_type, 0) + 1

            if emotion_counts:
                fig.add_trace(
                    go.Bar(
                        x=list(emotion_counts.keys()),
                        y=list(emotion_counts.values()),
                        name="Emotions"
                    ),
                    row=2, col=1
                )

        # 4. Domain analysis scatter plot
        if self.domains:
            domain_data = {}
            for domain in self.domains:
                domain_name = domain.get('domain', 'unknown')
                complexity = domain.get('complexity_score', 0)
                if domain_name not in domain_data:
                    domain_data[domain_name] = []
                domain_data[domain_name].append(complexity)

            if domain_data:
                domain_names = list(domain_data.keys())
                avg_complexities = [np.mean(domain_data[d]) for d in domain_names]
                counts = [len(domain_data[d]) for d in domain_names]

                fig.add_trace(
                    go.Scatter(
                        x=avg_complexities,
                        y=counts,
                        mode='markers+text',
                        text=domain_names,
                        textposition="top center",
                        marker=dict(size=10),
                        name="Domains"
                    ),
                    row=2, col=2
                )

        fig.update_layout(
            title="Conversation Analysis Dashboard",
            height=800,
            template="plotly_white",
            showlegend=False
        )

        return fig

    def create_word_cloud(self, data_type: str = 'semantic_anchors') -> plt.Figure:
        """Create word cloud visualization."""
        logger.info(f"Creating word cloud for {data_type}")

        data = getattr(self, data_type, [])
        if not data:
            logger.error(f"No data available for {data_type}")
            return plt.figure()

        # Extract text and frequencies
        text_freq = {}
        for item in data:
            if data_type == 'semantic_anchors':
                text = item.get('text', '')
                freq = item.get('frequency', 1)
            elif data_type == 'messages':
                text = item.get('content', '')
                freq = 1
            else:
                text = item.get('text', '')
                freq = 1

            if text:
                text_freq[text] = text_freq.get(text, 0) + freq

        if not text_freq:
            logger.error("No text data available for word cloud")
            return plt.figure()

        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(text_freq)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud: {data_type.replace("_", " ").title()}')

        return fig

    def save_all_visualizations(self):
        """Generate and save all visualizations."""
        logger.info("Generating all visualizations...")

        # Create subdirectories
        (self.output_dir / 'interactive').mkdir(exist_ok=True)
        (self.output_dir / 'static').mkdir(exist_ok=True)

        # 1. Semantic space visualizations
        for data_type in ['semantic_anchors', 'messages', 'sentences']:
            try:
                fig = self.create_semantic_space_visualization(data_type)
                fig.write_html(self.output_dir / 'interactive' / f'semantic_space_{data_type}.html')
                logger.info(f"Saved semantic space visualization for {data_type}")
            except Exception as e:
                logger.error(f"Failed to create semantic space for {data_type}: {e}")

        # 2. Temporal evolution
        try:
            fig = self.create_temporal_evolution_plot()
            fig.write_html(self.output_dir / 'interactive' / 'temporal_evolution.html')
            logger.info("Saved temporal evolution plot")
        except Exception as e:
            logger.error(f"Failed to create temporal evolution: {e}")

        # 3. Network graphs
        for data_type in ['semantic_anchors', 'entities']:
            try:
                fig = self.create_network_graph(data_type)
                fig.write_html(self.output_dir / 'interactive' / f'network_graph_{data_type}.html')
                logger.info(f"Saved network graph for {data_type}")
            except Exception as e:
                logger.error(f"Failed to create network graph for {data_type}: {e}")

        # 4. Dashboard
        try:
            fig = self.create_dashboard()
            fig.write_html(self.output_dir / 'interactive' / 'dashboard.html')
            logger.info("Saved dashboard")
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")

        # 5. Word clouds
        for data_type in ['semantic_anchors', 'messages']:
            try:
                fig = self.create_word_cloud(data_type)
                fig.savefig(self.output_dir / 'static' / f'wordcloud_{data_type}.png',
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved word cloud for {data_type}")
            except Exception as e:
                logger.error(f"Failed to create word cloud for {data_type}: {e}")

        # 6. Create index HTML
        self.create_index_html()

        logger.info(f"All visualizations saved to {self.output_dir}")

    def create_index_html(self):
        """Create an index HTML file with links to all visualizations."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Conversation Analysis Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .section { margin: 20px 0; }
        .link { margin: 10px 0; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        h1 { color: #333; }
        h2 { color: #666; }
    </style>
</head>
<body>
    <h1>Conversation Analysis Visualizations</h1>
    <p>Generated on: {timestamp}</p>

    <div class="section">
        <h2>Interactive Visualizations</h2>
        <div class="link"><a href="interactive/dashboard.html">📊 Comprehensive Dashboard</a></div>
        <div class="link"><a href="interactive/temporal_evolution.html">📈 Temporal Evolution</a></div>
        <div class="link"><a href="interactive/semantic_space_semantic_anchors.html">🎯 Semantic Anchors Space</a></div>
        <div class="link"><a href="interactive/semantic_space_messages.html">💬 Messages Space</a></div>
        <div class="link"><a href="interactive/semantic_space_sentences.html">📝 Sentences Space</a></div>
        <div class="link"><a href="interactive/network_graph_semantic_anchors.html">🕸️ Semantic Anchors Network</a></div>
        <div class="link"><a href="interactive/network_graph_entities.html">🏷️ Entities Network</a></div>
    </div>

    <div class="section">
        <h2>Static Visualizations</h2>
        <div class="link"><a href="static/wordcloud_semantic_anchors.png">☁️ Semantic Anchors Word Cloud</a></div>
        <div class="link"><a href="static/wordcloud_messages.png">☁️ Messages Word Cloud</a></div>
    </div>

    <div class="section">
        <h2>Data Summary</h2>
        <p>Total Conversations: {conversation_count}</p>
        <p>Total Messages: {message_count}</p>
        <p>Total Sentences: {sentence_count}</p>
        <p>Total Semantic Anchors: {anchor_count}</p>
        <p>Total Entities: {entity_count}</p>
    </div>
</body>
</html>
        """.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            conversation_count=len(self.conversations),
            message_count=len(self.messages),
            sentence_count=len(self.sentences),
            anchor_count=len(self.semantic_anchors),
            entity_count=len(self.entities)
        )

        with open(self.output_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Created index HTML: {self.output_dir / 'index.html'}")

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Create visualizations for conversation data')
    parser.add_argument('--data-dir', required=True, help='Directory containing parsed data')
    parser.add_argument('--output-dir', default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--interactive-only', action='store_true', help='Generate only interactive visualizations')
    parser.add_argument('--static-only', action='store_true', help='Generate only static visualizations')

    args = parser.parse_args()

    # Initialize visualization layer
    viz_layer = SemanticVisualizationLayer(args.data_dir, args.output_dir)

    # Generate visualizations
    viz_layer.save_all_visualizations()

    print(f"✅ Visualizations saved to: {args.output_dir}")
    print(f"🌐 Open {args.output_dir}/index.html to view all visualizations")

if __name__ == "__main__":
    main()
