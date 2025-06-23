import chromadb
from chromadb.config import Settings
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import hashlib
import uuid
from sentence_transformers import SentenceTransformer
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Comprehensive ChromaDB manager for conversation data with new client format."""

    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB manager with new client format.

        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model to use for embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize ChromaDB client with new format
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {embedding_model}: {e}")
            self.embedding_model = None

        # Collection names for different data types
        self.collections = {
            'conversations': 'conversations',
            'messages': 'messages',
            'sentences': 'sentences',
            'concepts': 'concepts',
            'entities': 'entities',
            'semantic_anchors': 'semantic_anchors',
            'qa_pairs': 'qa_pairs',
            'code_blocks': 'code_blocks',
            'topics': 'topics',
            'emotions': 'emotions',
            'intents': 'intents',
            'domains': 'domains'
        }

        # Initialize collections
        self._initialize_collections()

    def _initialize_collections(self):
        """Initialize all ChromaDB collections with proper metadata."""
        for collection_name, collection_id in self.collections.items():
            try:
                # Get or create collection
                collection = self.client.get_or_create_collection(
                    name=collection_id,
                    metadata={
                        "description": f"Collection for {collection_name}",
                        "created_at": datetime.now().isoformat(),
                        "data_type": collection_name,
                        "embedding_model": self.embedding_model.name if self.embedding_model else "unknown"
                    }
                )
                logger.info(f"Initialized collection: {collection_id}")
            except Exception as e:
                logger.error(f"Failed to initialize collection {collection_id}: {e}")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using the sentence transformer model."""
        if not self.embedding_model or not text:
            return None

        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return None

    def add_conversations(self, conversations_data: List[Dict[str, Any]]):
        """Add conversation data to ChromaDB."""
        collection = self.client.get_collection(self.collections['conversations'])

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for conv in conversations_data:
            # Create document text from conversation info
            doc_text = f"Title: {conv.get('title', '')}\n"
            doc_text += f"Model: {conv.get('model', '')}\n"
            doc_text += f"Messages: {conv.get('message_count', 0)}\n"

            # Add learning curve info
            if 'learning_curve' in conv:
                lc = conv['learning_curve']
                doc_text += f"Learning Stage: {lc.get('stage', '')}\n"
                doc_text += f"Progress: {lc.get('progress', 0):.2f}\n"

            # Generate embedding
            embedding = self.get_embedding(doc_text)
            if not embedding:
                continue

            # Create unique ID
            conv_id = conv.get('conversation_id', str(uuid.uuid4()))

            documents.append(doc_text)
            metadatas.append({
                'conversation_id': conv_id,
                'title': conv.get('title', ''),
                'model': conv.get('model', ''),
                'message_count': conv.get('message_count', 0),
                'timestamp': conv.get('timestamp', ''),
                'learning_stage': conv.get('learning_curve', {}).get('stage', ''),
                'learning_progress': conv.get('learning_curve', {}).get('progress', 0),
                'data_type': 'conversation'
            })
            ids.append(conv_id)
            embeddings.append(embedding)

        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"Added {len(documents)} conversations to ChromaDB")

    def add_messages(self, messages_data: List[Dict[str, Any]]):
        """Add message data to ChromaDB."""
        collection = self.client.get_collection(self.collections['messages'])

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for msg in messages_data:
            content = msg.get('content', '')
            if not content:
                continue

            # Generate embedding
            embedding = self.get_embedding(content)
            if not embedding:
                continue

            # Create unique ID
            msg_id = msg.get('message_id', str(uuid.uuid4()))

            documents.append(content)
            metadatas.append({
                'message_id': msg_id,
                'conversation_id': msg.get('conversation_id', ''),
                'speaker': msg.get('speaker', ''),
                'timestamp': msg.get('timestamp', ''),
                'complexity_score': msg.get('complexity', {}).get('score', 0),
                'domains': ','.join(msg.get('domains', [])),
                'intents': ','.join(msg.get('intents', [])),
                'has_metaphor': msg.get('has_metaphor', False),
                'metaphor_type': msg.get('metaphor_type', ''),
                'data_type': 'message'
            })
            ids.append(msg_id)
            embeddings.append(embedding)

        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"Added {len(documents)} messages to ChromaDB")

    def add_sentences(self, sentences_data: List[Dict[str, Any]]):
        """Add sentence data to ChromaDB."""
        collection = self.client.get_collection(self.collections['sentences'])

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for sent in sentences_data:
            text = sent.get('text', '')
            if not text:
                continue

            # Generate embedding
            embedding = self.get_embedding(text)
            if not embedding:
                continue

            # Create unique ID
            sent_id = sent.get('sentence_id', str(uuid.uuid4()))

            documents.append(text)
            metadatas.append({
                'sentence_id': sent_id,
                'message_id': sent.get('message_id', ''),
                'conversation_id': sent.get('conversation_id', ''),
                'speaker': sent.get('speaker', ''),
                'polarity': sent.get('polarity', ''),
                'subjectivity': sent.get('subjectivity', 0),
                'complexity_score': sent.get('complexity', {}).get('score', 0),
                'entity_count': len(sent.get('entities', [])),
                'data_type': 'sentence'
            })
            ids.append(sent_id)
            embeddings.append(embedding)

        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"Added {len(documents)} sentences to ChromaDB")

    def add_semantic_anchors(self, anchors_data: List[Dict[str, Any]]):
        """Add semantic anchors to ChromaDB."""
        collection = self.client.get_collection(self.collections['semantic_anchors'])

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for anchor in anchors_data:
            text = anchor.get('text', '')
            if not text:
                continue

            # Generate embedding
            embedding = self.get_embedding(text)
            if not embedding:
                continue

            # Create unique ID
            anchor_id = anchor.get('anchor_id', str(uuid.uuid4()))

            documents.append(text)
            metadatas.append({
                'anchor_id': anchor_id,
                'conversation_id': anchor.get('conversation_id', ''),
                'message_id': anchor.get('message_id', ''),
                'anchor_type': anchor.get('type', ''),
                'confidence': anchor.get('confidence', 0),
                'frequency': anchor.get('frequency', 1),
                'cluster_id': anchor.get('cluster_id', ''),
                'is_high_value': anchor.get('is_high_value', False),
                'data_type': 'semantic_anchor'
            })
            ids.append(anchor_id)
            embeddings.append(embedding)

        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"Added {len(documents)} semantic anchors to ChromaDB")

    def add_qa_pairs(self, qa_data: List[Dict[str, Any]]):
        """Add Q&A pairs to ChromaDB."""
        collection = self.client.get_collection(self.collections['qa_pairs'])

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for qa in qa_data:
            question = qa.get('question', '')
            answer = qa.get('answer', '')

            if not question or not answer:
                continue

            # Combine question and answer for embedding
            combined_text = f"Q: {question}\nA: {answer}"

            # Generate embedding
            embedding = self.get_embedding(combined_text)
            if not embedding:
                continue

            # Create unique ID
            qa_id = qa.get('qa_id', str(uuid.uuid4()))

            documents.append(combined_text)
            metadatas.append({
                'qa_id': qa_id,
                'conversation_id': qa.get('conversation_id', ''),
                'question': question,
                'answer': answer,
                'model': qa.get('model', ''),
                'timestamp': qa.get('timestamp', ''),
                'complexity_score': qa.get('complexity_score', 0),
                'domains': ','.join(qa.get('domains', [])),
                'data_type': 'qa_pair'
            })
            ids.append(qa_id)
            embeddings.append(embedding)

        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"Added {len(documents)} Q&A pairs to ChromaDB")

    def search_similar(self, query: str, collection_name: str = 'messages', n_results: int = 10,
                      filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content in ChromaDB.

        Args:
            query: Search query text
            collection_name: Name of collection to search
            n_results: Number of results to return
            filter_dict: Optional filter criteria

        Returns:
            List of similar documents with metadata
        """
        try:
            collection = self.client.get_collection(self.collections[collection_name])

            # Generate query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []

            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}

        for collection_name, collection_id in self.collections.items():
            try:
                collection = self.client.get_collection(collection_id)
                count = collection.count()
                stats[collection_name] = {
                    'count': count,
                    'collection_id': collection_id
                }
            except Exception as e:
                logger.error(f"Failed to get stats for {collection_name}: {e}")
                stats[collection_name] = {'count': 0, 'error': str(e)}

        return stats

    def export_collection_to_csv(self, collection_name: str, output_path: str):
        """Export a collection to CSV format."""
        try:
            collection = self.client.get_collection(self.collections[collection_name])

            # Get all data
            results = collection.get()

            if not results['ids']:
                logger.warning(f"Collection {collection_name} is empty")
                return

            # Create DataFrame
            df_data = []
            for i in range(len(results['ids'])):
                row = {
                    'id': results['ids'][i],
                    'document': results['documents'][i]
                }
                # Add metadata
                if results['metadatas']:
                    row.update(results['metadatas'][i])

                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df_data)} records from {collection_name} to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export collection {collection_name}: {e}")

    def reset_collection(self, collection_name: str):
        """Reset/clear a specific collection."""
        try:
            collection = self.client.get_collection(self.collections[collection_name])
            collection.delete()
            logger.info(f"Reset collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to reset collection {collection_name}: {e}")

    def reset_all_collections(self):
        """Reset all collections."""
        for collection_name in self.collections.keys():
            self.reset_collection(collection_name)

def load_parsed_data(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load parsed data from JSON files."""
    data_dir = Path(data_dir)
    data = {}

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return data

    # Find all JSON files
    json_files = list(data_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)

            # Extract data type from filename
            data_type = json_file.stem.split('_')[0]
            data[data_type] = file_data

            logger.info(f"Loaded {len(file_data)} {data_type} from {json_file}")

        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")

    return data

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='ChromaDB migration and management')
    parser.add_argument('--data-dir', required=True, help='Directory containing parsed data')
    parser.add_argument('--chroma-dir', default='./chroma_db', help='ChromaDB persistence directory')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', help='Sentence transformer model')
    parser.add_argument('--reset', action='store_true', help='Reset all collections before migration')
    parser.add_argument('--export-csv', help='Export collections to CSV directory')

    args = parser.parse_args()

    # Initialize ChromaDB manager
    chroma_manager = ChromaDBManager(args.chroma_dir, args.embedding_model)

    # Reset collections if requested
    if args.reset:
        logger.info("Resetting all collections...")
        chroma_manager.reset_all_collections()
        chroma_manager._initialize_collections()

    # Load parsed data
    data = load_parsed_data(args.data_dir)

    # Migrate data to ChromaDB
    if 'conversations' in data:
        chroma_manager.add_conversations(data['conversations'])

    if 'messages' in data:
        chroma_manager.add_messages(data['messages'])

    if 'sentences' in data:
        chroma_manager.add_sentences(data['sentences'])

    if 'semantic_anchors' in data:
        chroma_manager.add_semantic_anchors(data['semantic_anchors'])

    if 'qa_pairs' in data:
        chroma_manager.add_qa_pairs(data['qa_pairs'])

    # Print statistics
    stats = chroma_manager.get_collection_stats()
    logger.info("Collection statistics:")
    for collection_name, stat in stats.items():
        logger.info(f"  {collection_name}: {stat['count']} records")

    # Export to CSV if requested
    if args.export_csv:
        export_dir = Path(args.export_csv)
        export_dir.mkdir(exist_ok=True)

        for collection_name in chroma_manager.collections.keys():
            output_path = export_dir / f"{collection_name}.csv"
            chroma_manager.export_collection_to_csv(collection_name, str(output_path))

if __name__ == "__main__":
    main()
