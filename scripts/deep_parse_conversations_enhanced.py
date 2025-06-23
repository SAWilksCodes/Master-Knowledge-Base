import json
import spacy
from textblob import TextBlob
import re
from pathlib import Path
from collections import Counter, defaultdict
import uuid
from datetime import datetime
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model - you may need to install: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class EnhancedConversationParser:
    """Multi-layer parser for conversation analysis with ID generation and graph preparation."""

    def __init__(self, output_dir: str = "parsed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Data storage
        self.conversations = []
        self.messages = []
        self.sentences = []
        self.words = []
        self.concepts = []
        self.metaphors = []
        self.emotions = []
        self.entities = []
        self.intents = []
        self.complexity_scores = []
        self.domains = []
        self.sequences = []
        self.evolutions = []
        self.milestones = []
        self.learning_curves = []
        self.threads = []
        self.clusters = []
        self.dependencies = []
        self.synonyms = []
        self.references = []
        self.applications = []
        self.reasoning_patterns = []
        self.creativity_indicators = []
        self.problem_solving = []
        self.decisions = []
        self.nodes = []
        self.edges = []
        self.paths = []
        self.branches = []
        self.tools = []
        self.platforms = []
        self.resources = []
        self.constraints = []
        self.breakthroughs = []
        self.failures = []
        self.optimizations = []
        self.validations = []

        # Statistics
        self.stats = defaultdict(Counter)

        # ID generation helpers
        self.id_counter = 0

        # Emotion patterns based on Plutchik's wheel
        self.emotion_patterns = {
            'joy': ['happy', 'glad', 'cheerful', 'delighted', 'excited', 'love', 'wonderful', 'great', 'amazing'],
            'trust': ['trust', 'believe', 'faith', 'confident', 'reliable', 'sure', 'certain', 'secure'],
            'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified', 'concerned', 'doubt'],
            'surprise': ['surprised', 'amazed', 'astonished', 'unexpected', 'shock', 'wow', 'incredible'],
            'sadness': ['sad', 'depressed', 'miserable', 'unhappy', 'sorry', 'grief', 'disappointed'],
            'disgust': ['disgusted', 'hate', 'awful', 'terrible', 'gross', 'revolting', 'horrible'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'rage', 'irritated'],
            'anticipation': ['expect', 'hope', 'look forward', 'await', 'anticipate', 'plan', 'prepare']
        }

        # Intent patterns
        self.intent_patterns = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which', '?'],
            'request': ['can you', 'please', 'help', 'need', 'want', 'would you'],
            'clarification': ['clarify', 'explain', 'elaborate', 'what do you mean', 'i don\'t understand'],
            'confirmation': ['is that correct', 'are you sure', 'confirm', 'verify'],
            'suggestion': ['maybe', 'perhaps', 'consider', 'suggest', 'recommend'],
            'agreement': ['yes', 'agree', 'correct', 'right', 'exactly'],
            'disagreement': ['no', 'disagree', 'wrong', 'incorrect', 'not really'],
            'gratitude': ['thank', 'thanks', 'appreciate', 'grateful'],
            'apology': ['sorry', 'apologize', 'excuse', 'regret']
        }

        # Complexity indicators
        self.complexity_indicators = {
            'technical_terms': ['algorithm', 'api', 'database', 'framework', 'protocol', 'architecture'],
            'abstract_concepts': ['paradigm', 'philosophy', 'theory', 'concept', 'principle'],
            'conditional_logic': ['if', 'else', 'when', 'unless', 'provided that'],
            'causal_relationships': ['because', 'therefore', 'thus', 'hence', 'as a result'],
            'comparative_analysis': ['compared to', 'versus', 'in contrast', 'however', 'nevertheless']
        }

        # Domain classifiers
        self.domain_patterns = {
            'programming': ['code', 'function', 'class', 'variable', 'loop', 'debug', 'compile'],
            'ai_ml': ['model', 'training', 'neural', 'algorithm', 'prediction', 'learning'],
            'business': ['strategy', 'market', 'revenue', 'profit', 'customer', 'product'],
            'creative': ['design', 'art', 'creative', 'imagination', 'inspiration', 'aesthetic'],
            'research': ['study', 'analysis', 'experiment', 'hypothesis', 'methodology'],
            'personal': ['family', 'friend', 'relationship', 'personal', 'life', 'experience'],
            'technical': ['system', 'technology', 'hardware', 'software', 'network', 'security']
        }

    def generate_id(self, prefix: str, content: str = None) -> str:
        """Generate a unique ID with optional content-based hash."""
        self.id_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        if content:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            return f"{prefix}_{timestamp}_{content_hash}_{self.id_counter:06d}"
        else:
            return f"{prefix}_{timestamp}_{self.id_counter:06d}"

    def analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity using multiple metrics."""
        if not nlp:
            return {'score': 0.5, 'factors': ['spacy_not_available']}

        doc = nlp(text)

        # Basic metrics
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(list(doc.sents))
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Vocabulary complexity
        unique_words = len(set([token.lemma_.lower() for token in doc if not token.is_punct]))
        lexical_diversity = unique_words / max(word_count, 1)

        # Technical complexity
        technical_terms = sum(1 for token in doc if token.text.lower() in
                            [term for terms in self.complexity_indicators.values() for term in terms])

        # Syntactic complexity
        complex_sentences = sum(1 for sent in doc.sents if len([token for token in sent if token.dep_ in ['nsubj', 'dobj']]) > 2)

        # Calculate overall complexity score (0-1)
        complexity_score = min(1.0, (
            (avg_sentence_length / 20) * 0.3 +
            (lexical_diversity * 2) * 0.2 +
            (technical_terms / max(word_count, 1) * 10) * 0.3 +
            (complex_sentences / max(sentence_count, 1)) * 0.2
        ))

        return {
            'score': complexity_score,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'lexical_diversity': lexical_diversity,
            'technical_terms': technical_terms,
            'complex_sentences': complex_sentences,
            'factors': []
        }

    def detect_domain(self, text: str) -> List[str]:
        """Detect domains present in the text."""
        text_lower = text.lower()
        detected_domains = []

        for domain, patterns in self.domain_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_domains.append(domain)

        return detected_domains if detected_domains else ['general']

    def detect_intent(self, text: str) -> List[str]:
        """Detect user intent from text."""
        text_lower = text.lower()
        detected_intents = []

        for intent, patterns in self.intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_intents.append(intent)

        return detected_intents if detected_intents else ['statement']

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities and their relationships."""
        if not nlp:
            return []

        doc = nlp(text)
        entities = []

        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            }
            entities.append(entity_data)

        return entities

    def detect_metaphor_analogy(self, text: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Enhanced metaphor and analogy detection."""
        if not nlp:
            return False, None, {}

        doc = nlp(text)
        text_lower = text.lower()

        # Explicit analogy indicators
        analogy_indicators = ['like', 'as', 'similar to', 'resembles', 'as if', 'as though', 'just as']
        metaphor_verbs = ['is', 'are', 'was', 'were', 'become', 'became', 'represents']

        # Check for explicit analogies
        for indicator in analogy_indicators:
            if indicator in text_lower:
                return True, 'analogy', {'indicator': indicator, 'confidence': 0.8}

        # Check for implicit metaphors (X is Y pattern)
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in metaphor_verbs:
                subj = None
                obj = None

                for child in token.children:
                    if child.dep_ == "nsubj":
                        subj = child
                    elif child.dep_ in ["attr", "dobj"]:
                        obj = child

                if subj and obj:
                    # Check if they're different entity types
                    if subj.ent_type_ != obj.ent_type_ and obj.pos_ == "NOUN":
                        return True, 'metaphor', {
                            'subject': subj.text,
                            'object': obj.text,
                            'verb': token.text,
                            'confidence': 0.6
                        }

        return False, None, {}

    def analyze_learning_curve(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze learning progression throughout conversation."""
        if len(conversation_history) < 3:
            return {'stage': 'beginning', 'progress': 0.0, 'indicators': []}

        # Analyze complexity progression
        complexities = []
        for msg in conversation_history:
            if msg.get('content'):
                complexity = self.analyze_complexity(msg['content'])
                complexities.append(complexity['score'])

        if not complexities:
            return {'stage': 'beginning', 'progress': 0.0, 'indicators': []}

        # Calculate learning indicators
        avg_complexity = np.mean(complexities)
        complexity_trend = np.polyfit(range(len(complexities)), complexities, 1)[0]

        # Determine learning stage
        if avg_complexity < 0.3:
            stage = 'beginning'
        elif avg_complexity < 0.6:
            stage = 'intermediate'
        else:
            stage = 'advanced'

        # Calculate progress (0-1)
        progress = min(1.0, avg_complexity * 1.5)

        return {
            'stage': stage,
            'progress': progress,
            'avg_complexity': avg_complexity,
            'complexity_trend': complexity_trend,
            'indicators': ['complexity_increase' if complexity_trend > 0.01 else 'complexity_stable']
        }

    def process_conversation(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single conversation with full analysis."""
        conversation_id = self.generate_id('conv', conversation_data.get('title', ''))

        # Extract basic conversation info
        conversation_info = {
            'conversation_id': conversation_id,
            'title': conversation_data.get('title', ''),
            'model': conversation_data.get('model', 'unknown'),
            'timestamp': conversation_data.get('timestamp', ''),
            'message_count': len(conversation_data.get('messages', [])),
            'total_tokens': conversation_data.get('total_tokens', 0)
        }

        # Process messages
        messages = conversation_data.get('messages', [])
        processed_messages = []
        conversation_history = []

        for msg_idx, message in enumerate(messages):
            if not message.get('content'):
                continue

            message_id = self.generate_id('msg', message['content'])
            speaker = message.get('role', 'unknown')
            content = message['content']

            # Add to conversation history for learning curve analysis
            conversation_history.append({'content': content, 'role': speaker})

            # Analyze message
            complexity = self.analyze_complexity(content)
            domains = self.detect_domain(content)
            intents = self.detect_intent(content) if speaker == 'user' else ['response']
            entities = self.extract_entities(content)
            has_metaphor, metaphor_type, metaphor_data = self.detect_metaphor_analogy(content)

            # Process with spaCy for detailed analysis
            if nlp:
                doc = nlp(content)

                # Extract sentences
                for sent_idx, sent in enumerate(doc.sents):
                    sentence_id = self.generate_id('sent', sent.text)

                    # Analyze sentence
                    blob = TextBlob(sent.text)
                    polarity = 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'

                    sentence_data = {
                        'sentence_id': sentence_id,
                        'message_id': message_id,
                        'conversation_id': conversation_id,
                        'speaker': speaker,
                        'text': sent.text.strip(),
                        'polarity': polarity,
                        'subjectivity': blob.sentiment.subjectivity,
                        'complexity': self.analyze_complexity(sent.text),
                        'entities': self.extract_entities(sent.text)
                    }
                    self.sentences.append(sentence_data)

                    # Extract words with detailed analysis
                    for token in sent:
                        if not token.is_punct and not token.is_space:
                            word_id = self.generate_id('word', token.text)
                            word_data = {
                                'word_id': word_id,
                                'sentence_id': sentence_id,
                                'message_id': message_id,
                                'conversation_id': conversation_id,
                                'text': token.text,
                                'lemma': token.lemma_,
                                'pos': token.pos_,
                                'tag': token.tag_,
                                'dep': token.dep_,
                                'is_stop': token.is_stop,
                                'is_alpha': token.is_alpha,
                                'is_digit': token.is_digit
                            }
                            self.words.append(word_data)

            # Create message record
            message_data = {
                'message_id': message_id,
                'conversation_id': conversation_id,
                'speaker': speaker,
                'content': content,
                'timestamp': message.get('timestamp', ''),
                'complexity': complexity,
                'domains': domains,
                'intents': intents,
                'entities': entities,
                'has_metaphor': has_metaphor,
                'metaphor_type': metaphor_type,
                'metaphor_data': metaphor_data
            }

            processed_messages.append(message_data)
            self.messages.append(message_data)

        # Analyze learning curve
        learning_analysis = self.analyze_learning_curve(conversation_history)

        # Create conversation record
        conversation_record = {
            **conversation_info,
            'learning_curve': learning_analysis,
            'processed_messages': processed_messages
        }

        self.conversations.append(conversation_record)

        return conversation_record

    def save_results(self, output_dir: Optional[str] = None):
        """Save all parsed data to JSON files."""
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save main data structures
        data_files = {
            'conversations': self.conversations,
            'messages': self.messages,
            'sentences': self.sentences,
            'words': self.words,
            'concepts': self.concepts,
            'metaphors': self.metaphors,
            'emotions': self.emotions,
            'entities': self.entities,
            'intents': self.intents,
            'complexity_scores': self.complexity_scores,
            'domains': self.domains,
            'sequences': self.sequences,
            'evolutions': self.evolutions,
            'milestones': self.milestones,
            'learning_curves': self.learning_curves,
            'threads': self.threads,
            'clusters': self.clusters,
            'dependencies': self.dependencies,
            'synonyms': self.synonyms,
            'references': self.references,
            'applications': self.applications,
            'reasoning_patterns': self.reasoning_patterns,
            'creativity_indicators': self.creativity_indicators,
            'problem_solving': self.problem_solving,
            'decisions': self.decisions,
            'nodes': self.nodes,
            'edges': self.edges,
            'paths': self.paths,
            'branches': self.branches,
            'tools': self.tools,
            'platforms': self.platforms,
            'resources': self.resources,
            'constraints': self.constraints,
            'breakthroughs': self.breakthroughs,
            'failures': self.failures,
            'optimizations': self.optimizations,
            'validations': self.validations
        }

        for name, data in data_files.items():
            if data:  # Only save non-empty data
                output_file = self.output_dir / f"{name}_{timestamp}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"Saved {len(data)} {name} to {output_file}")

        # Save summary statistics
        stats_file = self.output_dir / f"parsing_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.stats), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved parsing statistics to {stats_file}")

    def process_file(self, input_path: str):
        """Process a single conversation file."""
        input_path = Path(input_path)

        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return

        logger.info(f"Processing file: {input_path}")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)

            result = self.process_conversation(conversation_data)
            logger.info(f"Processed conversation: {result['title']} ({result['message_count']} messages)")

        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")

    def process_directory(self, input_dir: str, pattern: str = "*.json"):
        """Process all conversation files in a directory."""
        input_dir = Path(input_dir)

        if not input_dir.exists():
            logger.error(f"Directory not found: {input_dir}")
            return

        files = list(input_dir.glob(pattern))
        logger.info(f"Found {len(files)} files to process")

        for file_path in files:
            self.process_file(str(file_path))

        # Save all results
        self.save_results()

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced conversation parser with ID generation')
    parser.add_argument('-i', '--input', required=True, help='Input file or directory')
    parser.add_argument('-o', '--output', default='parsed_data', help='Output directory')
    parser.add_argument('-p', '--pattern', default='*.json', help='File pattern for directory processing')

    args = parser.parse_args()

    parser = EnhancedConversationParser(args.output)

    input_path = Path(args.input)

    if input_path.is_file():
        parser.process_file(str(input_path))
    elif input_path.is_dir():
        parser.process_directory(str(input_path), args.pattern)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return

    parser.save_results()

if __name__ == "__main__":
    main()
