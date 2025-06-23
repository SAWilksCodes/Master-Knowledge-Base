#!/usr/bin/env python3
"""
Enhanced Sentence and Word ID Generation Script
Generates canonical sentence IDs and word IDs across all conversations with deduplication.
Uses NLTK for advanced linguistic analysis including POS tagging, sentiment analysis, and lemmatization.
"""

import json
import re
import hashlib
import csv
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import logging

# NLTK imports
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk

# Download required NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLTK components
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def normalize_text(text):
    """Normalize text for consistent processing."""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Basic punctuation normalization
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\"\']', '', text)

    return text.strip()

def split_sentences(text):
    """Split text into sentences using NLTK's sentence tokenizer."""
    if not text:
        return []

    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except:
        # Fallback to regex if NLTK fails
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def extract_words(text):
    """Extract words from text using NLTK's word tokenizer."""
    if not text:
        return []

    try:
        words = word_tokenize(text.lower())
        return [word for word in words if word.isalnum()]
    except:
        # Fallback to regex if NLTK fails
        words = re.findall(r'\b\w+\b', text.lower())
        return words

def generate_sentence_id(sentence):
    """Generate a unique ID for a sentence."""
    normalized = normalize_text(sentence)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]

def generate_word_id(word):
    """Generate a unique ID for a word."""
    normalized = word.lower().strip()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:8]

def analyze_pos(text):
    """Analyze part-of-speech tags for text."""
    try:
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        return pos_tags
    except:
        return []

def analyze_sentiment_nltk(text):
    """Analyze sentiment using NLTK's VADER sentiment analyzer."""
    try:
        scores = sia.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    except:
        return 'neutral'

def extract_entities(text):
    """Extract named entities from text."""
    try:
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        entities = ne_chunk(pos_tags)

        entity_list = []
        for subtree in entities:
            if hasattr(subtree, 'label'):
                entity_type = subtree.label()
                entity_text = ' '.join([leaf[0] for leaf in subtree.leaves()])
                entity_list.append({
                    'text': entity_text,
                    'type': entity_type
                })

        return entity_list
    except:
        return []

def get_word_lemma(word):
    """Get the lemmatized form of a word."""
    try:
        return lemmatizer.lemmatize(word.lower())
    except:
        return word.lower()

def detect_question(sentence):
    """Detect if sentence is a question using NLTK and patterns."""
    # Check for question mark
    if sentence.strip().endswith('?'):
        return True

    # Check for question words at beginning
    question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']
    words = word_tokenize(sentence.lower())
    if words and words[0] in question_words:
        return True

    # Check for inverted questions
    inverted_patterns = [
        r'\b(is|are|was|were|do|does|did|can|could|will|would|should|may|might)\s+\w+',
        r'\b(do|does|did)\s+you\b',
        r'\b(can|could|will|would|should|may|might)\s+you\b'
    ]

    for pattern in inverted_patterns:
        if re.search(pattern, sentence.lower()):
            return True

    return False

def detect_metaphor(sentence):
    """Detect potential metaphors using keyword patterns."""
    metaphor_keywords = [
        'like', 'as', 'metaphor', 'symbol', 'represents', 'means', 'is like',
        'similar to', 'reminds me of', 'feels like', 'looks like', 'sounds like',
        'imagine', 'picture', 'think of', 'compare to'
    ]

    sentence_lower = sentence.lower()
    return any(keyword in sentence_lower for keyword in metaphor_keywords)

def process_conversations():
    """Process all conversation files and generate sentence/word IDs."""
    logger.info("Starting enhanced sentence and word ID generation...")

    # Paths
    split_dir = Path("split_chats")
    output_dir = Path("id_generation_results")
    output_dir.mkdir(exist_ok=True)

    # Storage for deduplication
    sentence_registry = {}  # sentence_hash -> sentence_data
    word_registry = {}      # word_hash -> word_data
    sentence_occurrences = defaultdict(list)  # sentence_hash -> list of occurrences
    word_occurrences = defaultdict(list)      # word_hash -> list of occurrences

    # Statistics
    total_files = 0
    total_sentences = 0
    unique_sentences = 0
    total_words = 0
    unique_words = 0

    # Process each conversation file
    for json_file in split_dir.glob("*.json"):
        total_files += 1
        logger.info(f"Processing {json_file.name} ({total_files})")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract messages from the correct structure
            messages = []
            if 'mapping' in data:
                for msg_id, msg_data in data['mapping'].items():
                    if 'message' in msg_data and msg_data['message']:
                        content = msg_data['message'].get('content', [])
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and 'text' in part:
                                    messages.append(part['text'])
                        elif isinstance(content, str):
                            messages.append(content)

            # Process each message
            for msg_idx, message in enumerate(messages):
                if not message:
                    continue

                # Split into sentences
                sentences = split_sentences(message)

                for sent_idx, sentence in enumerate(sentences):
                    if len(sentence) < 3:  # Skip very short sentences
                        continue

                    total_sentences += 1
                    sentence_hash = generate_sentence_id(sentence)

                    # Record occurrence
                    occurrence = {
                        'file': json_file.name,
                        'message_index': msg_idx,
                        'sentence_index': sent_idx,
                        'original_text': sentence
                    }
                    sentence_occurrences[sentence_hash].append(occurrence)

                    # Register sentence if new
                    if sentence_hash not in sentence_registry:
                        unique_sentences += 1

                        # Enhanced analysis
                        pos_tags = analyze_pos(sentence)
                        sentiment = analyze_sentiment_nltk(sentence)
                        entities = extract_entities(sentence)

                        sentence_registry[sentence_hash] = {
                            'id': sentence_hash,
                            'text': sentence,
                            'normalized_text': normalize_text(sentence),
                            'length': len(sentence),
                            'word_count': len(sentence.split()),
                            'is_question': detect_question(sentence),
                            'has_metaphor': detect_metaphor(sentence),
                            'sentiment': sentiment,
                            'pos_tags': pos_tags,
                            'entities': entities,
                            'occurrence_count': 1
                        }
                    else:
                        sentence_registry[sentence_hash]['occurrence_count'] += 1

                    # Process words in sentence
                    words = extract_words(sentence)
                    for word in words:
                        if len(word) < 2:  # Skip very short words
                            continue

                        total_words += 1
                        word_hash = generate_word_id(word)

                        # Record word occurrence
                        word_occurrence = {
                            'file': json_file.name,
                            'message_index': msg_idx,
                            'sentence_index': sent_idx,
                            'sentence_id': sentence_hash,
                            'word': word
                        }
                        word_occurrences[word_hash].append(word_occurrence)

                        # Register word if new
                        if word_hash not in word_registry:
                            unique_words += 1

                            # Enhanced word analysis
                            lemma = get_word_lemma(word)
                            is_stopword = word.lower() in stop_words

                            word_registry[word_hash] = {
                                'id': word_hash,
                                'word': word,
                                'lemma': lemma,
                                'length': len(word),
                                'is_stopword': is_stopword,
                                'occurrence_count': 1
                            }
                        else:
                            word_registry[word_hash]['occurrence_count'] += 1

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue

    # Calculate frequency ranks
    logger.info("Calculating frequency ranks...")

    # Sort sentences by occurrence count
    sorted_sentences = sorted(sentence_registry.values(),
                            key=lambda x: x['occurrence_count'], reverse=True)
    for rank, sentence in enumerate(sorted_sentences, 1):
        sentence['frequency_rank'] = rank

    # Sort words by occurrence count
    sorted_words = sorted(word_registry.values(),
                         key=lambda x: x['occurrence_count'], reverse=True)
    for rank, word in enumerate(sorted_words, 1):
        word['frequency_rank'] = rank

    # Save results
    logger.info("Saving results...")

    # Save sentence registry
    with open(output_dir / "sentences_enhanced.json", 'w', encoding='utf-8') as f:
        json.dump(list(sentence_registry.values()), f, indent=2, ensure_ascii=False)

    # Save word registry
    with open(output_dir / "words_enhanced.json", 'w', encoding='utf-8') as f:
        json.dump(list(word_registry.values()), f, indent=2, ensure_ascii=False)

    # Save sentence occurrences
    with open(output_dir / "sentence_occurrences_enhanced.json", 'w', encoding='utf-8') as f:
        json.dump(dict(sentence_occurrences), f, indent=2, ensure_ascii=False)

    # Save word occurrences
    with open(output_dir / "word_occurrences_enhanced.json", 'w', encoding='utf-8') as f:
        json.dump(dict(word_occurrences), f, indent=2, ensure_ascii=False)

    # Create summary report
    summary = {
        'processing_date': datetime.now().isoformat(),
        'total_files_processed': total_files,
        'total_sentences_found': total_sentences,
        'unique_sentences': unique_sentences,
        'total_words_found': total_words,
        'unique_words': unique_words,
        'deduplication_ratio_sentences': unique_sentences / total_sentences if total_sentences > 0 else 0,
        'deduplication_ratio_words': unique_words / total_words if total_words > 0 else 0,
        'top_10_sentences': sorted_sentences[:10],
        'top_10_words': sorted_words[:10]
    }

    with open(output_dir / "sentence_word_analysis_report_enhanced.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Create CSV exports
    with open(output_dir / "sentences_enhanced.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'length', 'word_count', 'is_question',
                                              'has_metaphor', 'sentiment', 'occurrence_count', 'frequency_rank'])
        writer.writeheader()
        writer.writerows(sentence_registry.values())

    with open(output_dir / "words_enhanced.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'word', 'lemma', 'length', 'is_stopword',
                                              'occurrence_count', 'frequency_rank'])
        writer.writeheader()
        writer.writerows(word_registry.values())

    logger.info(f"Processing complete!")
    logger.info(f"Files processed: {total_files}")
    logger.info(f"Total sentences: {total_sentences}")
    logger.info(f"Unique sentences: {unique_sentences}")
    logger.info(f"Total words: {total_words}")
    logger.info(f"Unique words: {unique_words}")
    logger.info(f"Results saved to: {output_dir}")

    return summary

if __name__ == "__main__":
    process_conversations()
