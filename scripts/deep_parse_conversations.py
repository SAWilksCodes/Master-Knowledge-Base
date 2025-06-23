import json
import spacy
from textblob import TextBlob
import re
from pathlib import Path
from collections import Counter, defaultdict
import uuid
from datetime import datetime

# Load spaCy model - you may need to install: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

class ConversationDeepParser:
    """Multi-layer parser for conversation analysis."""

    def __init__(self):
        self.sentences_data = []
        self.words_data = []
        self.concepts_data = []
        self.metaphors_data = []
        self.stats = defaultdict(Counter)

        # Emotion patterns based on Plutchik's wheel
        self.emotion_patterns = {
            'joy': ['happy', 'glad', 'cheerful', 'delighted', 'excited', 'love', 'wonderful'],
            'trust': ['trust', 'believe', 'faith', 'confident', 'reliable', 'sure'],
            'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified'],
            'surprise': ['surprised', 'amazed', 'astonished', 'unexpected', 'shock'],
            'sadness': ['sad', 'depressed', 'miserable', 'unhappy', 'sorry', 'grief'],
            'disgust': ['disgusted', 'hate', 'awful', 'terrible', 'gross', 'revolting'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'rage'],
            'anticipation': ['expect', 'hope', 'look forward', 'await', 'anticipate', 'plan']
        }

        # Metaphor/analogy indicators
        self.metaphor_indicators = ['like', 'as', 'similar to', 'resembles', 'as if', 'as though']
        self.metaphor_verbs = ['is', 'are', 'was', 'were', 'become', 'became']

    def detect_emotion(self, text):
        """Detect emotion in text using patterns and sentiment."""
        text_lower = text.lower()

        # Check emotion patterns
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return emotion

        # Fallback to sentiment-based emotion
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.5:
            return 'joy'
        elif polarity > 0.1:
            return 'trust'
        elif polarity < -0.5:
            return 'anger'
        elif polarity < -0.1:
            return 'sadness'
        else:
            return 'neutral'

    def detect_tense(self, doc):
        """Detect the primary tense of a sentence."""
        verb_tenses = []

        for token in doc:
            if token.pos_ == "VERB":
                if token.tag_ in ["VBD", "VBN"]:
                    verb_tenses.append("past")
                elif token.tag_ in ["VBG", "VBZ", "VBP", "VB"]:
                    verb_tenses.append("present")
                elif "will" in [t.text.lower() for t in doc] or "going to" in doc.text.lower():
                    verb_tenses.append("future")

        if not verb_tenses:
            return "present"  # Default

        # Return most common tense
        return Counter(verb_tenses).most_common(1)[0][0]

    def detect_metaphor_analogy(self, text):
        """Detect if sentence contains metaphor or analogy."""
        text_lower = text.lower()

        # Check for explicit analogies
        for indicator in self.metaphor_indicators:
            if indicator in text_lower:
                return True, 'analogy'

        # Check for implicit metaphors (X is Y pattern)
        doc = nlp(text)
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in self.metaphor_verbs:
                # Check if subject and object are different types of entities
                subj = None
                obj = None
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subj = child
                    elif child.dep_ in ["attr", "dobj"]:
                        obj = child

                if subj and obj:
                    # Simple heuristic: if they're different entity types, might be metaphor
                    if subj.ent_type_ != obj.ent_type_ and obj.pos_ == "NOUN":
                        return True, 'metaphor'

        return False, None

    def extract_grammatical_role(self, token):
        """Extract grammatical role of a token."""
        dep_to_role = {
            'nsubj': 'subject',
            'dobj': 'direct_object',
            'iobj': 'indirect_object',
            'pobj': 'object_of_preposition',
            'attr': 'attribute',
            'ROOT': 'root_verb',
            'aux': 'auxiliary',
            'amod': 'adjective_modifier',
            'advmod': 'adverb_modifier'
        }
        return dep_to_role.get(token.dep_, token.dep_)

    def process_message_pair(self, pair, pair_index):
        """Process a single user-assistant message pair."""
        for speaker, message in [('user', pair['user']), ('assistant', pair['assistant'])]:
            if not message:
                continue

            # Process with spaCy
            doc = nlp(message)

            # Extract sentences
            for sent_idx, sent in enumerate(doc.sents):
                sentence_id = f"{pair_index}_{speaker}_{sent_idx}"

                # Analyze sentence
                blob = TextBlob(sent.text)
                polarity_score = blob.sentiment.polarity

                if polarity_score > 0.1:
                    polarity = 'positive'
                elif polarity_score < -0.1:
                    polarity = 'negative'
                else:
                    polarity = 'neutral'

                emotion = self.detect_emotion(sent.text)
                tense = self.detect_tense(sent)
                has_metaphor, metaphor_type = self.detect_metaphor_analogy(sent.text)

                # Store sentence data
                sentence_data = {
                    'sentence_id': sentence_id,
                    'speaker': speaker,
                    'original_message': message,
                    'text': sent.text.strip(),
                    'tense': tense,
                    'polarity': polarity,
                    'emotion': emotion,
                    'has_metaphor': metaphor_type == 'metaphor',
                    'has_analogy': metaphor_type == 'analogy'
                }
                self.sentences_data.append(sentence_data)

                # Update stats
                self.stats['emotions'][emotion] += 1
                self.stats['tenses'][tense] += 1
                self.stats['polarities'][polarity] += 1

                # Extract words
                for token in sent:
                    if not token.is_punct and not token.is_space:
                        word_data = {
                            'word': token.text,
                            'lemma': token.lemma_,
                            'pos': token.pos_,
                            'sentence_id': sentence_id,
                            'speaker': speaker,
                            'grammatical_role': self.extract_grammatical_role(token)
                        }
                        self.words_data.append(word_data)
                        self.stats['pos_tags'][token.pos_] += 1

                # Extract concepts (noun phrases)
                for chunk in sent.noun_chunks:
                    concept_type = 'proper_noun' if any(t.pos_ == 'PROPN' for t in chunk) else 'concept'

                    concept_data = {
                        'phrase': chunk.text,
                        'type': concept_type,
                        'sentence_id': sentence_id,
                        'speaker': speaker
                    }
                    self.concepts_data.append(concept_data)
                    self.stats['concepts'][chunk.text.lower()] += 1

                # Extract metaphors/analogies
                if has_metaphor or metaphor_type == 'analogy':
                    metaphor_data = {
                        'text': sent.text,
                        'type': metaphor_type,
                        'source_sentence_id': sentence_id,
                        'speaker': speaker,
                        'emotion': emotion,
                        'interpretation': self.interpret_figurative_language(sent.text, metaphor_type)
                    }
                    self.metaphors_data.append(metaphor_data)

    def interpret_figurative_language(self, text, fig_type):
        """Attempt to interpret metaphor or analogy."""
        if fig_type == 'analogy':
            # Look for "X is like Y" pattern
            for indicator in self.metaphor_indicators:
                if indicator in text.lower():
                    parts = text.lower().split(indicator)
                    if len(parts) == 2:
                        return f"Comparing '{parts[0].strip()}' to '{parts[1].strip()}'"

        elif fig_type == 'metaphor':
            # Simple interpretation based on structure
            doc = nlp(text)
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_ in self.metaphor_verbs:
                    subj = [t.text for t in token.children if t.dep_ == "nsubj"]
                    obj = [t.text for t in token.children if t.dep_ in ["attr", "dobj"]]
                    if subj and obj:
                        return f"'{' '.join(subj)}' is metaphorically described as '{' '.join(obj)}'"

        return "Complex figurative language - interpretation unclear"

    def generate_summary(self):
        """Generate summary statistics and report."""
        # Find most common concepts (excluding very common words)
        filtered_concepts = {k: v for k, v in self.stats['concepts'].items()
                           if len(k) > 3 and v > 1}
        top_concepts = Counter(filtered_concepts).most_common(10)

        # Sample metaphors/analogies
        sample_metaphors = self.metaphors_data[:5] if self.metaphors_data else []

        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'total_messages_processed': len(self.sentences_data),
            'total_sentences': len(self.sentences_data),
            'total_words': len(self.words_data),
            'total_concepts': len(self.concepts_data),
            'total_figurative_expressions': len(self.metaphors_data),
            'emotion_distribution': dict(self.stats['emotions']),
            'tense_distribution': dict(self.stats['tenses']),
            'polarity_distribution': dict(self.stats['polarities']),
            'pos_distribution': dict(self.stats['pos_tags'].most_common(10)),
            'top_concepts': [{'concept': c[0], 'frequency': c[1]} for c in top_concepts],
            'sample_metaphors_analogies': [
                {
                    'text': m['text'],
                    'type': m['type'],
                    'speaker': m['speaker'],
                    'interpretation': m['interpretation']
                } for m in sample_metaphors
            ]
        }

        return summary

    def process_file(self, input_path):
        """Process the input JSON file containing message pairs."""
        print(f"🔍 Loading conversations from: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            # Handle both JSONL and JSON array formats
            content = f.read()
            if content.strip().startswith('['):
                # JSON array
                pairs = json.loads(content)
            else:
                # JSONL format
                pairs = []
                for line in content.strip().split('\n'):
                    if line:
                        pairs.append(json.loads(line))

        print(f"📊 Processing {len(pairs)} message pairs...")

        # Process each pair
        for idx, pair in enumerate(pairs):
            if idx % 100 == 0:
                print(f"   Progress: {idx}/{len(pairs)} pairs processed...")
            self.process_message_pair(pair, idx)

        print(f"✅ Processing complete!")

    def save_results(self, output_dir):
        """Save all parsed data to JSON files."""
        output_path = Path(output_dir)

        # Save sentences
        with open(output_path / 'sentences.json', 'w', encoding='utf-8') as f:
            json.dump(self.sentences_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(self.sentences_data)} sentences to sentences.json")

        # Save words
        with open(output_path / 'words.json', 'w', encoding='utf-8') as f:
            json.dump(self.words_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(self.words_data)} words to words.json")

        # Save concepts
        with open(output_path / 'concepts.json', 'w', encoding='utf-8') as f:
            json.dump(self.concepts_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(self.concepts_data)} concepts to concepts.json")

        # Save metaphors
        with open(output_path / 'metaphors.json', 'w', encoding='utf-8') as f:
            json.dump(self.metaphors_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(self.metaphors_data)} figurative expressions to metaphors.json")

        # Save summary
        summary = self.generate_summary()
        with open(output_path / 'summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"📊 Saved analysis summary to summary_report.json")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep parsing of conversation message pairs into multiple analytical layers"
    )
    parser.add_argument(
        'input_file',
        help='Path to JSON/JSONL file containing user/assistant message pairs'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory to save output files (default: current directory)'
    )

    args = parser.parse_args()

    # Initialize parser
    parser = ConversationDeepParser()

    # Process the file
    parser.process_file(args.input_file)

    # Save results
    parser.save_results(args.output_dir)

    print("\n✨ Deep parsing complete! Check the output files for detailed analysis.")

if __name__ == "__main__":
    main()
