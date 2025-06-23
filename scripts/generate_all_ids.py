#!/usr/bin/env python3
"""
Generate All IDs - Main Orchestrator
Comprehensive ID generation across all conversation data
"""

import pandas as pd
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any
import argparse
import os
from collections import defaultdict
import sqlite3
import networkx as nx

class IDGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize ID counters and mappings
        self.id_mappings = {
            'conversation_ids': {},
            'pair_ids': {},
            'sentence_ids': {},
            'word_ids': {},
            'topic_ids': {},
            'concept_ids': {},
            'metaphor_ids': {},
            'project_ids': {},
            'code_block_ids': {},
            'question_ids': {},
            'answer_ids': {},
            'emotion_ids': {},
            'entity_ids': {},
            'intent_ids': {},
            'complexity_ids': {},
            'domain_ids': {},
            'sequence_ids': {},
            'evolution_ids': {},
            'milestone_ids': {},
            'learning_curve_ids': {},
            'thread_ids': {},
            'cluster_ids': {},
            'dependency_ids': {},
            'synonym_ids': {},
            'reference_ids': {},
            'application_ids': {},
            'reasoning_ids': {},
            'creativity_ids': {},
            'problem_solving_ids': {},
            'decision_ids': {},
            'node_ids': {},
            'edge_ids': {},
            'path_ids': {},
            'branch_ids': {},
            'tool_ids': {},
            'platform_ids': {},
            'resource_ids': {},
            'constraint_ids': {},
            'breakthrough_ids': {},
            'failure_ids': {},
            'optimization_ids': {},
            'validation_ids': {}
        }

        # Load existing data
        self.load_existing_data()

    def load_existing_data(self):
        """Load existing ID data from previous runs."""
        print("Loading existing ID data...")

        # Load conversation IDs
        conv_id_file = "id_generation_results/conversation_ids.csv"
        if os.path.exists(conv_id_file):
            self.conversation_ids = pd.read_csv(conv_id_file)
            print(f"Loaded {len(self.conversation_ids)} conversation IDs")

        # Load semantic anchors
        anchors_file = "semantic_intelligence/semantic_anchors.csv"
        if os.path.exists(anchors_file):
            self.semantic_anchors = pd.read_csv(anchors_file)
            print(f"Loaded {len(self.semantic_anchors)} semantic anchors")

        # Load Q&A pairs
        qa_file = "processed_20250621_032815/09_training_data/qa_pairs.jsonl"
        if os.path.exists(qa_file):
            self.qa_pairs = []
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.qa_pairs.append(json.loads(line))
                print(f"Loaded {len(self.qa_pairs)} Q&A pairs")
            except UnicodeDecodeError:
                try:
                    with open(qa_file, 'r', encoding='latin-1') as f:
                        for line in f:
                            self.qa_pairs.append(json.loads(line))
                    print(f"Loaded {len(self.qa_pairs)} Q&A pairs (latin-1 encoding)")
                except Exception as e:
                    print(f"Warning: Could not load Q&A pairs: {e}")
                    self.qa_pairs = []

        # Load code blocks
        code_file = "id_generation_results/code_block_ids.csv"
        if os.path.exists(code_file):
            self.code_blocks = pd.read_csv(code_file)
            print(f"Loaded {len(self.code_blocks)} code blocks")

    def generate_conversation_ids(self):
        """Generate conversation IDs (already done, just verify)."""
        print("Verifying conversation IDs...")
        if hasattr(self, 'conversation_ids'):
            print(f"✅ {len(self.conversation_ids)} conversation IDs verified")
            return self.conversation_ids
        else:
            print("❌ No conversation IDs found")
            return None

    def generate_pair_ids(self):
        """Generate pair IDs from message pairs."""
        print("Generating pair IDs...")

        # Load message pairs
        pairs_file = "processed_20250621_032815/08_message_pairs/all_pairs.jsonl"
        if not os.path.exists(pairs_file):
            print("❌ Message pairs file not found")
            return None

        pairs = []
        try:
            with open(pairs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    pairs.append(json.loads(line))
        except UnicodeDecodeError:
            try:
                with open(pairs_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        pairs.append(json.loads(line))
            except Exception as e:
                print(f"Warning: Could not load message pairs: {e}")
                return None

        pair_ids = []
        for i, pair in enumerate(pairs):
            pair_id = f"pair_{uuid.uuid4().hex[:8]}"
            pair_ids.append({
                'pair_id': pair_id,
                'conversation_id': pair.get('conversation_id', ''),
                'user_message': pair.get('user_message', ''),
                'assistant_message': pair.get('assistant_message', ''),
                'timestamp': pair.get('timestamp', ''),
                'sequence_order': i
            })

        df = pd.DataFrame(pair_ids)
        df.to_csv(f"{self.output_dir}/pair_ids.csv", index=False)
        print(f"✅ Generated {len(pair_ids)} pair IDs")
        return df

    def generate_sentence_ids(self):
        """Generate sentence IDs from sentence analysis."""
        print("Generating sentence IDs...")

        # Load sentence analysis
        sentences_file = "linguistic_analysis/sentences.json"
        if not os.path.exists(sentences_file):
            print("❌ Sentence analysis file not found")
            return None

        try:
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences_data = json.load(f)
        except UnicodeDecodeError:
            try:
                with open(sentences_file, 'r', encoding='latin-1') as f:
                    sentences_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load sentences: {e}")
                return None

        sentence_ids = []

        # Handle both list and dict structures
        if isinstance(sentences_data, dict):
            # Dictionary structure: {conversation_id: [sentences]}
            for conv_id, conv_sentences in sentences_data.items():
                for i, sentence in enumerate(conv_sentences):
                    sentence_id = f"sentence_{uuid.uuid4().hex[:8]}"
                    sentence_ids.append({
                        'sentence_id': sentence_id,
                        'conversation_id': conv_id,
                        'text': sentence.get('text', ''),
                        'position': i,
                        'length': len(sentence.get('text', '')),
                        'complexity_score': sentence.get('complexity', 0)
                    })
        elif isinstance(sentences_data, list):
            # List structure: [sentence_objects]
            for i, sentence in enumerate(sentences_data):
                sentence_id = f"sentence_{uuid.uuid4().hex[:8]}"
                sentence_ids.append({
                    'sentence_id': sentence_id,
                    'conversation_id': sentence.get('conversation_id', ''),
                    'text': sentence.get('text', ''),
                    'position': i,
                    'length': len(sentence.get('text', '')),
                    'complexity_score': sentence.get('complexity', 0)
                })
        else:
            print(f"❌ Unexpected sentences data structure: {type(sentences_data)}")
            return None

        df = pd.DataFrame(sentence_ids)
        df.to_csv(f"{self.output_dir}/sentence_ids.csv", index=False)
        print(f"✅ Generated {len(sentence_ids)} sentence IDs")
        return df

    def generate_concept_ids(self):
        """Generate concept IDs from semantic anchors."""
        print("Generating concept IDs...")

        if not hasattr(self, 'semantic_anchors'):
            print("❌ Semantic anchors not loaded")
            return None

        # Filter for concept-type anchors
        concept_anchors = self.semantic_anchors[self.semantic_anchors['type'] == 'concept'].copy()

        concept_ids = []
        for _, anchor in concept_anchors.iterrows():
            concept_id = f"concept_{uuid.uuid4().hex[:8]}"
            concept_ids.append({
                'concept_id': concept_id,
                'anchor_id': anchor['anchor_id'],
                'text': anchor['text'],
                'conversation_id': anchor['conversation_id'],
                'confidence': anchor['confidence'],
                'cluster_id': anchor['cluster_id']
            })

        df = pd.DataFrame(concept_ids)
        df.to_csv(f"{self.output_dir}/concept_ids.csv", index=False)
        print(f"✅ Generated {len(concept_ids)} concept IDs")
        return df

    def generate_emotion_ids(self):
        """Generate emotion IDs using Plutchik wheel."""
        print("Generating emotion IDs...")

        # Plutchik emotion wheel
        emotions = {
            'joy': ['ecstasy', 'joy', 'serenity'],
            'trust': ['admiration', 'trust', 'acceptance'],
            'fear': ['terror', 'fear', 'apprehension'],
            'surprise': ['amazement', 'surprise', 'distraction'],
            'sadness': ['grief', 'sadness', 'pensiveness'],
            'disgust': ['loathing', 'disgust', 'boredom'],
            'anger': ['rage', 'anger', 'annoyance'],
            'anticipation': ['vigilance', 'anticipation', 'interest']
        }

        # For now, create placeholder emotion IDs
        # In a full implementation, you'd run emotion analysis on the text
        emotion_ids = []
        for emotion_category, intensities in emotions.items():
            for intensity in intensities:
                emotion_id = f"emotion_{uuid.uuid4().hex[:8]}"
                emotion_ids.append({
                    'emotion_id': emotion_id,
                    'category': emotion_category,
                    'intensity': intensity,
                    'description': f"{intensity} level of {emotion_category}"
                })

        df = pd.DataFrame(emotion_ids)
        df.to_csv(f"{self.output_dir}/emotion_ids.csv", index=False)
        print(f"✅ Generated {len(emotion_ids)} emotion IDs (framework)")
        return df

    def generate_entity_ids(self):
        """Generate entity IDs from semantic anchors and named entity recognition."""
        print("Generating entity IDs...")

        if not hasattr(self, 'semantic_anchors'):
            print("❌ Semantic anchors not loaded")
            return None

        # Filter for entity-type anchors
        entity_anchors = self.semantic_anchors[self.semantic_anchors['type'] == 'entity'].copy()

        entity_ids = []
        for _, anchor in entity_anchors.iterrows():
            entity_id = f"entity_{uuid.uuid4().hex[:8]}"
            entity_ids.append({
                'entity_id': entity_id,
                'anchor_id': anchor['anchor_id'],
                'text': anchor['text'],
                'conversation_id': anchor['conversation_id'],
                'confidence': anchor['confidence'],
                'entity_type': self.categorize_entity(anchor['text']),
                'cluster_id': anchor['cluster_id']
            })

        df = pd.DataFrame(entity_ids)
        df.to_csv(f"{self.output_dir}/entity_ids.csv", index=False)
        print(f"✅ Generated {len(entity_ids)} entity IDs")
        return df

    def categorize_entity(self, entity_text: str) -> str:
        """Categorize an entity based on its text."""
        text_lower = entity_text.lower()

        if any(x in text_lower for x in ['gpt', 'claude', 'llama', 'bert', 'transformer']):
            return 'AI_Model'
        elif any(x in text_lower for x in ['python', 'javascript', 'java', 'cpp', 'rust', 'go']):
            return 'Programming_Language'
        elif any(x in text_lower for x in ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'heroku']):
            return 'Platform_Service'
        elif any(x in text_lower for x in ['github', 'gitlab', 'bitbucket', 'git']):
            return 'Version_Control'
        elif any(x in text_lower for x in ['openai', 'anthropic', 'google', 'microsoft', 'meta']):
            return 'Company'
        elif any(x in text_lower for x in ['john', 'jane', 'alice', 'bob', 'user', 'developer']):
            return 'Person'
        else:
            return 'Other'

    def generate_intent_ids(self):
        """Generate intent IDs for message purposes."""
        print("Generating intent IDs...")

        intents = [
            'ask_question', 'provide_answer', 'debug_code', 'explain_concept',
            'build_system', 'optimize_performance', 'plan_project', 'reflect_learning',
            'share_insight', 'request_help', 'give_feedback', 'explore_idea',
            'implement_solution', 'test_hypothesis', 'document_process',
            'analyze_data', 'design_architecture', 'review_code', 'troubleshoot_issue',
            'brainstorm_solution', 'compare_approaches', 'validate_assumption'
        ]

        intent_ids = []
        for intent in intents:
            intent_id = f"intent_{uuid.uuid4().hex[:8]}"
            intent_ids.append({
                'intent_id': intent_id,
                'intent_type': intent,
                'description': intent.replace('_', ' ').title(),
                'category': self.categorize_intent(intent)
            })

        df = pd.DataFrame(intent_ids)
        df.to_csv(f"{self.output_dir}/intent_ids.csv", index=False)
        print(f"✅ Generated {len(intent_ids)} intent IDs")
        return df

    def categorize_intent(self, intent: str) -> str:
        """Categorize an intent based on its type."""
        if 'question' in intent or 'ask' in intent:
            return 'Inquiry'
        elif 'answer' in intent or 'explain' in intent:
            return 'Explanation'
        elif 'debug' in intent or 'troubleshoot' in intent:
            return 'Problem_Solving'
        elif 'build' in intent or 'implement' in intent:
            return 'Implementation'
        elif 'plan' in intent or 'design' in intent:
            return 'Planning'
        elif 'reflect' in intent or 'analyze' in intent:
            return 'Analysis'
        else:
            return 'Other'

    def generate_complexity_ids(self):
        """Generate complexity IDs for skill levels."""
        print("Generating complexity IDs...")

        complexity_levels = [
            {'level': 'beginner', 'description': 'Basic concepts, simple explanations', 'score': 1},
            {'level': 'intermediate', 'description': 'Moderate complexity, practical applications', 'score': 2},
            {'level': 'advanced', 'description': 'Complex topics, deep technical details', 'score': 3},
            {'level': 'expert', 'description': 'Cutting-edge research, sophisticated analysis', 'score': 4},
            {'level': 'research', 'description': 'Novel approaches, experimental methods', 'score': 5}
        ]

        complexity_ids = []
        for comp in complexity_levels:
            complexity_id = f"complexity_{uuid.uuid4().hex[:8]}"
            complexity_ids.append({
                'complexity_id': complexity_id,
                'level': comp['level'],
                'description': comp['description'],
                'score': comp['score']
            })

        df = pd.DataFrame(complexity_ids)
        df.to_csv(f"{self.output_dir}/complexity_ids.csv", index=False)
        print(f"✅ Generated {len(complexity_ids)} complexity IDs")
        return df

    def generate_domain_ids(self):
        """Generate domain IDs for different knowledge areas."""
        print("Generating domain IDs...")

        domains = [
            {'domain': 'ai_ml', 'description': 'Artificial Intelligence and Machine Learning'},
            {'domain': 'programming', 'description': 'Software Development and Programming'},
            {'domain': 'data_science', 'description': 'Data Analysis and Statistics'},
            {'domain': 'business', 'description': 'Business Strategy and Operations'},
            {'domain': 'research', 'description': 'Academic Research and Methodology'},
            {'domain': 'design', 'description': 'User Experience and Design'},
            {'domain': 'infrastructure', 'description': 'DevOps and Infrastructure'},
            {'domain': 'security', 'description': 'Cybersecurity and Privacy'},
            {'domain': 'product', 'description': 'Product Management and Strategy'},
            {'domain': 'education', 'description': 'Learning and Teaching'}
        ]

        domain_ids = []
        for domain in domains:
            domain_id = f"domain_{uuid.uuid4().hex[:8]}"
            domain_ids.append({
                'domain_id': domain_id,
                'domain': domain['domain'],
                'description': domain['description']
            })

        df = pd.DataFrame(domain_ids)
        df.to_csv(f"{self.output_dir}/domain_ids.csv", index=False)
        print(f"✅ Generated {len(domain_ids)} domain IDs")
        return df

    def generate_sequence_ids(self):
        """Generate sequence IDs for message ordering."""
        print("Generating sequence IDs...")

        # This would require analyzing conversation flow
        # For now, create a framework
        sequence_ids = []

        sequence_types = [
            'conversation_flow', 'thought_progression', 'problem_evolution',
            'solution_development', 'learning_sequence', 'decision_chain'
        ]

        for seq_type in sequence_types:
            sequence_id = f"sequence_{uuid.uuid4().hex[:8]}"
            sequence_ids.append({
                'sequence_id': sequence_id,
                'sequence_type': seq_type,
                'description': seq_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(sequence_ids)
        df.to_csv(f"{self.output_dir}/sequence_ids.csv", index=False)
        print(f"✅ Generated {len(sequence_ids)} sequence IDs (framework)")
        return df

    def generate_evolution_ids(self):
        """Generate evolution IDs for idea development trails."""
        print("Generating evolution IDs...")

        evolution_types = [
            'concept_refinement', 'approach_iteration', 'solution_evolution',
            'understanding_deepening', 'skill_progression', 'idea_maturation'
        ]

        evolution_ids = []
        for evo_type in evolution_types:
            evolution_id = f"evolution_{uuid.uuid4().hex[:8]}"
            evolution_ids.append({
                'evolution_id': evolution_id,
                'evolution_type': evo_type,
                'description': evo_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(evolution_ids)
        df.to_csv(f"{self.output_dir}/evolution_ids.csv", index=False)
        print(f"✅ Generated {len(evolution_ids)} evolution IDs")
        return df

    def generate_milestone_ids(self):
        """Generate milestone IDs for decisions or pivots."""
        print("Generating milestone IDs...")

        milestone_types = [
            'decision_point', 'approach_change', 'breakthrough_insight',
            'problem_solved', 'concept_mastered', 'tool_adopted',
            'methodology_shift', 'perspective_change'
        ]

        milestone_ids = []
        for milestone_type in milestone_types:
            milestone_id = f"milestone_{uuid.uuid4().hex[:8]}"
            milestone_ids.append({
                'milestone_id': milestone_id,
                'milestone_type': milestone_type,
                'description': milestone_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(milestone_ids)
        df.to_csv(f"{self.output_dir}/milestone_ids.csv", index=False)
        print(f"✅ Generated {len(milestone_ids)} milestone IDs")
        return df

    def generate_learning_curve_ids(self):
        """Generate learning curve IDs for skill acquisition markers."""
        print("Generating learning curve IDs...")

        learning_stages = [
            {'stage': 'awareness', 'description': 'Initial awareness of concept'},
            {'stage': 'interest', 'description': 'Growing interest and exploration'},
            {'stage': 'trial', 'description': 'First attempts and experimentation'},
            {'stage': 'practice', 'description': 'Regular practice and application'},
            {'stage': 'mastery', 'description': 'Confident and skilled application'},
            {'stage': 'teaching', 'description': 'Able to teach and explain to others'}
        ]

        learning_curve_ids = []
        for stage in learning_stages:
            learning_curve_id = f"learning_curve_{uuid.uuid4().hex[:8]}"
            learning_curve_ids.append({
                'learning_curve_id': learning_curve_id,
                'stage': stage['stage'],
                'description': stage['description']
            })

        df = pd.DataFrame(learning_curve_ids)
        df.to_csv(f"{self.output_dir}/learning_curve_ids.csv", index=False)
        print(f"✅ Generated {len(learning_curve_ids)} learning curve IDs")
        return df

    def generate_dependency_ids(self):
        """Generate dependency IDs for logical or conceptual dependencies."""
        print("Generating dependency IDs...")

        dependency_types = [
            'prerequisite', 'builds_on', 'requires', 'depends_on',
            'enables', 'blocks', 'conflicts_with', 'complements'
        ]

        dependency_ids = []
        for dep_type in dependency_types:
            dependency_id = f"dependency_{uuid.uuid4().hex[:8]}"
            dependency_ids.append({
                'dependency_id': dependency_id,
                'dependency_type': dep_type,
                'description': dep_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(dependency_ids)
        df.to_csv(f"{self.output_dir}/dependency_ids.csv", index=False)
        print(f"✅ Generated {len(dependency_ids)} dependency IDs")
        return df

    def generate_synonym_ids(self):
        """Generate synonym IDs for linguistic equivalence across terms."""
        print("Generating synonym IDs...")

        # This would require semantic analysis to find synonyms
        # For now, create a framework
        synonym_ids = []

        synonym_groups = [
            ['ai', 'artificial intelligence', 'machine intelligence'],
            ['llm', 'large language model', 'language model'],
            ['api', 'application programming interface', 'interface'],
            ['database', 'db', 'data store', 'storage'],
            ['algorithm', 'algo', 'method', 'procedure']
        ]

        for i, group in enumerate(synonym_groups):
            synonym_id = f"synonym_{uuid.uuid4().hex[:8]}"
            synonym_ids.append({
                'synonym_id': synonym_id,
                'group_id': i,
                'terms': '|'.join(group),
                'primary_term': group[0]
            })

        df = pd.DataFrame(synonym_ids)
        df.to_csv(f"{self.output_dir}/synonym_ids.csv", index=False)
        print(f"✅ Generated {len(synonym_ids)} synonym IDs (framework)")
        return df

    def generate_reference_ids(self):
        """Generate reference IDs for cross-conversation links."""
        print("Generating reference IDs...")

        reference_types = [
            'previous_conversation', 'external_resource', 'documentation',
            'research_paper', 'blog_post', 'video_tutorial', 'book_reference',
            'tool_documentation', 'api_reference', 'best_practice'
        ]

        reference_ids = []
        for ref_type in reference_types:
            reference_id = f"reference_{uuid.uuid4().hex[:8]}"
            reference_ids.append({
                'reference_id': reference_id,
                'reference_type': ref_type,
                'description': ref_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(reference_ids)
        df.to_csv(f"{self.output_dir}/reference_ids.csv", index=False)
        print(f"✅ Generated {len(reference_ids)} reference IDs")
        return df

    def generate_application_ids(self):
        """Generate application IDs for concepts applied in new ways."""
        print("Generating application IDs...")

        application_types = [
            'concept_application', 'tool_usage', 'methodology_adaptation',
            'pattern_implementation', 'principle_application', 'technique_usage'
        ]

        application_ids = []
        for app_type in application_types:
            application_id = f"application_{uuid.uuid4().hex[:8]}"
            application_ids.append({
                'application_id': application_id,
                'application_type': app_type,
                'description': app_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(application_ids)
        df.to_csv(f"{self.output_dir}/application_ids.csv", index=False)
        print(f"✅ Generated {len(application_ids)} application IDs")
        return df

    def generate_reasoning_ids(self):
        """Generate reasoning IDs for chains of thought."""
        print("Generating reasoning IDs...")

        reasoning_types = [
            'logical_deduction', 'inductive_reasoning', 'abductive_reasoning',
            'analogical_reasoning', 'causal_reasoning', 'comparative_analysis',
            'hypothesis_testing', 'evidence_evaluation'
        ]

        reasoning_ids = []
        for reason_type in reasoning_types:
            reasoning_id = f"reasoning_{uuid.uuid4().hex[:8]}"
            reasoning_ids.append({
                'reasoning_id': reasoning_id,
                'reasoning_type': reason_type,
                'description': reason_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(reasoning_ids)
        df.to_csv(f"{self.output_dir}/reasoning_ids.csv", index=False)
        print(f"✅ Generated {len(reasoning_ids)} reasoning IDs")
        return df

    def generate_creativity_ids(self):
        """Generate creativity IDs for ideation or synthesis moments."""
        print("Generating creativity IDs...")

        creativity_types = [
            'idea_generation', 'concept_synthesis', 'innovation_moment',
            'creative_solution', 'novel_approach', 'breakthrough_thinking',
            'lateral_thinking', 'creative_combination'
        ]

        creativity_ids = []
        for creative_type in creativity_types:
            creativity_id = f"creativity_{uuid.uuid4().hex[:8]}"
            creativity_ids.append({
                'creativity_id': creativity_id,
                'creativity_type': creative_type,
                'description': creative_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(creativity_ids)
        df.to_csv(f"{self.output_dir}/creativity_ids.csv", index=False)
        print(f"✅ Generated {len(creativity_ids)} creativity IDs")
        return df

    def generate_problem_solving_ids(self):
        """Generate problem solving IDs for diagnosis & solution patterns."""
        print("Generating problem solving IDs...")

        problem_solving_stages = [
            'problem_identification', 'root_cause_analysis', 'solution_generation',
            'solution_evaluation', 'implementation', 'verification',
            'optimization', 'prevention'
        ]

        problem_solving_ids = []
        for stage in problem_solving_stages:
            problem_solving_id = f"problem_solving_{uuid.uuid4().hex[:8]}"
            problem_solving_ids.append({
                'problem_solving_id': problem_solving_id,
                'stage': stage,
                'description': stage.replace('_', ' ').title()
            })

        df = pd.DataFrame(problem_solving_ids)
        df.to_csv(f"{self.output_dir}/problem_solving_ids.csv", index=False)
        print(f"✅ Generated {len(problem_solving_ids)} problem solving IDs")
        return df

    def generate_decision_ids(self):
        """Generate decision IDs for forks or trade-offs."""
        print("Generating decision IDs...")

        decision_types = [
            'approach_choice', 'technology_selection', 'architecture_decision',
            'trade_off_analysis', 'priority_setting', 'resource_allocation',
            'timeline_decision', 'scope_decision'
        ]

        decision_ids = []
        for decision_type in decision_types:
            decision_id = f"decision_{uuid.uuid4().hex[:8]}"
            decision_ids.append({
                'decision_id': decision_id,
                'decision_type': decision_type,
                'description': decision_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(decision_ids)
        df.to_csv(f"{self.output_dir}/decision_ids.csv", index=False)
        print(f"✅ Generated {len(decision_ids)} decision IDs")
        return df

    def generate_tool_ids(self):
        """Generate tool IDs from mentioned technologies."""
        print("Generating tool IDs...")

        # Extract tools from semantic anchors
        if hasattr(self, 'semantic_anchors'):
            tech_anchors = self.semantic_anchors[
                self.semantic_anchors['type'] == 'technology'
            ].copy()

            # Get unique tools
            unique_tools = tech_anchors['text'].value_counts().head(100)

            tool_ids = []
            for tool_name, count in unique_tools.items():
                tool_id = f"tool_{uuid.uuid4().hex[:8]}"
                tool_ids.append({
                    'tool_id': tool_id,
                    'name': tool_name,
                    'mention_count': count,
                    'category': self.categorize_tool(tool_name)
                })

            df = pd.DataFrame(tool_ids)
            df.to_csv(f"{self.output_dir}/tool_ids.csv", index=False)
            print(f"✅ Generated {len(tool_ids)} tool IDs")
            return df
        else:
            print("❌ No semantic anchors available for tool extraction")
            return None

    def categorize_tool(self, tool_name: str) -> str:
        """Categorize a tool based on its name."""
        tool_name_lower = tool_name.lower()

        if any(x in tool_name_lower for x in ['gpt', 'claude', 'llm', 'ai', 'model']):
            return 'AI_Model'
        elif any(x in tool_name_lower for x in ['python', 'javascript', 'java', 'cpp', 'rust']):
            return 'Programming_Language'
        elif any(x in tool_name_lower for x in ['docker', 'kubernetes', 'aws', 'azure', 'gcp']):
            return 'Infrastructure'
        elif any(x in tool_name_lower for x in ['git', 'github', 'gitlab']):
            return 'Version_Control'
        elif any(x in tool_name_lower for x in ['pandas', 'numpy', 'tensorflow', 'pytorch']):
            return 'Data_Science'
        else:
            return 'Other'

    def generate_platform_ids(self):
        """Generate platform IDs for where conversations took place."""
        print("Generating platform IDs...")

        platforms = [
            {'platform': 'chatgpt', 'description': 'OpenAI ChatGPT'},
            {'platform': 'claude', 'description': 'Anthropic Claude'},
            {'platform': 'bard', 'description': 'Google Bard'},
            {'platform': 'bing', 'description': 'Microsoft Bing Chat'},
            {'platform': 'replit', 'description': 'Replit AI'},
            {'platform': 'github_copilot', 'description': 'GitHub Copilot'},
            {'platform': 'cursor', 'description': 'Cursor IDE'},
            {'platform': 'other', 'description': 'Other platforms'}
        ]

        platform_ids = []
        for platform in platforms:
            platform_id = f"platform_{uuid.uuid4().hex[:8]}"
            platform_ids.append({
                'platform_id': platform_id,
                'platform': platform['platform'],
                'description': platform['description']
            })

        df = pd.DataFrame(platform_ids)
        df.to_csv(f"{self.output_dir}/platform_ids.csv", index=False)
        print(f"✅ Generated {len(platform_ids)} platform IDs")
        return df

    def generate_resource_ids(self):
        """Generate resource IDs for external links, books, docs."""
        print("Generating resource IDs...")

        resource_types = [
            'documentation', 'tutorial', 'research_paper', 'blog_post',
            'video', 'book', 'course', 'workshop', 'conference',
            'github_repo', 'npm_package', 'pypi_package', 'docker_image'
        ]

        resource_ids = []
        for resource_type in resource_types:
            resource_id = f"resource_{uuid.uuid4().hex[:8]}"
            resource_ids.append({
                'resource_id': resource_id,
                'resource_type': resource_type,
                'description': resource_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(resource_ids)
        df.to_csv(f"{self.output_dir}/resource_ids.csv", index=False)
        print(f"✅ Generated {len(resource_ids)} resource IDs")
        return df

    def generate_constraint_ids(self):
        """Generate constraint IDs for problem framing limits."""
        print("Generating constraint IDs...")

        constraint_types = [
            'time_constraint', 'budget_constraint', 'technical_constraint',
            'security_constraint', 'performance_constraint', 'compatibility_constraint',
            'resource_constraint', 'regulatory_constraint'
        ]

        constraint_ids = []
        for constraint_type in constraint_types:
            constraint_id = f"constraint_{uuid.uuid4().hex[:8]}"
            constraint_ids.append({
                'constraint_id': constraint_id,
                'constraint_type': constraint_type,
                'description': constraint_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(constraint_ids)
        df.to_csv(f"{self.output_dir}/constraint_ids.csv", index=False)
        print(f"✅ Generated {len(constraint_ids)} constraint IDs")
        return df

    def generate_breakthrough_ids(self):
        """Generate breakthrough IDs for insight emergence."""
        print("Generating breakthrough IDs...")

        breakthrough_types = [
            'aha_moment', 'pattern_recognition', 'connection_discovery',
            'solution_insight', 'concept_clarification', 'methodology_breakthrough',
            'tool_revelation', 'approach_innovation'
        ]

        breakthrough_ids = []
        for breakthrough_type in breakthrough_types:
            breakthrough_id = f"breakthrough_{uuid.uuid4().hex[:8]}"
            breakthrough_ids.append({
                'breakthrough_id': breakthrough_id,
                'breakthrough_type': breakthrough_type,
                'description': breakthrough_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(breakthrough_ids)
        df.to_csv(f"{self.output_dir}/breakthrough_ids.csv", index=False)
        print(f"✅ Generated {len(breakthrough_ids)} breakthrough IDs")
        return df

    def generate_failure_ids(self):
        """Generate failure IDs for problem detection or missteps."""
        print("Generating failure IDs...")

        failure_types = [
            'approach_failure', 'tool_failure', 'concept_misunderstanding',
            'implementation_error', 'design_flaw', 'assumption_error',
            'timeline_overrun', 'scope_creep'
        ]

        failure_ids = []
        for failure_type in failure_types:
            failure_id = f"failure_{uuid.uuid4().hex[:8]}"
            failure_ids.append({
                'failure_id': failure_id,
                'failure_type': failure_type,
                'description': failure_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(failure_ids)
        df.to_csv(f"{self.output_dir}/failure_ids.csv", index=False)
        print(f"✅ Generated {len(failure_ids)} failure IDs")
        return df

    def generate_optimization_ids(self):
        """Generate optimization IDs for systemic improvement."""
        print("Generating optimization IDs...")

        optimization_types = [
            'performance_optimization', 'code_optimization', 'process_optimization',
            'workflow_optimization', 'resource_optimization', 'efficiency_improvement',
            'quality_enhancement', 'user_experience_optimization'
        ]

        optimization_ids = []
        for optimization_type in optimization_types:
            optimization_id = f"optimization_{uuid.uuid4().hex[:8]}"
            optimization_ids.append({
                'optimization_id': optimization_id,
                'optimization_type': optimization_type,
                'description': optimization_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(optimization_ids)
        df.to_csv(f"{self.output_dir}/optimization_ids.csv", index=False)
        print(f"✅ Generated {len(optimization_ids)} optimization IDs")
        return df

    def generate_validation_ids(self):
        """Generate validation IDs for tests or confirmations."""
        print("Generating validation IDs...")

        validation_types = [
            'unit_test', 'integration_test', 'user_testing', 'performance_test',
            'security_test', 'compatibility_test', 'usability_test', 'acceptance_test'
        ]

        validation_ids = []
        for validation_type in validation_types:
            validation_id = f"validation_{uuid.uuid4().hex[:8]}"
            validation_ids.append({
                'validation_id': validation_id,
                'validation_type': validation_type,
                'description': validation_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(validation_ids)
        df.to_csv(f"{self.output_dir}/validation_ids.csv", index=False)
        print(f"✅ Generated {len(validation_ids)} validation IDs")
        return df

    def generate_knowledge_graph_ids(self):
        """Generate knowledge graph layer IDs (node_id, edge_id, path_id, branch_id)."""
        print("Generating knowledge graph layer IDs...")

        # Node IDs (representing entities in the graph)
        node_types = [
            'conversation_node', 'sentence_node', 'concept_node', 'entity_node',
            'emotion_node', 'tool_node', 'intent_node', 'code_block_node'
        ]

        node_ids = []
        for node_type in node_types:
            node_id = f"node_{uuid.uuid4().hex[:8]}"
            node_ids.append({
                'node_id': node_id,
                'node_type': node_type,
                'description': node_type.replace('_', ' ').title()
            })

        # Edge IDs (representing relationships)
        edge_types = [
            'contains', 'expresses', 'mentions', 'has_intent', 'references',
            'depends_on', 'enables', 'conflicts_with', 'complements', 'evolves_from'
        ]

        edge_ids = []
        for edge_type in edge_types:
            edge_id = f"edge_{uuid.uuid4().hex[:8]}"
            edge_ids.append({
                'edge_id': edge_id,
                'edge_type': edge_type,
                'description': edge_type.replace('_', ' ').title()
            })

        # Path IDs (representing logical/cognitive trails)
        path_types = [
            'learning_path', 'problem_solving_path', 'decision_path',
            'concept_evolution_path', 'skill_development_path'
        ]

        path_ids = []
        for path_type in path_types:
            path_id = f"path_{uuid.uuid4().hex[:8]}"
            path_ids.append({
                'path_id': path_id,
                'path_type': path_type,
                'description': path_type.replace('_', ' ').title()
            })

        # Branch IDs (representing divergent or parallel explorations)
        branch_types = [
            'approach_branch', 'technology_branch', 'methodology_branch',
            'concept_branch', 'solution_branch'
        ]

        branch_ids = []
        for branch_type in branch_types:
            branch_id = f"branch_{uuid.uuid4().hex[:8]}"
            branch_ids.append({
                'branch_id': branch_id,
                'branch_type': branch_type,
                'description': branch_type.replace('_', ' ').title()
            })

        # Save all knowledge graph IDs
        node_df = pd.DataFrame(node_ids)
        edge_df = pd.DataFrame(edge_ids)
        path_df = pd.DataFrame(path_ids)
        branch_df = pd.DataFrame(branch_ids)

        node_df.to_csv(f"{self.output_dir}/node_ids.csv", index=False)
        edge_df.to_csv(f"{self.output_dir}/edge_ids.csv", index=False)
        path_df.to_csv(f"{self.output_dir}/path_ids.csv", index=False)
        branch_df.to_csv(f"{self.output_dir}/branch_ids.csv", index=False)

        print(f"✅ Generated {len(node_ids)} node IDs")
        print(f"✅ Generated {len(edge_ids)} edge IDs")
        print(f"✅ Generated {len(path_ids)} path IDs")
        print(f"✅ Generated {len(branch_ids)} branch IDs")

        return {
            'node_ids': node_df,
            'edge_ids': edge_df,
            'path_ids': path_df,
            'branch_ids': branch_df
        }

    def generate_thread_ids(self):
        """Generate thread IDs for cross-message continuity."""
        print("Generating thread IDs...")

        # This would require analyzing message sequences for continuity
        # For now, create a framework
        thread_ids = []

        # Example thread types
        thread_types = [
            'code_debugging', 'concept_exploration', 'system_design',
            'problem_solving', 'learning_progression', 'idea_development'
        ]

        for thread_type in thread_types:
            thread_id = f"thread_{uuid.uuid4().hex[:8]}"
            thread_ids.append({
                'thread_id': thread_id,
                'thread_type': thread_type,
                'description': thread_type.replace('_', ' ').title()
            })

        df = pd.DataFrame(thread_ids)
        df.to_csv(f"{self.output_dir}/thread_ids.csv", index=False)
        print(f"✅ Generated {len(thread_ids)} thread IDs (framework)")
        return df

    def create_knowledge_graph_schema(self):
        """Create the knowledge graph schema."""
        print("Creating knowledge graph schema...")

        schema = {
            "nodes": {
                "conversation": {
                    "properties": ["conversation_id", "title", "date", "model", "topic"]
                },
                "sentence": {
                    "properties": ["sentence_id", "text", "complexity", "position"]
                },
                "concept": {
                    "properties": ["concept_id", "text", "confidence", "type"]
                },
                "entity": {
                    "properties": ["entity_id", "text", "entity_type", "confidence"]
                },
                "emotion": {
                    "properties": ["emotion_id", "category", "intensity"]
                },
                "tool": {
                    "properties": ["tool_id", "name", "category", "mention_count"]
                },
                "intent": {
                    "properties": ["intent_id", "intent_type", "description"]
                },
                "code_block": {
                    "properties": ["code_block_id", "language", "content", "context"]
                }
            },
            "edges": {
                "contains": {
                    "from": "conversation",
                    "to": "sentence",
                    "properties": ["weight"]
                },
                "expresses": {
                    "from": "sentence",
                    "to": "emotion",
                    "properties": ["confidence"]
                },
                "mentions": {
                    "from": "sentence",
                    "to": "tool",
                    "properties": ["frequency"]
                },
                "has_intent": {
                    "from": "sentence",
                    "to": "intent",
                    "properties": ["confidence"]
                },
                "references": {
                    "from": "sentence",
                    "to": "concept",
                    "properties": ["relevance"]
                },
                "contains_code": {
                    "from": "conversation",
                    "to": "code_block",
                    "properties": ["language"]
                },
                "depends_on": {
                    "from": "concept",
                    "to": "concept",
                    "properties": ["strength"]
                },
                "evolves_from": {
                    "from": "concept",
                    "to": "concept",
                    "properties": ["evolution_type"]
                }
            }
        }

        schema_file = f"{self.output_dir}/knowledge_graph_schema.json"
        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)

        print(f"✅ Knowledge graph schema saved to {schema_file}")
        return schema

    def generate_all_ids(self):
        """Generate all ID types."""
        print("🚀 Starting comprehensive ID generation...")

        results = {}

        # Generate all ID types
        results['conversation_ids'] = self.generate_conversation_ids()
        results['pair_ids'] = self.generate_pair_ids()
        results['sentence_ids'] = self.generate_sentence_ids()
        results['concept_ids'] = self.generate_concept_ids()
        results['emotion_ids'] = self.generate_emotion_ids()
        results['intent_ids'] = self.generate_intent_ids()
        results['complexity_ids'] = self.generate_complexity_ids()
        results['tool_ids'] = self.generate_tool_ids()
        results['thread_ids'] = self.generate_thread_ids()
        results['entity_ids'] = self.generate_entity_ids()
        results['domain_ids'] = self.generate_domain_ids()
        results['sequence_ids'] = self.generate_sequence_ids()
        results['evolution_ids'] = self.generate_evolution_ids()
        results['milestone_ids'] = self.generate_milestone_ids()
        results['learning_curve_ids'] = self.generate_learning_curve_ids()
        results['dependency_ids'] = self.generate_dependency_ids()
        results['synonym_ids'] = self.generate_synonym_ids()
        results['reference_ids'] = self.generate_reference_ids()
        results['application_ids'] = self.generate_application_ids()
        results['reasoning_ids'] = self.generate_reasoning_ids()
        results['creativity_ids'] = self.generate_creativity_ids()
        results['problem_solving_ids'] = self.generate_problem_solving_ids()
        results['decision_ids'] = self.generate_decision_ids()
        results['platform_ids'] = self.generate_platform_ids()
        results['resource_ids'] = self.generate_resource_ids()
        results['constraint_ids'] = self.generate_constraint_ids()
        results['breakthrough_ids'] = self.generate_breakthrough_ids()
        results['failure_ids'] = self.generate_failure_ids()
        results['optimization_ids'] = self.generate_optimization_ids()
        results['validation_ids'] = self.generate_validation_ids()
        results['knowledge_graph_ids'] = self.generate_knowledge_graph_ids()

        # Create knowledge graph schema
        results['knowledge_graph_schema'] = self.create_knowledge_graph_schema()

        # Generate summary report
        self.generate_summary_report(results)

        print("✅ All ID generation complete!")
        return results

    def generate_summary_report(self, results: Dict):
        """Generate a summary report of all generated IDs."""
        print("Generating summary report...")

        report = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_id_types": len(results),
            "id_counts": {}
        }

        for id_type, data in results.items():
            if data is not None and hasattr(data, '__len__'):
                if isinstance(data, pd.DataFrame):
                    report["id_counts"][id_type] = len(data)
                elif isinstance(data, list):
                    report["id_counts"][id_type] = len(data)
                else:
                    report["id_counts"][id_type] = "N/A"
            else:
                report["id_counts"][id_type] = "Not generated"

        report_file = f"{self.output_dir}/id_generation_summary.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Summary report saved to {report_file}")
        return report

def main():
    parser = argparse.ArgumentParser(description='Generate all ID types for conversation analysis')
    parser.add_argument('-o', '--output', default='id_generation_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Initialize generator
    generator = IDGenerator(args.output)

    # Generate all IDs
    results = generator.generate_all_ids()

    print("\n🎉 ID Generation Complete!")
    print(f"Results saved to: {args.output}/")

if __name__ == "__main__":
    main()
