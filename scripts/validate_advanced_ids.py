#!/usr/bin/env python3
"""
Validate Advanced ID Generation Scripts
Check all advanced ID scripts and create cross-mapping files
"""

import os
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any

class AdvancedIDValidator:
    def __init__(self, id_results_dir: str, output_dir: str):
        self.id_results_dir = Path(id_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Expected ID files
        self.expected_files = {
            'concept_ids.csv': 'concept_id',
            'emotion_ids.csv': 'emotion_id',
            'metaphor_ids.csv': 'metaphor_id',
            'project_ids.csv': 'project_id',
            'thread_ids.csv': 'thread_id',
            'sentence_ids.csv': 'sentence_id',
            'pair_ids.csv': 'pair_id',
            'conversation_ids.csv': 'conversation_id'
        }

        self.validation_results = {}

    def validate_id_files(self) -> Dict[str, Any]:
        """Validate that all expected ID files exist and have correct structure."""
        print("🔍 Validating ID generation files...")

        for filename, id_column in self.expected_files.items():
            filepath = self.id_results_dir / filename
            result = {
                'exists': False,
                'row_count': 0,
                'has_id_column': False,
                'sample_ids': [],
                'errors': []
            }

            if filepath.exists():
                result['exists'] = True
                try:
                    df = pd.read_csv(filepath)
                    result['row_count'] = len(df)

                    if id_column in df.columns:
                        result['has_id_column'] = True
                        result['sample_ids'] = df[id_column].head(5).tolist()
                    else:
                        result['errors'].append(f"Missing required column: {id_column}")

                except Exception as e:
                    result['errors'].append(f"Error reading file: {str(e)}")
            else:
                result['errors'].append("File not found")

            self.validation_results[filename] = result

        return self.validation_results

    def create_cross_mapping_files(self) -> Dict[str, str]:
        """Create cross-mapping files between different ID types."""
        print("🔗 Creating cross-mapping files...")

        mapping_files = {}

        # Load all ID data
        id_data = {}
        for filename in self.expected_files.keys():
            filepath = self.id_results_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    id_data[filename] = df
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
                    continue

        # Create conversation-to-sentence mapping
        if 'conversation_ids.csv' in id_data and 'sentence_ids.csv' in id_data:
            conv_sent_mapping = self._create_conversation_sentence_mapping(
                id_data['conversation_ids.csv'],
                id_data['sentence_ids.csv']
            )
            mapping_file = self.output_dir / 'conversation_sentence_mapping.csv'
            conv_sent_mapping.to_csv(mapping_file, index=False)
            mapping_files['conversation_sentence'] = str(mapping_file)

        # Create sentence-to-concept mapping
        if 'sentence_ids.csv' in id_data and 'concept_ids.csv' in id_data:
            sent_concept_mapping = self._create_sentence_concept_mapping(
                id_data['sentence_ids.csv'],
                id_data['concept_ids.csv']
            )
            mapping_file = self.output_dir / 'sentence_concept_mapping.csv'
            sent_concept_mapping.to_csv(mapping_file, index=False)
            mapping_files['sentence_concept'] = str(mapping_file)

        # Create thread-to-sentence mapping
        if 'thread_ids.csv' in id_data and 'sentence_ids.csv' in id_data:
            thread_sent_mapping = self._create_thread_sentence_mapping(
                id_data['thread_ids.csv'],
                id_data['sentence_ids.csv']
            )
            mapping_file = self.output_dir / 'thread_sentence_mapping.csv'
            thread_sent_mapping.to_csv(mapping_file, index=False)
            mapping_files['thread_sentence'] = str(mapping_file)

        return mapping_files

    def _create_conversation_sentence_mapping(self, conv_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
        """Create mapping between conversations and sentences."""
        # This would need conversation_id in sentence data
        # For now, create a framework mapping
        mapping_data = []

        if 'conversation_id' in sent_df.columns:
            for _, sent_row in sent_df.iterrows():
                mapping_data.append({
                    'conversation_id': sent_row['conversation_id'],
                    'sentence_id': sent_row['sentence_id'],
                    'mapping_type': 'contains'
                })

        return pd.DataFrame(mapping_data)

    def _create_sentence_concept_mapping(self, sent_df: pd.DataFrame, concept_df: pd.DataFrame) -> pd.DataFrame:
        """Create mapping between sentences and concepts."""
        # This would need semantic analysis to link sentences to concepts
        # For now, create a framework mapping
        mapping_data = []

        # Sample mapping - in reality this would come from semantic analysis
        for i, (_, sent_row) in enumerate(sent_df.head(100).iterrows()):
            if i < len(concept_df):
                concept_row = concept_df.iloc[i]
                mapping_data.append({
                    'sentence_id': sent_row['sentence_id'],
                    'concept_id': concept_row['concept_id'],
                    'relevance_score': 0.8,  # Placeholder
                    'mapping_type': 'expresses'
                })

        return pd.DataFrame(mapping_data)

    def _create_thread_sentence_mapping(self, thread_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
        """Create mapping between threads and sentences."""
        # This would need thread_id in sentence data
        # For now, create a framework mapping
        mapping_data = []

        if 'thread_id' in sent_df.columns:
            for _, sent_row in sent_df.iterrows():
                mapping_data.append({
                    'thread_id': sent_row['thread_id'],
                    'sentence_id': sent_row['sentence_id'],
                    'position_in_thread': 0,  # Placeholder
                    'mapping_type': 'belongs_to'
                })

        return pd.DataFrame(mapping_data)

    def create_unified_id_registry(self) -> str:
        """Create a unified registry of all IDs with metadata."""
        print("📋 Creating unified ID registry...")

        registry_data = []

        # Load all ID data and create unified registry
        for filename, id_column in self.expected_files.items():
            filepath = self.id_results_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    id_type = filename.replace('_ids.csv', '')

                    for _, row in df.iterrows():
                        registry_entry = {
                            'id': row[id_column],
                            'id_type': id_type,
                            'source_file': filename,
                            'created_timestamp': datetime.now().isoformat(),
                            'metadata': {}
                        }

                        # Add relevant metadata based on ID type
                        if id_type == 'conversation':
                            registry_entry['metadata'] = {
                                'title': row.get('title', ''),
                                'date': row.get('date', ''),
                                'model': row.get('model', '')
                            }
                        elif id_type == 'sentence':
                            registry_entry['metadata'] = {
                                'text_preview': str(row.get('text', ''))[:100],
                                'complexity': row.get('complexity', ''),
                                'position': row.get('position', '')
                            }
                        elif id_type == 'concept':
                            registry_entry['metadata'] = {
                                'concept_text': row.get('concept', ''),
                                'confidence': row.get('confidence', ''),
                                'type': row.get('type', '')
                            }

                        registry_data.append(registry_entry)

                except Exception as e:
                    print(f"⚠️ Error processing {filename}: {e}")

        registry_df = pd.DataFrame(registry_data)
        registry_file = self.output_dir / 'unified_id_registry.csv'
        registry_df.to_csv(registry_file, index=False)

        return str(registry_file)

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        print("📊 Generating validation report...")

        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files_expected': len(self.expected_files),
                'files_found': sum(1 for r in self.validation_results.values() if r['exists']),
                'files_with_errors': sum(1 for r in self.validation_results.values() if r['errors']),
                'total_ids_generated': sum(r['row_count'] for r in self.validation_results.values())
            },
            'file_validation': self.validation_results,
            'recommendations': []
        }

        # Generate recommendations
        for filename, result in self.validation_results.items():
            if not result['exists']:
                report['recommendations'].append(f"Generate missing file: {filename}")
            elif result['errors']:
                report['recommendations'].append(f"Fix errors in {filename}: {result['errors']}")
            elif result['row_count'] == 0:
                report['recommendations'].append(f"Empty file: {filename} - check generation script")

        report_file = self.output_dir / 'advanced_id_validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return str(report_file)

def main():
    parser = argparse.ArgumentParser(description="Validate advanced ID generation and create cross-mappings")
    parser.add_argument("--id-results-dir", default="integrated_20250623_011924/id_generation_results_new",
                       help="Directory containing ID generation results")
    parser.add_argument("--output-dir", default="integrated_20250623_011924/advanced_id_validation",
                       help="Output directory for validation results")

    args = parser.parse_args()

    validator = AdvancedIDValidator(args.id_results_dir, args.output_dir)

    # Run validation
    validation_results = validator.validate_id_files()

    # Create cross-mapping files
    mapping_files = validator.create_cross_mapping_files()

    # Create unified registry
    registry_file = validator.create_unified_id_registry()

    # Generate report
    report_file = validator.generate_validation_report()

    print(f"\n✅ Validation complete!")
    print(f"📊 Report: {report_file}")
    print(f"📋 Registry: {registry_file}")
    print(f"🔗 Cross-mappings: {len(mapping_files)} files created")

    # Print summary
    total_files = len(validation_results)
    found_files = sum(1 for r in validation_results.values() if r['exists'])
    total_ids = sum(r['row_count'] for r in validation_results.values())

    print(f"\n📈 Summary:")
    print(f"   Files found: {found_files}/{total_files}")
    print(f"   Total IDs: {total_ids:,}")
    print(f"   Cross-mappings: {len(mapping_files)}")

if __name__ == "__main__":
    main()
