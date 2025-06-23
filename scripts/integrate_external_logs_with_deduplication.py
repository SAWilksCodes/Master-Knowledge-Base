import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import argparse
import csv

class ExternalLogsIntegratorWithDeduplication:
    """Integrate external logs with intelligent deduplication."""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.integration_results = {
            'timestamp': self.timestamp,
            'sources_processed': {},
            'files_integrated': [],
            'deduplication_stats': {
                'total_conversations': 0,
                'duplicate_conversations': 0,
                'new_conversations': 0,
                'duplicate_identifiers': []
            },
            'total_files': 0,
            'successful_integrations': 0
        }

        # Load existing conversation IDs for deduplication
        self.existing_conversation_ids = self.load_existing_conversation_ids()
        self.existing_conversation_hashes = self.load_existing_conversation_hashes()

    def load_existing_conversation_ids(self):
        """Load conversation IDs from existing processed data."""
        existing_ids = set()

        # Load from conversation index
        index_file = self.base_dir / "processed_20250621_032815" / "10_reports" / "conversation_index.csv"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('conversation_id'):
                            existing_ids.add(row['conversation_id'])
            except Exception as e:
                print(f"Warning: Could not load conversation index: {e}")

        # Load from individual conversation files
        split_dir = self.base_dir / "processed_20250621_032815" / "01_split_json"
        if split_dir.exists():
            for conv_file in split_dir.glob("*.json"):
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        conv_data = json.load(f)
                        if isinstance(conv_data, dict) and conv_data.get('conversation_id'):
                            existing_ids.add(conv_data['conversation_id'])
                except Exception as e:
                    print(f"Warning: Could not read {conv_file}: {e}")

        print(f"📊 Loaded {len(existing_ids)} existing conversation IDs for deduplication")
        return existing_ids

    def load_existing_conversation_hashes(self):
        """Load conversation content hashes for additional deduplication."""
        existing_hashes = set()

        # Load from individual conversation files
        split_dir = self.base_dir / "processed_20250621_032815" / "01_split_json"
        if split_dir.exists():
            for conv_file in split_dir.glob("*.json"):
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        conv_data = json.load(f)
                        if isinstance(conv_data, dict):
                            # Create hash from conversation content
                            content_hash = self.create_conversation_hash(conv_data)
                            existing_hashes.add(content_hash)
                except Exception as e:
                    print(f"Warning: Could not read {conv_file}: {e}")

        print(f"📊 Loaded {len(existing_hashes)} existing conversation hashes for deduplication")
        return existing_hashes

    def create_conversation_hash(self, conversation):
        """Create a hash from conversation content for deduplication."""
        # Extract key content for hashing
        content_parts = []

        if isinstance(conversation, dict):
            # Add title
            if conversation.get('title'):
                content_parts.append(conversation['title'])

            # Add first few messages
            if conversation.get('mapping'):
                messages = []
                for msg_id, msg_data in conversation['mapping'].items():
                    if isinstance(msg_data, dict) and msg_data.get('message'):
                        message = msg_data['message']
                        if isinstance(message, dict):
                            content = message.get('content', [])
                            if isinstance(content, list):
                                for part in content[:2]:  # First 2 content parts
                                    if isinstance(part, dict) and part.get('text'):
                                        messages.append(part['text'][:200])  # First 200 chars
                            elif isinstance(content, str):
                                messages.append(content[:200])

                content_parts.extend(messages[:3])  # First 3 messages

        # Create hash
        content_string = "|".join(content_parts)
        return hashlib.md5(content_string.encode('utf-8')).hexdigest()

    def define_integration_sources(self):
        """Define all sources to be integrated."""
        return {
            'chatgpt_external': {
                'name': 'ChatGPT External Conversations',
                'files': [
                    'external_logs/chatgpt/conversations.json'
                ],
                'type': 'conversation_json'
            },
            'claude': {
                'name': 'Claude Conversations',
                'files': [
                    'external_logs/claude/raw_exports/conversations.json',
                    'external_logs/claude/raw_exports/projects.json',
                    'external_logs/claude/raw_exports/users.json'
                ],
                'type': 'conversation_json'
            },
            'manus_omnisync': {
                'name': 'Manus OmniSync Gold Standard Training Data System',
                'files': [
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/1_prompt.txt',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/2_execution_plan.md',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/3_execution_log.txt',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/validate_jsonl.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/usage_examples.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/training-data-v1.0.0.json',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/tokenizer.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/test_suite.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/schema_validator.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/quality_engine.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/omnisync.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/OmniSync Training Data API Documentation.md',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/OmniSync Gold Standard Training Data System.md',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/OmniSync Gold Standard Training Data System - Project Summary.md',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/ingestion_pipeline.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/contributor_workflow.py',
                    'external_logs/manus/sessions/2025_06_08_omnisync_gold_standard_training_data_system/artifacts/ci.yml'
                ],
                'type': 'project_files'
            },
            'manus_youtube_qa': {
                'name': 'Manus YouTube QA Extractor',
                'files': [
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/1_prompt.txt',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/2_execution_plan.md',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/3_execution_log.txt',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/artifacts/youtube_qa_extractor.py',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/artifacts/YouTube QA Extractor.md',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/artifacts/YouTube QA Extractor - Project Summary.md',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/artifacts/validate_output.py',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/artifacts/setup.sh',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/artifacts/run_tests.py',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/artifacts/final_test_report.json',
                    'external_logs/manus/sessions/2025_06_08_youtube_qa_extractor/artifacts/config.json.template'
                ],
                'type': 'project_files'
            },
            'manus_sessions': {
                'name': 'Manus Session Data',
                'files': [
                    'external_logs/manus/sessions/sessions.txt',
                    'external_logs/manus/sessions/session_events.txt'
                ],
                'type': 'session_data'
            },
            'cursor': {
                'name': 'Cursor Project Data',
                'files': [
                    'external_logs/cursor/cursor_project_progress_and_next_steps.md',
                    'external_logs/cursor/cli_wrapper_execution_plan.json'
                ],
                'type': 'project_files'
            }
        }

    def create_integration_structure(self):
        """Create directory structure for integrated data."""
        integration_base = self.base_dir / f"integrated_{self.timestamp}"

        directories = {
            'conversations': integration_base / '01_conversations',
            'new_conversations': integration_base / '01_conversations' / 'new_only',
            'duplicate_conversations': integration_base / '01_conversations' / 'duplicates',
            'project_files': integration_base / '02_project_files',
            'session_data': integration_base / '03_session_data',
            'code_artifacts': integration_base / '04_code_artifacts',
            'documentation': integration_base / '05_documentation',
            'reports': integration_base / '06_reports'
        }

        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return integration_base, directories

    def process_conversation_json_with_deduplication(self, source_file, output_dir, source_name):
        """Process conversation JSON files with intelligent deduplication."""
        try:
            print(f"🔄 Processing {source_file} for deduplication...")

            with open(source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different conversation formats
            conversations = []
            if isinstance(data, list):
                conversations = data
            elif isinstance(data, dict) and 'conversations' in data:
                conversations = data['conversations']
            elif isinstance(data, dict) and 'data' in data:
                conversations = data['data']

            print(f"📊 Found {len(conversations)} conversations in {source_file}")

            new_conversations = []
            duplicate_conversations = []

            for conv in conversations:
                if not isinstance(conv, dict):
                    continue

                self.integration_results['deduplication_stats']['total_conversations'] += 1

                # Check for duplicates using conversation_id
                conv_id = conv.get('conversation_id')
                is_duplicate = False
                duplicate_reason = None

                if conv_id and conv_id in self.existing_conversation_ids:
                    is_duplicate = True
                    duplicate_reason = f"conversation_id: {conv_id}"

                # Check for duplicates using content hash
                if not is_duplicate:
                    content_hash = self.create_conversation_hash(conv)
                    if content_hash in self.existing_conversation_hashes:
                        is_duplicate = True
                        duplicate_reason = f"content_hash: {content_hash}"

                if is_duplicate:
                    duplicate_conversations.append({
                        'conversation': conv,
                        'reason': duplicate_reason
                    })
                    self.integration_results['deduplication_stats']['duplicate_conversations'] += 1
                    self.integration_results['deduplication_stats']['duplicate_identifiers'].append({
                        'conversation_id': conv_id,
                        'title': conv.get('title', 'Unknown'),
                        'reason': duplicate_reason
                    })
                else:
                    new_conversations.append(conv)
                    self.integration_results['deduplication_stats']['new_conversations'] += 1

            # Save new conversations
            if new_conversations:
                new_file = output_dir['new_conversations'] / f"{source_name}_new_conversations.json"
                with open(new_file, 'w', encoding='utf-8') as f:
                    json.dump(new_conversations, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved {len(new_conversations)} new conversations to {new_file}")

            # Save duplicate conversations for review
            if duplicate_conversations:
                duplicate_file = output_dir['duplicate_conversations'] / f"{source_name}_duplicate_conversations.json"
                with open(duplicate_file, 'w', encoding='utf-8') as f:
                    json.dump(duplicate_conversations, f, indent=2, ensure_ascii=False)
                print(f"⚠️  Found {len(duplicate_conversations)} duplicate conversations saved to {duplicate_file}")

            # Save original file for reference
            original_file = output_dir['conversations'] / f"{source_name}_{Path(source_file).name}"
            shutil.copy2(source_file, original_file)

            return True, {
                'new_conversations': len(new_conversations),
                'duplicate_conversations': len(duplicate_conversations),
                'new_file': str(new_file) if new_conversations else None,
                'duplicate_file': str(duplicate_file) if duplicate_conversations else None
            }

        except Exception as e:
            return False, str(e)

    def process_project_files(self, source_file, output_dir, source_name):
        """Process project files (code, documentation, etc.)."""
        try:
            file_path = Path(source_file)

            # Determine category based on file extension
            if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.sh', '.yml', '.yaml']:
                dest_dir = output_dir['code_artifacts'] / source_name
            elif file_path.suffix in ['.md', '.txt']:
                dest_dir = output_dir['documentation'] / source_name
            elif file_path.suffix == '.json':
                dest_dir = output_dir['project_files'] / source_name
            else:
                dest_dir = output_dir['project_files'] / source_name

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / file_path.name
            shutil.copy2(source_file, dest_file)

            return True, dest_file
        except Exception as e:
            return False, str(e)

    def process_session_data(self, source_file, output_dir, source_name):
        """Process session data files."""
        try:
            dest_dir = output_dir['session_data'] / source_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / Path(source_file).name
            shutil.copy2(source_file, dest_file)

            return True, dest_file
        except Exception as e:
            return False, str(e)

    def integrate_all_sources(self):
        """Integrate all external sources with deduplication."""
        print(f"\n{'='*80}")
        print(f"🚀 Starting External Logs Integration with Deduplication")
        print(f"{'='*80}")

        integration_base, dirs = self.create_integration_structure()
        sources = self.define_integration_sources()

        print(f"📁 Integration directory: {integration_base}")
        print(f"🔍 Deduplication: {len(self.existing_conversation_ids)} existing conversation IDs loaded")

        for source_key, source_info in sources.items():
            print(f"\n📂 Processing: {source_info['name']}")
            print(f"{'─' * 60}")

            source_results = {
                'files_processed': 0,
                'files_successful': 0,
                'files_failed': 0,
                'errors': [],
                'deduplication_results': {}
            }

            for file_path in source_info['files']:
                self.integration_results['total_files'] += 1
                source_results['files_processed'] += 1

                if not Path(file_path).exists():
                    error_msg = f"File not found: {file_path}"
                    source_results['errors'].append(error_msg)
                    source_results['files_failed'] += 1
                    print(f"❌ {error_msg}")
                    continue

                try:
                    if source_info['type'] == 'conversation_json':
                        success, result = self.process_conversation_json_with_deduplication(
                            file_path, dirs, source_key
                        )
                        if success:
                            source_results['deduplication_results'][Path(file_path).name] = result
                    elif source_info['type'] == 'project_files':
                        success, result = self.process_project_files(
                            file_path, dirs, source_key
                        )
                    elif source_info['type'] == 'session_data':
                        success, result = self.process_session_data(
                            file_path, dirs, source_key
                        )

                    if success:
                        source_results['files_successful'] += 1
                        self.integration_results['successful_integrations'] += 1
                        self.integration_results['files_integrated'].append({
                            'source': source_key,
                            'file': file_path,
                            'destination': str(result) if not isinstance(result, dict) else str(result.get('new_file', result.get('duplicate_file')))
                        })
                        print(f"✅ {Path(file_path).name}")
                    else:
                        source_results['files_failed'] += 1
                        source_results['errors'].append(result)
                        print(f"❌ {Path(file_path).name}: {result}")

                except Exception as e:
                    error_msg = f"Error processing {file_path}: {e}"
                    source_results['errors'].append(error_msg)
                    source_results['files_failed'] += 1
                    print(f"❌ {error_msg}")

            self.integration_results['sources_processed'][source_key] = source_results

            print(f"📊 {source_info['name']}: {source_results['files_successful']}/{source_results['files_processed']} successful")

            # Show deduplication results for conversation sources
            if source_info['type'] == 'conversation_json':
                for file_name, dedup_result in source_results['deduplication_results'].items():
                    print(f"   📄 {file_name}: {dedup_result['new_conversations']} new, {dedup_result['duplicate_conversations']} duplicates")

        # Save integration report
        self.save_integration_report(integration_base)

        return integration_base

    def save_integration_report(self, integration_base):
        """Save integration report with deduplication statistics."""
        report_path = integration_base / '06_reports' / 'integration_report.json'

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.integration_results, f, indent=2)

        # Create markdown summary
        summary_path = integration_base / 'README.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# External Logs Integration Report with Deduplication\n\n")
            f.write(f"**Integration completed:** {self.timestamp}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total files processed: {self.integration_results['total_files']}\n")
            f.write(f"- Successful integrations: {self.integration_results['successful_integrations']}\n")
            f.write(f"- Failed integrations: {self.integration_results['total_files'] - self.integration_results['successful_integrations']}\n\n")

            f.write(f"## Deduplication Statistics\n\n")
            dedup_stats = self.integration_results['deduplication_stats']
            f.write(f"- Total conversations processed: {dedup_stats['total_conversations']}\n")
            f.write(f"- New conversations: {dedup_stats['new_conversations']}\n")
            f.write(f"- Duplicate conversations: {dedup_stats['duplicate_conversations']}\n")
            f.write(f"- Duplication rate: {(dedup_stats['duplicate_conversations'] / max(dedup_stats['total_conversations'], 1) * 100):.1f}%\n\n")

            f.write(f"## Sources Processed\n\n")
            for source_key, results in self.integration_results['sources_processed'].items():
                f.write(f"### {source_key}\n")
                f.write(f"- Files processed: {results['files_processed']}\n")
                f.write(f"- Successful: {results['files_successful']}\n")
                f.write(f"- Failed: {results['files_failed']}\n\n")

                # Show deduplication results for conversation sources
                if results.get('deduplication_results'):
                    f.write(f"#### Deduplication Results:\n")
                    for file_name, dedup_result in results['deduplication_results'].items():
                        f.write(f"- {file_name}: {dedup_result['new_conversations']} new, {dedup_result['duplicate_conversations']} duplicates\n")
                    f.write("\n")

            f.write(f"## Directory Structure\n\n")
            f.write("```\n")
            f.write("📁 integrated_[timestamp]/\n")
            f.write("├── 01_conversations/\n")
            f.write("│   ├── new_only/           # New conversations only\n")
            f.write("│   └── duplicates/         # Duplicate conversations for review\n")
            f.write("├── 02_project_files/       # Project files and artifacts\n")
            f.write("├── 03_session_data/        # Session logs and events\n")
            f.write("├── 04_code_artifacts/      # Code files and scripts\n")
            f.write("├── 05_documentation/       # Documentation and guides\n")
            f.write("└── 06_reports/             # Integration reports\n")
            f.write("```\n\n")

            f.write(f"## Next Steps\n\n")
            f.write(f"1. **Review new conversations** in `01_conversations/new_only/`\n")
            f.write(f"2. **Check duplicates** in `01_conversations/duplicates/` if needed\n")
            f.write(f"3. **Process new conversations** with the main pipeline:\n")
            f.write(f"   ```bash\n")
            f.write(f"   python scripts/master_process.py -i integrated_{self.timestamp}/01_conversations/new_only/chatgpt_external_new_conversations.json --operations all\n")
            f.write(f"   ```\n")

def main():
    parser = argparse.ArgumentParser(
        description="Integrate external logs with intelligent deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--output-dir', default='.',
                       help='Base directory for output (default: current directory)')

    args = parser.parse_args()

    # Run integration with deduplication
    integrator = ExternalLogsIntegratorWithDeduplication(args.output_dir)
    integration_base = integrator.integrate_all_sources()

    # Print summary
    dedup_stats = integrator.integration_results['deduplication_stats']
    print(f"\n{'='*80}")
    print(f"✅ Integration Complete!")
    print(f"   Total files: {integrator.integration_results['total_files']}")
    print(f"   Successful: {integrator.integration_results['successful_integrations']}")
    print(f"   Total conversations: {dedup_stats['total_conversations']}")
    print(f"   New conversations: {dedup_stats['new_conversations']}")
    print(f"   Duplicate conversations: {dedup_stats['duplicate_conversations']}")
    print(f"   Duplication rate: {(dedup_stats['duplicate_conversations'] / max(dedup_stats['total_conversations'], 1) * 100):.1f}%")
    print(f"   Output directory: {integration_base}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
