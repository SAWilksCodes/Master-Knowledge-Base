import sys
import io

# Fix Unicode issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

class ChatGPTExportProcessor:
    """Master script to run all ChatGPT export processing operations."""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {
            'timestamp': self.timestamp,
            'operations': {}
        }

    def run_script(self, script_name, args, operation_name):
        """Run a processing script and capture results."""
        print(f"\n{'='*60}")
        print(f"🚀 Running: {operation_name}")
        print(f"{'='*60}")

        cmd = [sys.executable, script_name] + args
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ {operation_name} completed successfully!")
                self.results['operations'][operation_name] = {
                    'status': 'success',
                    'output': result.stdout
                }
            else:
                print(f"❌ {operation_name} failed!")
                print(f"Error: {result.stderr}")
                self.results['operations'][operation_name] = {
                    'status': 'failed',
                    'error': result.stderr
                }

            return result.returncode == 0

        except Exception as e:
            print(f"❌ Failed to run {script_name}: {e}")
            self.results['operations'][operation_name] = {
                'status': 'error',
                'error': str(e)
            }
            return False

    def create_output_structure(self):
        """Create organized output directory structure."""
        output_base = self.base_dir / f"processed_{self.timestamp}"

        directories = {
            'split_json': output_base / '01_split_json',
            'split_markdown': output_base / '02_split_markdown',
            'by_date': output_base / '03_organized_by_date',
            'by_topic': output_base / '04_organized_by_topic',
            'by_model': output_base / '05_organized_by_model',
            'long_conversations': output_base / '06_long_conversations',
            'extracted_data': output_base / '07_extracted_data',
            'message_pairs': output_base / '08_message_pairs',
            'training_data': output_base / '09_training_data',
            'reports': output_base / '10_reports'
        }

        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return output_base, directories

    def process_all(self, conversations_file, operations):
        """Run all selected processing operations."""
        output_base, dirs = self.create_output_structure()

        print(f"\n📁 Output directory: {output_base}")
        print(f"📄 Processing: {conversations_file}")

        # Save initial setup info
        self.results['input_file'] = str(conversations_file)
        self.results['output_directory'] = str(output_base)

        success_count = 0
        total_count = 0

        # 1. Split conversations to JSON (required for most operations)
        if 'split' in operations or 'all' in operations:
            total_count += 1
            if self.run_script('split_conversations.py', [
                '-i', conversations_file,
                '-o', str(dirs['split_json']),
                '-v'
            ], 'Split to JSON'):
                success_count += 1
            else:
                print("⚠️  Split to JSON failed - some operations may not work!")

        # 2. Split conversations to Markdown
        if 'split' in operations or 'all' in operations:
            total_count += 1
            if self.run_script('split_conversations.py', [
                '-i', conversations_file,
                '-o', str(dirs['split_markdown']),
                '-f', 'markdown',
                '-v'
            ], 'Split to Markdown'):
                success_count += 1

        # Operations that require split JSON files
        json_source = dirs['split_json']

        # 3. Extract message pairs
        if 'pairs' in operations or 'all' in operations:
            total_count += 2
            # JSONL format
            if self.run_script('extract_pairs.py', [
                '-d', str(json_source),
                '-o', str(dirs['message_pairs'] / 'all_pairs.jsonl'),
                '-v'
            ], 'Extract Message Pairs (JSONL)'):
                success_count += 1

            # Markdown format
            if self.run_script('extract_pairs.py', [
                '-d', str(json_source),
                '-o', str(dirs['message_pairs'] / 'all_pairs.md'),
                '-f', 'markdown',
                '-v'
            ], 'Extract Message Pairs (Markdown)'):
                success_count += 1

        # 4. Organize by date
        if 'organize' in operations or 'all' in operations:
            total_count += 1
            if self.run_script('split_by_date.py', [
                '-i', str(json_source),
                '-o', str(dirs['by_date'])
            ], 'Organize by Date'):
                success_count += 1

        # 5. Organize by topic
        if 'organize' in operations or 'all' in operations:
            total_count += 1
            if self.run_script('split_by_topic.py', [
                '-i', str(json_source),
                '-o', str(dirs['by_topic'])
            ], 'Organize by Topic'):
                success_count += 1

        # 6. Organize by model
        if 'organize' in operations or 'all' in operations:
            total_count += 1
            if self.run_script('split_by_model.py', [
                '-i', str(json_source),
                '-o', str(dirs['by_model'])
            ], 'Organize by Model'):
                success_count += 1

        # 7. Extract long conversations
        if 'analysis' in operations or 'all' in operations:
            total_count += 1
            if self.run_script('extract_long_conversations.py', [
                '-i', str(json_source),
                '-o', str(dirs['long_conversations']),
                '--min-messages', '20'
            ], 'Extract Long Conversations'):
                success_count += 1

        # 8. Extract code blocks
        if 'extract' in operations or 'all' in operations:
            total_count += 1
            if self.run_script('extract_code.py', [
                '-i', str(json_source),
                '-o', str(dirs['extracted_data'] / 'all_code.md'),
                '-f', 'markdown'
            ], 'Extract Code Blocks'):
                success_count += 1

        # 9. Create searchable index
        if 'index' in operations or 'all' in operations:
            total_count += 1
            if self.run_script('create_index.py', [
                '-i', str(json_source),
                '-o', str(dirs['reports'] / 'conversation_index.csv')
            ], 'Create Searchable Index'):
                success_count += 1

        # 10. Extract Q&A pairs for training
        if 'training' in operations or 'all' in operations:
            total_count += 3
            # JSONL format
            if self.run_script('extract_qa_pairs.py', [
                '-i', str(json_source),
                '-o', str(dirs['training_data'] / 'qa_pairs.jsonl'),
                '-f', 'jsonl'
            ], 'Extract Q&A Pairs (JSONL)'):
                success_count += 1

            # Alpaca format
            if self.run_script('extract_qa_pairs.py', [
                '-i', str(json_source),
                '-o', str(dirs['training_data'] / 'qa_pairs_alpaca.json'),
                '-f', 'alpaca'
            ], 'Extract Q&A Pairs (Alpaca)'):
                success_count += 1

            # Markdown format
            if self.run_script('extract_qa_pairs.py', [
                '-i', str(json_source),
                '-o', str(dirs['training_data'] / 'qa_pairs.md'),
                '-f', 'markdown'
            ], 'Extract Q&A Pairs (Markdown)'):
                success_count += 1

        # Save processing report
        self.save_report(output_base, success_count, total_count)

        return success_count, total_count

    def save_report(self, output_base, success_count, total_count):
        """Save a processing report."""
        report_path = output_base / '10_reports' / 'processing_report.json'

        self.results['summary'] = {
            'total_operations': total_count,
            'successful_operations': success_count,
            'failed_operations': total_count - success_count
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

        # Also create a markdown summary
        summary_path = output_base / 'README.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# ChatGPT Export Processing Results\n\n")
            f.write(f"**Processed on:** {self.timestamp}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total operations: {total_count}\n")
            f.write(f"- Successful: {success_count}\n")
            f.write(f"- Failed: {total_count - success_count}\n\n")
            f.write(f"## Directory Structure\n\n")
            f.write("```\n")
            f.write("📁 processed_[timestamp]/\n")
            f.write("├── 01_split_json/          # Individual conversations (JSON)\n")
            f.write("├── 02_split_markdown/      # Individual conversations (Markdown)\n")
            f.write("├── 03_organized_by_date/   # Conversations by year/month\n")
            f.write("├── 04_organized_by_topic/  # Conversations by detected topic\n")
            f.write("├── 05_organized_by_model/  # Conversations by GPT model\n")
            f.write("├── 06_long_conversations/  # Conversations with 20+ messages\n")
            f.write("├── 07_extracted_data/      # Code blocks and other extracts\n")
            f.write("├── 08_message_pairs/       # User/Assistant pairs\n")
            f.write("├── 09_training_data/       # Q&A pairs for fine-tuning\n")
            f.write("└── 10_reports/             # Index and processing reports\n")
            f.write("```\n\n")
            f.write("## Next Steps\n\n")
            f.write("1. Check each directory for the processed data\n")
            f.write("2. Review `10_reports/conversation_index.csv` for searchable index\n")
            f.write("3. Use message pairs for AnythingLLM import\n")
            f.write("4. Use training data for fine-tuning projects\n")

def main():
    parser = argparse.ArgumentParser(
        description="Master script for processing ChatGPT exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Operations:
  all        - Run all processing operations
  split      - Split conversations into individual files
  pairs      - Extract message pairs
  organize   - Organize by date, topic, and model
  extract    - Extract code blocks
  analysis   - Extract long conversations
  index      - Create searchable index
  training   - Extract Q&A pairs for training

Examples:
  # Run everything
  python master_process.py -i conversations.json --operations all

  # Just split and extract pairs
  python master_process.py -i conversations.json --operations split pairs

  # Run multiple specific operations
  python master_process.py -i conversations.json --operations split organize index
        """
    )

    parser.add_argument('-i', '--input', required=True,
                       help='Path to conversations.json')
    parser.add_argument('--operations', nargs='+', required=True,
                       choices=['all', 'split', 'pairs', 'organize', 'extract',
                               'analysis', 'index', 'training'],
                       help='Operations to perform')
    parser.add_argument('--output-dir', default='.',
                       help='Base directory for output (default: current directory)')

    args = parser.parse_args()

    # Check if required scripts exist
    required_scripts = [
        'split_conversations.py', 'extract_pairs.py', 'split_by_date.py',
        'split_by_topic.py', 'split_by_model.py', 'extract_long_conversations.py',
        'extract_code.py', 'create_index.py', 'extract_qa_pairs.py'
    ]

    missing_scripts = []
    for script in required_scripts:
        if not Path(script).exists():
            missing_scripts.append(script)

    if missing_scripts:
        print("❌ Missing required scripts:")
        for script in missing_scripts:
            print(f"   - {script}")
        print("\nPlease ensure all processing scripts are in the same directory.")
        sys.exit(1)

    # Run processing
    processor = ChatGPTExportProcessor(args.output_dir)
    success, total = processor.process_all(args.input, args.operations)

    print(f"\n{'='*60}")
    print(f"✅ Processing Complete!")
    print(f"   Successful operations: {success}/{total}")
    print(f"   Output directory: processed_{processor.timestamp}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
