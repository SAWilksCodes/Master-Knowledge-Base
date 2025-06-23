import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullIngestionPipeline:
    """Complete pipeline for ingesting, processing, and storing conversation data."""

    def __init__(self, base_dir: str, output_dir: str = None):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir) if output_dir else self.base_dir / f"pipeline_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(exist_ok=True)

        # Pipeline stages
        self.stages = {
            'split': 'Split conversations to individual files',
            'parse': 'Deep parse conversations with enhanced analysis',
            'ids': 'Generate comprehensive IDs for all entities',
            'chroma': 'Migrate data to ChromaDB',
            'graph': 'Construct knowledge graph',
            'retrieval': 'Set up retrieval system',
            'visualization': 'Create visualization layer'
        }

        # Stage results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'errors': [],
            'warnings': []
        }

        # Configuration
        self.config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'chroma_persist_dir': str(self.output_dir / 'chroma_db'),
            'graph_output_dir': str(self.output_dir / 'knowledge_graph'),
            'visualization_dir': str(self.output_dir / 'visualization'),
            'batch_size': 100,
            'max_workers': 4
        }

    def log_stage_result(self, stage: str, success: bool, message: str = "", error: str = None):
        """Log the result of a pipeline stage."""
        self.results['stages'][stage] = {
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'error': error
        }

        if success:
            logger.info(f"✅ {stage}: {message}")
        else:
            logger.error(f"❌ {stage}: {error}")
            self.results['errors'].append(f"{stage}: {error}")

    def run_script(self, script_name: str, args: List[str], stage: str) -> bool:
        """Run a Python script and capture results."""
        script_path = self.base_dir / 'scripts' / script_name

        if not script_path.exists():
            error_msg = f"Script not found: {script_path}"
            self.log_stage_result(stage, False, error=error_msg)
            return False

        cmd = [sys.executable, str(script_path)] + args
        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                self.log_stage_result(stage, True, f"Script completed successfully")
                logger.info(f"Output: {result.stdout}")
                return True
            else:
                error_msg = f"Script failed with return code {result.returncode}: {result.stderr}"
                self.log_stage_result(stage, False, error=error_msg)
                logger.error(f"Error output: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            error_msg = f"Script timed out after 1 hour"
            self.log_stage_result(stage, False, error=error_msg)
            return False
        except Exception as e:
            error_msg = f"Failed to run script: {str(e)}"
            self.log_stage_result(stage, False, error=error_msg)
            return False

    def stage_split_conversations(self, input_file: str) -> bool:
        """Stage 1: Split conversations to individual files."""
        logger.info("🚀 Stage 1: Splitting conversations...")

        # Create output directories
        split_json_dir = self.output_dir / '01_split_json'
        split_markdown_dir = self.output_dir / '02_split_markdown'
        split_json_dir.mkdir(exist_ok=True)
        split_markdown_dir.mkdir(exist_ok=True)

        # Split to JSON
        success1 = self.run_script('split_conversations.py', [
            '-i', input_file,
            '-o', str(split_json_dir),
            '-v'
        ], 'split_json')

        # Split to Markdown
        success2 = self.run_script('split_conversations.py', [
            '-i', input_file,
            '-o', str(split_markdown_dir),
            '-f', 'markdown',
            '-v'
        ], 'split_markdown')

        return success1 and success2

    def stage_deep_parse(self) -> bool:
        """Stage 2: Deep parse conversations with enhanced analysis."""
        logger.info("🚀 Stage 2: Deep parsing conversations...")

        split_json_dir = self.output_dir / '01_split_json'
        parsed_dir = self.output_dir / '03_parsed_data'
        parsed_dir.mkdir(exist_ok=True)

        if not split_json_dir.exists():
            error_msg = "Split JSON directory not found. Run split stage first."
            self.log_stage_result('parse', False, error=error_msg)
            return False

        # Run enhanced deep parsing
        success = self.run_script('deep_parse_conversations_enhanced.py', [
            '-i', str(split_json_dir),
            '-o', str(parsed_dir),
            '-p', '*.json'
        ], 'parse')

        return success

    def stage_generate_ids(self) -> bool:
        """Stage 3: Generate comprehensive IDs for all entities."""
        logger.info("🚀 Stage 3: Generating comprehensive IDs...")

        parsed_dir = self.output_dir / '03_parsed_data'
        ids_dir = self.output_dir / '04_generated_ids'
        ids_dir.mkdir(exist_ok=True)

        if not parsed_dir.exists():
            error_msg = "Parsed data directory not found. Run parse stage first."
            self.log_stage_result('ids', False, error=error_msg)
            return False

        # Run comprehensive ID generation
        success = self.run_script('generate_all_ids.py', [
            '--input-dir', str(parsed_dir),
            '--output-dir', str(ids_dir),
            '--all-types'
        ], 'ids')

        return success

    def stage_chromadb_migration(self) -> bool:
        """Stage 4: Migrate data to ChromaDB."""
        logger.info("🚀 Stage 4: Migrating to ChromaDB...")

        ids_dir = self.output_dir / '04_generated_ids'

        if not ids_dir.exists():
            error_msg = "Generated IDs directory not found. Run IDs stage first."
            self.log_stage_result('chroma', False, error=error_msg)
            return False

        # Run ChromaDB migration
        success = self.run_script('chromadb_migration.py', [
            '--data-dir', str(ids_dir),
            '--chroma-dir', self.config['chroma_persist_dir'],
            '--embedding-model', self.config['embedding_model']
        ], 'chroma')

        return success

    def stage_knowledge_graph(self) -> bool:
        """Stage 5: Construct knowledge graph."""
        logger.info("🚀 Stage 5: Constructing knowledge graph...")

        ids_dir = self.output_dir / '04_generated_ids'
        graph_dir = Path(self.config['graph_output_dir'])
        graph_dir.mkdir(exist_ok=True)

        if not ids_dir.exists():
            error_msg = "Generated IDs directory not found. Run IDs stage first."
            self.log_stage_result('graph', False, error=error_msg)
            return False

        # Run knowledge graph construction
        success = self.run_script('build_knowledge_graph.py', [
            '--input-dir', str(ids_dir),
            '--output-dir', str(graph_dir),
            '--graph-type', 'comprehensive'
        ], 'graph')

        return success

    def stage_retrieval_system(self) -> bool:
        """Stage 6: Set up retrieval system."""
        logger.info("🚀 Stage 6: Setting up retrieval system...")

        chroma_dir = self.config['chroma_persist_dir']
        graph_dir = self.config['graph_output_dir']
        retrieval_dir = self.output_dir / '06_retrieval_system'
        retrieval_dir.mkdir(exist_ok=True)

        if not Path(chroma_dir).exists():
            error_msg = "ChromaDB directory not found. Run ChromaDB stage first."
            self.log_stage_result('retrieval', False, error=error_msg)
            return False

        # Run retrieval system setup
        success = self.run_script('setup_retrieval_system.py', [
            '--chroma-dir', chroma_dir,
            '--graph-dir', graph_dir,
            '--output-dir', str(retrieval_dir),
            '--embedding-model', self.config['embedding_model']
        ], 'retrieval')

        return success

    def stage_visualization(self) -> bool:
        """Stage 7: Create visualization layer."""
        logger.info("🚀 Stage 7: Creating visualization layer...")

        ids_dir = self.output_dir / '04_generated_ids'
        graph_dir = self.config['graph_output_dir']
        viz_dir = Path(self.config['visualization_dir'])
        viz_dir.mkdir(exist_ok=True)

        if not ids_dir.exists():
            error_msg = "Generated IDs directory not found. Run IDs stage first."
            self.log_stage_result('visualization', False, error=error_msg)
            return False

        # Run visualization generation
        success = self.run_script('create_visualization.py', [
            '--input-dir', str(ids_dir),
            '--graph-dir', graph_dir,
            '--output-dir', str(viz_dir),
            '--viz-type', 'comprehensive'
        ], 'visualization')

        return False  # Placeholder - script doesn't exist yet

    def run_pipeline(self, input_file: str, stages: List[str] = None) -> bool:
        """Run the complete pipeline or specified stages."""
        if stages is None:
            stages = list(self.stages.keys())

        logger.info(f"🎯 Starting pipeline with stages: {stages}")
        logger.info(f"📁 Output directory: {self.output_dir}")

        # Validate input file
        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return False

        # Run stages
        stage_functions = {
            'split': lambda: self.stage_split_conversations(input_file),
            'parse': self.stage_deep_parse,
            'ids': self.stage_generate_ids,
            'chroma': self.stage_chromadb_migration,
            'graph': self.stage_knowledge_graph,
            'retrieval': self.stage_retrieval_system,
            'visualization': self.stage_visualization
        }

        success_count = 0
        total_count = len(stages)

        for stage in stages:
            if stage not in stage_functions:
                logger.warning(f"Unknown stage: {stage}")
                continue

            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🚀 Running stage: {stage}")
                logger.info(f"{'='*60}")

                success = stage_functions[stage]()
                if success:
                    success_count += 1
                else:
                    logger.error(f"Stage {stage} failed. Pipeline may be incomplete.")
                    # Continue with next stage unless it's a critical dependency
                    if stage in ['split', 'parse']:
                        logger.error("Critical stage failed. Stopping pipeline.")
                        break

            except Exception as e:
                error_msg = f"Unexpected error in stage {stage}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self.results['errors'].append(error_msg)

        # Save pipeline results
        self.save_pipeline_results()

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Pipeline completed: {success_count}/{total_count} stages successful")
        logger.info(f"📁 Output directory: {self.output_dir}")

        if self.results['errors']:
            logger.error(f"❌ Errors encountered: {len(self.results['errors'])}")
            for error in self.results['errors']:
                logger.error(f"  - {error}")

        return success_count == total_count

    def save_pipeline_results(self):
        """Save pipeline results to JSON file."""
        results_file = self.output_dir / 'pipeline_results.json'

        # Add summary statistics
        self.results['summary'] = {
            'total_stages': len(self.stages),
            'successful_stages': sum(1 for stage in self.results['stages'].values() if stage['success']),
            'failed_stages': sum(1 for stage in self.results['stages'].values() if not stage['success']),
            'output_directory': str(self.output_dir)
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 Pipeline results saved to: {results_file}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'timestamp': self.results['timestamp'],
            'stages': self.results['stages'],
            'summary': {
                'total_stages': len(self.stages),
                'successful_stages': sum(1 for stage in self.results['stages'].values() if stage['success']),
                'failed_stages': sum(1 for stage in self.results['stages'].values() if not stage['success']),
                'errors': len(self.results['errors'])
            }
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Full ingestion pipeline for conversation data')
    parser.add_argument('--input', required=True, help='Input conversation file')
    parser.add_argument('--output-dir', help='Output directory (default: auto-generated)')
    parser.add_argument('--stages', nargs='+', choices=['split', 'parse', 'ids', 'chroma', 'graph', 'retrieval', 'visualization'],
                       help='Specific stages to run (default: all)')
    parser.add_argument('--base-dir', default='.', help='Base directory containing scripts')
    parser.add_argument('--config', help='Configuration file (JSON)')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = FullIngestionPipeline(args.base_dir, args.output_dir)

    # Load configuration if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                pipeline.config.update(config)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    # Run pipeline
    success = pipeline.run_pipeline(args.input, args.stages)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
