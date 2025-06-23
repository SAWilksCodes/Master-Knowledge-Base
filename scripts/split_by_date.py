import sys
import io

# Fix Unicode issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json
import os
from pathlib import Path
from datetime import datetime
import argparse
import shutil

def get_conversation_date(convo):
    """Extract date from conversation, trying multiple timestamp fields."""
    timestamp = None

    # Try different timestamp fields
    for field in ['create_time', 'created_at', 'timestamp', 'update_time']:
        if field in convo and convo[field]:
            timestamp = convo[field]
            break

    if timestamp:
        try:
            # Handle Unix timestamps
            if isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp)
            # Handle ISO format
            else:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            pass

    return None

def split_by_date(input_dir, output_dir, copy_files=True):
    """Split conversations into Year/Month folders."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    json_files = list(input_path.glob('*.json'))
    print(f"📅 Processing {len(json_files)} conversations by date...")

    stats = {}
    no_date_count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                convo = json.load(f)

            date = get_conversation_date(convo)

            if date:
                # Create year/month folder structure
                year_month = date.strftime('%Y/%m')
                date_folder = output_path / year_month
                date_folder.mkdir(parents=True, exist_ok=True)

                # Update stats
                stats[year_month] = stats.get(year_month, 0) + 1

                # Copy or move file
                dest_file = date_folder / json_file.name
                if copy_files:
                    shutil.copy2(json_file, dest_file)
                else:
                    shutil.move(str(json_file), str(dest_file))
            else:
                no_date_count += 1
                # Put dateless files in 'undated' folder
                undated_folder = output_path / 'undated'
                undated_folder.mkdir(exist_ok=True)
                dest_file = undated_folder / json_file.name
                if copy_files:
                    shutil.copy2(json_file, dest_file)
                else:
                    shutil.move(str(json_file), str(dest_file))

        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")

    # Print summary
    print(f"\n✅ Date-based split complete!")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"\n📊 Conversations by month:")

    for year_month in sorted(stats.keys()):
        print(f"   {year_month}: {stats[year_month]} conversations")

    if no_date_count > 0:
        print(f"   undated: {no_date_count} conversations")

def main():
    parser = argparse.ArgumentParser(
        description="Split conversations by date into Year/Month folders"
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with JSON files")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for organized files")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")

    args = parser.parse_args()
    split_by_date(args.input_dir, args.output_dir, copy_files=not args.move)

if __name__ == "__main__":
    main()
