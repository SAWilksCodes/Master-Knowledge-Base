#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix Unicode issues in all scripts for Windows compatibility.
This script will update all the processing scripts to handle Unicode properly on Windows.
"""

import os
import re
from pathlib import Path

# Add this to the beginning of each script
UNICODE_FIX_HEADER = """import sys
import io

# Fix Unicode issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

"""

def fix_script(script_path):
    """Add Unicode fix to a Python script."""
    print(f"Fixing {script_path}...")

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if fix already applied
    if 'io.TextIOWrapper' in content:
        print(f"  ✓ Already fixed")
        return False

    # Find the first import statement or after shebang/encoding
    lines = content.split('\n')
    insert_position = 0

    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('#'):
            # Found first non-comment line
            insert_position = i
            break
        elif line.startswith('import') or line.startswith('from'):
            insert_position = i
            break

    # Insert the Unicode fix
    lines.insert(insert_position, UNICODE_FIX_HEADER.strip())

    # Write back
    fixed_content = '\n'.join(lines)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"  ✓ Fixed!")
    return True

def main():
    """Fix all processing scripts."""
    scripts_to_fix = [
        'split_conversations.py',
        'extract_pairs.py',
        'split_by_date.py',
        'split_by_topic.py',
        'split_by_model.py',
        'extract_long_conversations.py',
        'extract_code.py',
        'create_index.py',
        'extract_qa_pairs.py',
        'master_process.py'
    ]

    fixed_count = 0

    print("🔧 Fixing Unicode issues in scripts for Windows compatibility...\n")

    for script in scripts_to_fix:
        if Path(script).exists():
            if fix_script(script):
                fixed_count += 1
        else:
            print(f"⚠️  {script} not found")

    print(f"\n✅ Fixed {fixed_count} scripts!")
    print("\nYou can now run the scripts without Unicode errors.")
    print("Try running: python master_process.py -i conversations.json --operations all")

if __name__ == "__main__":
    main()
