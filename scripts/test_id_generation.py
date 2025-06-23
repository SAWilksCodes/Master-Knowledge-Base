#!/usr/bin/env python3
"""
Test ID Generation
Runs the ID generation scripts on a small subset of data to test functionality.
"""

import subprocess
import sys
from pathlib import Path
import os

def run_script(script_name, args):
    """Run a Python script with arguments."""
    cmd = [sys.executable, script_name] + args
    print(f"\n🚀 Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    print("🧪 Testing ID Generation Scripts")
    print("=" * 50)

    # Test directory (using a subset of data)
    test_input_dir = "test_fixed_scripts/models/gpt-4o"
    test_output_dir = "test_id_generation"

    # Create output directory
    Path(test_output_dir).mkdir(exist_ok=True)

    # Test 1: Generate Conversation IDs
    print("\n📋 Test 1: Generating Conversation IDs")
    success1 = run_script("generate_conversation_ids.py", [
        "-i", test_input_dir,
        "-o", f"{test_output_dir}/conversation_ids.csv"
    ])

    if not success1:
        print("❌ Conversation ID generation failed. Stopping.")
        return

    # Test 2: Generate Pair IDs
    print("\n📋 Test 2: Generating Pair IDs")
    success2 = run_script("generate_pair_ids.py", [
        "-i", test_input_dir,
        "-c", f"{test_output_dir}/conversation_ids.csv",
        "-o", f"{test_output_dir}/pair_ids.csv"
    ])

    if not success2:
        print("❌ Pair ID generation failed. Stopping.")
        return

    # Test 3: Generate Topic IDs
    print("\n📋 Test 3: Generating Topic IDs")
    success3 = run_script("generate_topic_ids.py", [
        "-i", test_input_dir,
        "-c", f"{test_output_dir}/conversation_ids.csv",
        "-o", f"{test_output_dir}/topic_ids.csv"
    ])

    if not success3:
        print("❌ Topic ID generation failed. Stopping.")
        return

    # Test 4: Generate Code Block IDs
    print("\n📋 Test 4: Generating Code Block IDs")
    success4 = run_script("generate_code_block_ids.py", [
        "-i", test_input_dir,
        "-c", f"{test_output_dir}/conversation_ids.csv",
        "-o", f"{test_output_dir}/code_block_ids.csv"
    ])

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    results = [
        ("Conversation IDs", success1),
        ("Pair IDs", success2),
        ("Topic IDs", success3),
        ("Code Block IDs", success4)
    ]

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")

    # Show generated files
    print(f"\n📁 Generated files in: {test_output_dir}")
    if Path(test_output_dir).exists():
        for file in Path(test_output_dir).glob("*"):
            if file.is_file():
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")

    print("\n🎉 ID generation test complete!")

if __name__ == "__main__":
    main()
