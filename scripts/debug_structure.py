import json
import sys
from pathlib import Path

def analyze_conversation_structure(file_path):
    """Analyze the structure of a conversation file to understand why pairs aren't being extracted."""

    print(f"\n{'='*60}")
    print(f"Analyzing: {file_path}")
    print('='*60)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Show top-level keys
    print("\nTop-level keys:")
    for key in list(data.keys())[:20]:  # Show first 20 keys
        value_type = type(data[key]).__name__
        print(f"  - {key}: {value_type}")

    # Look for messages in common locations
    print("\nLooking for messages...")

    # Check direct messages key
    if 'messages' in data:
        messages = data['messages']
        print(f"  ✓ Found 'messages' key with {len(messages)} items")
        if messages and len(messages) > 0:
            print("\n  First message structure:")
            first_msg = messages[0]
            for k, v in first_msg.items():
                print(f"    - {k}: {type(v).__name__} = {str(v)[:50]}...")

    # Check mapping structure
    if 'mapping' in data:
        mapping = data['mapping']
        print(f"  ✓ Found 'mapping' key with {len(mapping)} nodes")

        # Find a node with a message
        for node_id, node in list(mapping.items())[:5]:
            if 'message' in node and node['message']:
                print("\n  Sample message node structure:")
                msg = node['message']
                for k, v in msg.items():
                    if k == 'content':
                        print(f"    - {k}: {type(v).__name__}")
                        if isinstance(v, dict):
                            for ck, cv in v.items():
                                print(f"      - {ck}: {type(cv).__name__}")
                        elif isinstance(v, list) and len(v) > 0:
                            print(f"      - First item type: {type(v[0]).__name__}")
                            if isinstance(v[0], dict):
                                for ck, cv in v[0].items():
                                    print(f"        - {ck}: {type(cv).__name__}")
                    else:
                        print(f"    - {k}: {type(v).__name__} = {str(v)[:50]}...")
                break

    # Check if it's a list
    if isinstance(data, list):
        print(f"  ✓ Data is a list with {len(data)} items")
        if data and len(data) > 0:
            print("\n  First item structure:")
            first_item = data[0]
            for k, v in first_item.items():
                print(f"    - {k}: {type(v).__name__}")

def main():
    if len(sys.argv) < 2:
        # If no file specified, analyze the first file found
        split_dir = Path("./split_chats")
        json_files = list(split_dir.glob("*.json"))
        if json_files:
            analyze_conversation_structure(json_files[0])
            print(f"\n\nFound {len(json_files)} total files. Analyzing first one.")
            print("To analyze a specific file, run: python debug_structure.py path/to/file.json")
        else:
            print("No JSON files found in ./split_chats")
    else:
        # Analyze specified file
        analyze_conversation_structure(sys.argv[1])

if __name__ == "__main__":
    main()
