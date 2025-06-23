#!/usr/bin/env python3
"""
Runner script for the Semantic Anchor System
"""

import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Import and run the semantic anchor system
from semantic_anchor_system import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error running Semantic Anchor System: {e}")
        import traceback
        traceback.print_exc()
