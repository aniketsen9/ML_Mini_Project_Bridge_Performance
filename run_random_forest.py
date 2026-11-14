#!/usr/bin/env python3
"""
Convenience entry point to run the Random Forest Classifier from repository root.
"""
import os
import sys

# Add Final Project Code/Code to Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(repo_root, "Final Project Code", "Code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from random_forest import main

if __name__ == "__main__":
    main()
